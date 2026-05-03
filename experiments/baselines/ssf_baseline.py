"""SSF (Strategic Selection and Forgetting) baseline wrapped as StreamingModel.

Self-contained wrapper around the OFFICIAL SSF implementation from:
https://github.com/xinchen930/SSF-Strategic-Selection-and-Forgetting

All core classes and functions (AE, AE_classifier, InfoNCELoss, mask
optimisation, sample selection, drift detection) are copy-pasted from the
official repo to avoid import-path issues.  The StreamingModel adapter
(SSFBaseline) uses AE_classifier for both datasets since it is the more
complete model (encoder + decoder + sigmoid classifier head).

Official hyperparameter defaults:
    tem=0.02, bs=128, drift_threshold=0.05, lwf_lambda=0.5,
    new_sample_weight=100.0, opt_new_lr=50.0, opt_old_lr=1.0,
    SGD lr=0.001, initial training epochs=4, online epoch=1.
"""
import math
import time
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import WINDOW_SIZE

# ============================================================================
# Official SSF classes & functions (copied from utils.py / ssf.py)
# ============================================================================


class AE(nn.Module):
    """Autoencoder (encoder + decoder) -- used for NSL-KDD in original code."""

    def __init__(self, input_dim):
        super(AE, self).__init__()
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))
        second_fourth_layer_size = nearest_power_of_2 // 2
        third_layer_size = nearest_power_of_2 // 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class AE_classifier(nn.Module):
    """Autoencoder + binary classifier head (used for UNSW in original code).

    forward(x) -> (encode, decode, classify)
    classify is sigmoid output in [0, 1].
    """

    def __init__(self, input_dim):
        super(AE_classifier, self).__init__()
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))
        second_fourth_layer_size = nearest_power_of_2 // 2
        third_layer_size = nearest_power_of_2 // 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify


class InfoNCELoss(nn.Module):
    """Contrastive InfoNCE loss operating on decoder (reconstruction) output."""

    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # create mask -- remove self-similarity on the diagonal
        logits_mask = (torch.ones_like(mask).to(self.device)
                       - torch.eye(batch_size).to(self.device))
        logits_without_ii = logits * logits_mask

        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:, (labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:, (labels > 0).squeeze()]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        return loss


# ------------ Mask initialisation helper ------------------------------------

def _initialize_tensor(size, initialization, device):
    """Create a learnable mask parameter in a given range."""
    if initialization == '0-1':
        return torch.nn.Parameter(torch.rand(size, device=device), requires_grad=True)
    elif initialization == '0-0.5':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5, requires_grad=True)
    elif initialization == '0.5-1':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5 + 0.5, requires_grad=True)
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")


# ------------ Mask optimisation (official) ----------------------------------

def optimize_old_mask(control_res, treatment_res, device,
                      initialization='0.5-1', num_bins=10, lr=1.0, steps=100):
    """Optimise M_c (old-data selection mask) via KL-divergence matching."""
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)
    M_c = _initialize_tensor(control_res.size(0), initialization, device)
    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)
        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            bin_obs_c[i] = torch.sum(M_c * mask_c.float()) / torch.sum(M_c)
            bin_tgt_c[i] = treatment_hist[i] / len(treatment_res)

        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()

        loss = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction='sum')
        loss.backward()
        optimizer.step()

    return M_c


def optimize_new_mask(control_res, treatment_res, M_c, device,
                      initialization='0-0.5', num_bins=10, lr=50.0, steps=100):
    """Optimise M_t (new-data selection mask) via KL-divergence matching."""
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)
    M_t = _initialize_tensor(treatment_res.size(0), initialization, device)
    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)
        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_tgt_t = torch.zeros(num_bins, device=device)
        bin_combined = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            mask_t = (treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])
            bin_tgt_t[i] = treatment_hist[i] / len(treatment_res)
            bin_combined[i] = ((torch.sum(M_t * mask_t.float())
                                + torch.sum(M_c * mask_c.float()))
                               / (torch.sum(M_t) + torch.sum(M_c)))

        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()

        loss = F.kl_div(bin_combined.log(), bin_tgt_t, reduction='sum')
        loss.backward()
        optimizer.step()

    return M_t


# ------------ Drift detection (official) ------------------------------------

def detect_drift(new_data, control_data, window_size, drift_threshold):
    """KS-test based drift detection on classifier logits."""
    new_np = new_data.cpu().numpy() if isinstance(new_data, torch.Tensor) else np.asarray(new_data)
    ctrl_np = control_data.cpu().numpy() if isinstance(control_data, torch.Tensor) else np.asarray(control_data)
    for i in range(0, len(new_np), window_size):
        window_data = new_np[i:i + window_size]
        if len(window_data) < window_size:
            break
        _, p_value = ks_2samp(ctrl_np, window_data)
        if p_value < drift_threshold:
            return True
    return False


# ------------ Representative sample selection (official) --------------------

def select_and_update_representative_samples(
        x_train_this_epoch, y_train_this_epoch,
        x_test_this_epoch, y_test_this_epoch,
        M_c, M_t, num_labeled_sample, device):
    """Official non-drift sample selection: drop non-representative old, add representative new."""
    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False
    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        remove_indices = non_representative_old_indices
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]
        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[
            torch.randperm(len(non_representative_old_indices))[:num_to_remove]]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False
    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]
        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()].tolist())
        available_indices = list(set(range(len(x_test_this_epoch))) - selected_indices)
        available_indices = torch.tensor(available_indices, dtype=torch.long)
        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()],
                                y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask,
                                 torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask


def select_and_update_representative_samples_when_drift(
        x_train_this_epoch, y_train_this_epoch,
        x_test_this_epoch, y_test_this_epoch,
        M_c, M_t, num_labeled_sample, device,
        buffer_memory_size, model):
    """Official drift-path sample selection: remove all non-representative, fill buffer with pseudo-labels."""
    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False
    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    # Remove all non-representative samples
    remove_indices = non_representative_old_indices

    if len(non_representative_old_indices) < num_to_remove:
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]
        remove_indices = torch.cat([remove_indices, additional_remove_indices])

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False
    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]
        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()].tolist())
        available_indices = list(set(range(len(x_test_this_epoch))) - selected_indices)
        available_indices = torch.tensor(available_indices, dtype=torch.long)
        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()],
                                y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask,
                                 torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
    y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

    # Fill buffer to capacity with pseudo-labelled samples
    if len(x_train_this_epoch) < buffer_memory_size:
        additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)

        if representative_new.shape[0] > num_labeled_sample:
            remaining_new_samples = representative_new[
                torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]]
            if remaining_new_samples.size(0) >= additional_samples_needed:
                pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
            else:
                pseudo_labeled_samples = remaining_new_samples
                random_extra_needed = additional_samples_needed - remaining_new_samples.size(0)
                extra_idx = torch.randperm(len(x_test_this_epoch))[:random_extra_needed]
                pseudo_labeled_samples = torch.cat([pseudo_labeled_samples, x_test_this_epoch[extra_idx]], dim=0)
        else:
            extra_idx = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
            pseudo_labeled_samples = x_test_this_epoch[extra_idx]

        # Generate pseudo labels via the classifier head
        model.eval()
        with torch.no_grad():
            _, _, cls_out = model(pseudo_labeled_samples)
            pseudo_labels = (cls_out.squeeze(-1) > 0.5).long()
        if pseudo_labels.dim() == 0:
            pseudo_labels = pseudo_labels.unsqueeze(0)

        x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)
        y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.to(device)), dim=0)
        new_sample_mask = torch.cat([new_sample_mask,
                                     torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)])

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask


# ============================================================================
# StreamingModel wrapper
# ============================================================================

class SSFBaseline(StreamingModel):
    """SSF: Strategic Selection and Forgetting (official implementation wrapper).

    Uses AE_classifier for both datasets (encoder + decoder + sigmoid classifier).
    Drift detection via KS-test on classifier logits.
    Update follows the official algorithm: optimise M_c/M_t masks, select
    representative samples, retrain with weighted InfoNCE + weighted BCE +
    optional LwF distillation (no-drift path) or without distillation (drift path).
    """

    def __init__(
        self,
        tem: float = 0.02,
        bs: int = 128,
        drift_threshold: float = 0.05,
        lwf_lambda: float = 0.5,
        new_sample_weight: float = 100.0,
        opt_new_lr: float = 50.0,
        opt_old_lr: float = 1.0,
        num_labeled_sample: int = 200,
        init_epochs: int = 4,
        online_epochs: int = 1,
        sample_interval: int = WINDOW_SIZE,
    ):
        super().__init__('SSF')
        # Hyperparameters (official defaults)
        self.tem = tem
        self.bs = bs
        self.drift_threshold = drift_threshold
        self.lwf_lambda = lwf_lambda
        self.new_sample_weight = new_sample_weight
        self.opt_new_lr = opt_new_lr
        self.opt_old_lr = opt_old_lr
        self.num_labeled_sample = num_labeled_sample
        self.init_epochs = init_epochs
        self.online_epochs = online_epochs
        self.sample_interval = sample_interval

        # State (populated by initialize)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False

        # Losses
        self.infonce_criterion = None
        self.bce_criterion = nn.BCELoss(reduction='none')

        # Memory buffer (tensors on device)
        self.x_train_buf = None
        self.y_train_buf = None
        self.buffer_memory_size = None

        # Stored training logits for drift detection
        self._train_logits = None

        # Last drift state (so update knows which path to take)
        self._drift_state = False

        # Timing
        self._last_timing: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return self.scaler.transform(X).astype(np.float32)

    def _to_tensor(self, X_scaled: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X_scaled).to(self.device)

    def _store_train_logits(self):
        """Cache classifier logits on the current memory buffer for drift detection."""
        self.model.eval()
        with torch.no_grad():
            _, _, logits = self.model(self.x_train_buf)
            self._train_logits = logits.squeeze(-1)

    # ------------------------------------------------------------------ #
    # StreamingModel interface
    # ------------------------------------------------------------------ #

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        t0 = time.perf_counter()

        X_s = self._scale(X_init)
        input_dim = X_s.shape[1]

        # Build model & teacher
        self.model = AE_classifier(input_dim).to(self.device)
        self.teacher_model = AE_classifier(input_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.infonce_criterion = InfoNCELoss(self.device, self.tem)

        X_t = self._to_tensor(X_s)
        y_t = torch.LongTensor(y_init).to(self.device)

        # Initial training (official: 4 epochs)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(dataset=train_ds, batch_size=self.bs, shuffle=True)

        self.model.train()
        for epoch in range(self.init_epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                features, recon_vec, classifications = self.model(inputs)
                con_loss = self.infonce_criterion(recon_vec, labels)
                classification_loss = self.bce_criterion(classifications.squeeze(-1), labels.float())
                loss = con_loss.mean() + classification_loss.mean()
                loss.backward()
                self.optimizer.step()

        # Copy weights to teacher
        self.teacher_model.load_state_dict(self.model.state_dict())

        # Store training data as memory buffer
        self.x_train_buf = X_t.clone()
        self.y_train_buf = y_t.clone()
        self.buffer_memory_size = len(X_t)

        # Cache training logits
        self._store_train_logits()

        self._is_initialized = True
        self._last_timing['initialize_ms'] = (time.perf_counter() - t0) * 1000

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = self._to_tensor(X_s)
        self.model.eval()
        with torch.no_grad():
            _, _, classifications = self.model(X_t)
            preds = (classifications.squeeze(-1) > 0.5).long().cpu().numpy()
        # Handle scalar case (single sample)
        preds = np.atleast_1d(preds)
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def detect_drift(self, X: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = self._to_tensor(X_s)

        self.model.eval()
        with torch.no_grad():
            _, _, test_logits = self.model(X_t)
            test_logits = test_logits.squeeze(-1)

        # KS-test between training logits and test logits (official approach)
        drift = detect_drift(
            test_logits, self._train_logits,
            self.sample_interval, self.drift_threshold,
        )
        self._drift_state = drift

        # Compute a confidence value from the KS p-value
        test_np = test_logits.cpu().numpy()
        train_np = self._train_logits.cpu().numpy()
        _, p_value = ks_2samp(train_np, test_np)
        confidence = 1.0 - p_value if drift else 0.0

        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return drift, confidence

    def update(self, X: np.ndarray, y: np.ndarray):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = self._to_tensor(X_s)
        y_t = torch.LongTensor(y).to(self.device)

        # ----- Compute logits for mask optimisation -----
        self.model.eval()
        with torch.no_grad():
            _, _, test_logits = self.model(X_t)
            test_logits_np = test_logits.squeeze(-1).cpu().numpy()
        with torch.no_grad():
            _, _, train_logits = self.model(self.x_train_buf)
            train_logits_np = train_logits.squeeze(-1).cpu().numpy()

        control_res = train_logits_np
        treatment_res = test_logits_np

        # ----- Optimise masks -----
        M_c = optimize_old_mask(control_res, treatment_res, self.device,
                                initialization='0.5-1', lr=self.opt_old_lr)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, self.device,
                                initialization='0-0.5', lr=self.opt_new_lr)

        # ----- Select and update representative samples -----
        if self._drift_state:
            self.x_train_buf, self.y_train_buf, _, new_mask = \
                select_and_update_representative_samples_when_drift(
                    self.x_train_buf, self.y_train_buf,
                    X_t, y_t,
                    M_c, M_t,
                    self.num_labeled_sample, self.device,
                    self.buffer_memory_size, self.model,
                )
        else:
            self.x_train_buf, self.y_train_buf, _, new_mask = \
                select_and_update_representative_samples(
                    self.x_train_buf, self.y_train_buf,
                    X_t, y_t,
                    M_c, M_t,
                    self.num_labeled_sample, self.device,
                )

        # ----- Retrain -----
        train_ds = TensorDataset(self.x_train_buf, self.y_train_buf, new_mask)
        train_loader = DataLoader(dataset=train_ds, batch_size=self.bs, shuffle=True)

        self.model.train()
        for epoch in range(self.online_epochs):
            for inputs, labels, new_sample_mask in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                new_sample_mask = new_sample_mask.to(self.device)
                # normal_new_mask only applies to normal samples for InfoNCE
                normal_new_mask = new_sample_mask[labels == 0]

                self.optimizer.zero_grad()

                features, recon_vec, classifications = self.model(inputs)

                # Weighted InfoNCE loss
                con_loss = self.infonce_criterion(recon_vec, labels)
                weighted_con_loss = con_loss * ((1 - normal_new_mask) + normal_new_mask * self.new_sample_weight)

                # Weighted BCE loss
                classification_loss = self.bce_criterion(classifications.squeeze(-1), labels.float())
                weighted_classification_loss = classification_loss * (
                    (1 - new_sample_mask) + new_sample_mask * self.new_sample_weight)
                weighted_loss = weighted_con_loss.mean() + weighted_classification_loss.mean()

                if self._drift_state:
                    # Drift path: no distillation
                    total_loss = weighted_loss
                else:
                    # No-drift path: add LwF distillation
                    with torch.no_grad():
                        _, _, teacher_logits = self.teacher_model(inputs)
                    distillation_loss = F.mse_loss(classifications, teacher_logits)
                    total_loss = weighted_loss + self.lwf_lambda * distillation_loss

                total_loss.backward()
                self.optimizer.step()

        # Update teacher model after retraining
        self.teacher_model.load_state_dict(self.model.state_dict())

        # Refresh cached training logits
        self._store_train_logits()

        # Reset drift state
        self._drift_state = False

        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def get_timing(self) -> Dict[str, float]:
        return dict(self._last_timing)

    def reset(self):
        super().reset()
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.infonce_criterion = None
        self.x_train_buf = None
        self.y_train_buf = None
        self.buffer_memory_size = None
        self._train_logits = None
        self._drift_state = False
        self._last_timing = {}

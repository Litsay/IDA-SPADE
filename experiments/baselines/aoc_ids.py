"""AOC-IDS baseline: Official implementation wrapper.

Wraps the official AOC-IDS (xinchen930/AOC-IDS) into the StreamingModel
interface. All core components (AE, CRCLoss, evaluate with Gaussian PDF
fitting, score_detail) are copied directly from the official code to ensure
faithful reproduction.

Reference: Chen et al., "AOC-IDS: Autonomous Online Continual Learning
for Intrusion Detection Systems"
Official repo: https://github.com/xinchen930/AOC-IDS
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import scipy.optimize as opt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)

from ..streaming_interface import StreamingModel


# ---------------------------------------------------------------------------
# Official AOC-IDS components (copied from utils.py and online_training.py)
# ---------------------------------------------------------------------------

class AE(nn.Module):
    """Autoencoder from official AOC-IDS (utils.py).

    Architecture: input -> nearest_power_of_2//2 -> nearest_power_of_2//4
    (encoder), then reverse (decoder).
    """
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


class CRCLoss(nn.Module):
    """Contrastive Representation Concentration loss from official AOC-IDS.

    Key difference from InfoNCE: the denominator uses "TWO times of traversal"
    -- sum over ALL abnormal pairs globally, not per-row.
    """
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
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

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_mask = (torch.ones_like(mask).to(self.device)
                       - torch.eye(batch_size).to(self.device))
        logits_without_ii = logits * logits_mask

        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:, (labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:, (labels > 0).squeeze()]

        # TWO times of traversal (CRC): sum over ALL abnormal, not per-row
        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal))
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


def gaussian_pdf(x, mu, sigma):
    """Gaussian probability density function (from official utils.py)."""
    return ((1 / (np.sqrt(2 * np.pi) * sigma))
            * np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def log_likelihood(params, data):
    """Negative log-likelihood for a two-component Gaussian mixture
    (from official utils.py)."""
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2 + 1e-300))


def score_detail(y_test, y_test_pred, if_print=False):
    """Compute classification metrics (from official utils.py)."""
    if if_print:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        print('Accuracy ', accuracy_score(y_test, y_test_pred))
        print('Precision ', precision_score(y_test, y_test_pred, zero_division=0))
        print('Recall ', recall_score(y_test, y_test_pred, zero_division=0))
        print('F1 score ', f1_score(y_test, y_test_pred, zero_division=0))

    return (accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, zero_division=0),
            recall_score(y_test, y_test_pred, zero_division=0),
            f1_score(y_test, y_test_pred, zero_division=0))


def _evaluate_official(normal_temp, normal_recon_temp, x_train, y_train,
                       x_test, y_test, model):
    """Official evaluate() function from AOC-IDS utils.py.

    Fits two Gaussians (normal vs abnormal) to cosine similarities of
    encoder and decoder features, then classifies test samples based on
    PDF comparison. Uses voting between encoder and decoder predictions.

    When y_test is an integer (0), returns predictions only (no metrics).
    Otherwise returns (encoder_result, decoder_result, final_result).
    """
    # ---------- Encoder features ----------
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    # Guard: need both classes for Gaussian fitting
    if len(x_train_normal) == 0 or len(x_train_abnormal) == 0:
        # Fallback: predict all normal if no abnormal training data, or vice versa
        n_test = x_test.shape[0]
        default_label = 0 if len(x_train_abnormal) == 0 else 1
        dummy = np.full(n_test, default_label, dtype=np.int32)
        if isinstance(y_test, int):
            return dummy
        else:
            return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, _ = torch.sort(
        F.cosine_similarity(train_features,
                            normal_temp.reshape([-1, normal_temp.shape[0]]),
                            dim=1))
    values_features_normal, _ = torch.sort(
        F.cosine_similarity(train_features_normal,
                            normal_temp.reshape([-1, normal_temp.shape[0]]),
                            dim=1))
    values_features_abnormal, _ = torch.sort(
        F.cosine_similarity(train_features_abnormal,
                            normal_temp.reshape([-1, normal_temp.shape[0]]),
                            dim=1))

    values_features_all = values_features_all.cpu().detach().numpy()

    values_features_test = F.cosine_similarity(
        test_features,
        normal_temp.reshape([-1, normal_temp.shape[0]]))

    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy()) + 1e-6

    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy()) + 1e-6

    initial_params = np.array([mu1_initial, sigma1_initial,
                               mu2_initial, sigma2_initial])
    result = opt.minimize(log_likelihood, initial_params,
                          args=(values_features_all,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x

    # Ensure positive sigmas
    sigma1_fit = max(abs(sigma1_fit), 1e-6)
    sigma2_fit = max(abs(sigma2_fit), 1e-6)

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()
    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2 - pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) is False:
        if hasattr(y_test, 'device') and y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    # ---------- Decoder features ----------
    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, _ = torch.sort(
        F.cosine_similarity(train_recon,
                            normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                            dim=1))
    values_recon_normal, _ = torch.sort(
        F.cosine_similarity(train_recon_normal,
                            normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                            dim=1))
    values_recon_abnormal, _ = torch.sort(
        F.cosine_similarity(train_recon_abnormal,
                            normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                            dim=1))

    values_recon_all = values_recon_all.cpu().detach().numpy()

    values_recon_test = F.cosine_similarity(
        test_recon,
        normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy()) + 1e-6

    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy()) + 1e-6

    initial_params = np.array([mu3_initial, sigma3_initial,
                               mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params,
                          args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    sigma3_fit = max(abs(sigma3_fit), 1e-6)
    sigma4_fit = max(abs(sigma4_fit), 1e-6)

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()
    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4 - pdf3)).cpu().detach().numpy().astype("float32")

    # ---------- Voting: encoder vs decoder ----------
    y_test_pred_no_vote = torch.where(
        torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de),
        torch.from_numpy(y_test_pred_2),
        torch.from_numpy(y_test_pred_4))

    if not isinstance(y_test, int):
        if hasattr(y_test, 'device') and y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)
        result_final = score_detail(y_test, y_test_pred_no_vote)
        return result_encoder, result_decoder, result_final
    else:
        return y_test_pred_no_vote


# ---------------------------------------------------------------------------
# StreamingModel wrapper
# ---------------------------------------------------------------------------

class AOCIDSBaseline(StreamingModel):
    """AOC-IDS wrapped as a StreamingModel.

    Faithful to the official implementation:
    - AE with CRCLoss (encoder + decoder features)
    - Gaussian PDF fitting for classification (evaluate function)
    - Pseudo-label generation with 20% flip (noise injection)
    - SGD optimizer with lr=0.001
    - 4 initial epochs, 1 online epoch
    - Temperature 0.02, batch size 128
    """

    def __init__(self, tem=0.02, bs=128, flip_percent=0.2, lr=0.001,
                 epochs_init=4, epochs_online=1):
        super().__init__('AOC-IDS')
        self.tem = tem
        self.bs = bs
        self.flip_percent = flip_percent
        self.lr = lr
        self.epochs_init = epochs_init
        self.epochs_online = epochs_online

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False

        # Stored training data (needed for Gaussian fitting in evaluate)
        self.x_train_buf = None   # torch tensor on device
        self.y_train_buf = None   # torch tensor on device

        # Normal templates (mean of L2-normalized encoder/decoder features
        # for normal samples from the initial training set)
        self.normal_temp = None
        self.normal_recon_temp = None

        # Initial normal samples for template recomputation
        self.x_init_normal = None

        self._last_timing = {}

    # ---- Scaling ----

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
            return self.scaler.transform(X)
        return self.scaler.transform(X)

    # ---- Template computation ----

    def _compute_templates(self):
        """Recompute normal_temp and normal_recon_temp from initial normal
        samples, as done in the official code after every online step."""
        with torch.no_grad():
            self.model.eval()
            self.normal_temp = torch.mean(
                F.normalize(
                    self.model(self.x_init_normal)[0], p=2, dim=1),
                dim=0)
            self.normal_recon_temp = torch.mean(
                F.normalize(
                    self.model(self.x_init_normal)[1], p=2, dim=1),
                dim=0)

    # ---- CRC Training ----

    def _train_crc(self, x_train: torch.Tensor, y_train: torch.Tensor,
                   epochs: int):
        """Train the AE with CRCLoss on both encoder and decoder features."""
        ds = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset=ds, batch_size=self.bs, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for data in loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                features, recon_vec = self.model(inputs)

                # CRC loss on BOTH encoder and decoder features
                loss = (self.criterion(features, labels)
                        + self.criterion(recon_vec, labels))

                loss.backward()
                self.optimizer.step()

    # ---- StreamingModel interface ----

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        """Initial training phase (corresponds to "first round" in official
        online_training.py)."""
        t0 = time.perf_counter()

        X_s = self._scale(X_init)
        input_dim = X_s.shape[1]

        # Create model, optimizer, criterion
        self.model = AE(input_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = CRCLoss(self.device, self.tem)

        x_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y_init.astype(np.int64)).to(self.device)

        # Train for epochs_init epochs with CRCLoss
        self._train_crc(x_t, y_t, self.epochs_init)

        # Store training buffer
        self.x_train_buf = x_t.clone()
        self.y_train_buf = y_t.clone()

        # Store initial normal samples for template recomputation
        normal_mask = (y_t == 0).squeeze()
        if normal_mask.any():
            self.x_init_normal = x_t[normal_mask].clone()
        else:
            # Fallback: use all training data if no normals
            self.x_init_normal = x_t.clone()

        # Compute initial templates
        self._compute_templates()

        self._is_initialized = True
        self._last_timing['init_ms'] = (time.perf_counter() - t0) * 1000

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the official evaluate() logic.

        Uses Gaussian PDF fitting on encoder and decoder cosine similarities,
        with voting between encoder and decoder predictions.
        """
        t0 = time.perf_counter()

        X_s = self._scale(X)
        x_test = torch.FloatTensor(X_s).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Recompute templates (official code does this before every predict)
            self._compute_templates()

            # Use official evaluate function with y_test=0 to get predictions
            # only (no metric computation)
            predictions = _evaluate_official(
                self.normal_temp, self.normal_recon_temp,
                self.x_train_buf, self.y_train_buf,
                x_test, 0, self.model)

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return predictions.astype(np.int32)

    def detect_drift(self, X: np.ndarray) -> tuple:
        """AOC-IDS does not have explicit drift detection."""
        return (False, 0.0)

    def update(self, X: np.ndarray, y: np.ndarray):
        """Online update step (official AOC-IDS algorithm):

        1. Generate pseudo-labels on new data using evaluate()
        2. Flip 20% of pseudo-labels (noise injection)
        3. Append to training buffer
        4. Retrain with CRCLoss for 1 epoch
        5. Recompute normal templates
        """
        t0 = time.perf_counter()

        X_s = self._scale(X)
        x_new = torch.FloatTensor(X_s).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Recompute templates before pseudo-labeling
            self._compute_templates()

            # Generate pseudo-labels using official evaluate
            # y_train_detection includes original labels + all prior pseudo-labels
            pseudo_labels = _evaluate_official(
                self.normal_temp, self.normal_recon_temp,
                self.x_train_buf, self.y_train_buf,
                x_new, 0, self.model)

        if isinstance(pseudo_labels, torch.Tensor):
            pseudo_labels = pseudo_labels.cpu().numpy()
        pseudo_labels = pseudo_labels.astype(np.int64)

        # Store unflipped pseudo-labels for y_train_buf (detection buffer)
        # This matches official code: y_train_detection gets clean pseudo-labels,
        # while y_train_this_epoch gets flipped ones for training
        y_detection = torch.tensor(pseudo_labels.copy(),
                                   dtype=torch.long, device=self.device)

        # Flip flip_percent of pseudo-labels (noise injection, official code)
        num_flip = int(self.flip_percent * len(pseudo_labels))
        if num_flip > 0:
            flip_indices = np.random.choice(
                len(pseudo_labels), num_flip, replace=False)
            pseudo_labels[flip_indices] = 1 - pseudo_labels[flip_indices]

        y_new_flipped = torch.tensor(pseudo_labels,
                                     dtype=torch.long, device=self.device)

        # Append to training buffer (x_train_this_epoch in official code)
        x_train_extended = torch.cat([self.x_train_buf, x_new], dim=0)
        y_train_extended = torch.cat(
            [self.y_train_buf[:self.x_train_buf.shape[0]], y_new_flipped],
            dim=0)

        # Update the detection buffer with clean pseudo-labels
        self.y_train_buf = torch.cat([self.y_train_buf, y_detection], dim=0)
        self.x_train_buf = x_train_extended.clone()

        # Retrain with CRCLoss for epochs_online epochs
        self._train_crc(x_train_extended, y_train_extended, self.epochs_online)

        # Recompute templates after update
        self._compute_templates()

        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def get_timing(self) -> dict:
        """Return timing breakdown for the last operation in ms."""
        return dict(self._last_timing)

    def reset(self):
        """Clear all state for a fresh run."""
        super().reset()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.x_train_buf = None
        self.y_train_buf = None
        self.normal_temp = None
        self.normal_recon_temp = None
        self.x_init_normal = None
        self._last_timing = {}

"""FeCo baseline: StreamingModel wrapper around the OFFICIAL FeCo implementation.

Wraps the official FeCo (Federated Contrastive Learning) code from:
    https://github.com/ning-wang1/FeCo_federated-contrastive-learning

All essential classes (MLP, ProjectionHead, NCEAverage, NCECriterion) are
copy-pasted directly from the official repository to keep this file
self-contained.  The centralized (single-node) training loop and cosine-
similarity-based anomaly scoring follow the official centralized_main.py
and test.py exactly.

No drift detection -- FeCo centralized has none.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel

# ---------------------------------------------------------------------------
# Official FeCo constants
# ---------------------------------------------------------------------------
_EPS = 1e-7


# ===================================================================
# Copied from official FeCo: models/mlp.py
# ===================================================================
class MLP(nn.Module):
    """Official FeCo MLP encoder.

    Architecture: input -> 128 -> 256 -> output_size
    Returns both unnormalised and L2-normalised feature vectors.
    """

    def __init__(self, input_size=10, layer1_size=128,
                 layer2_size=256, output_size=64):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = F.relu(self.layer2(x))
        x = self.output(x)
        x = x.view(x.size(0), -1)
        normed_x = F.normalize(x, p=2, dim=1)
        return x, normed_x


class ProjectionHead(nn.Module):
    """Official FeCo projection head.

    Architecture: latent_dim -> 256 -> feature_dim, with L2 normalisation.
    """

    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(256, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(
                    m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ===================================================================
# Copied from official FeCo: nce_average.py
# ===================================================================
class NCEAverage(nn.Module):
    """Official NCE contrastive logit computation.

    Computes positive logits (normal-normal inner products, excluding
    self-pairs) and negative logits (normal-anomaly inner products),
    applies temperature scaling, and maintains a running normalisation
    constant *Z* with momentum.
    """

    def __init__(self, feature_dim, len_neg, len_pos,
                 tau, Z_momentum=0.9, Z=-1):
        super(NCEAverage, self).__init__()
        self.len_neg = len_neg
        self.len_pos = len_pos
        self.embed_dim = feature_dim
        self.register_buffer(
            'params', torch.tensor([float(Z), float(tau), float(Z_momentum)]))

    def nce_core(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.params[1].item())
        Z = self.params[0].item()
        if Z < 0:
            # initialise Z as mean of first batch
            self.params[0] = outs.mean() * self.len_neg
            Z = self.params[0].clone().detach().item()
        else:
            Z_new = outs.mean() * self.len_neg
            self.params[0] = ((1 - self.params[2]) * Z_new
                              + self.params[2] * self.params[0])
            Z = self.params[0].clone().detach().item()
        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)
        return outs, probs

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)
        return probs[:, 0].mean()

    def forward(self, n_vec, a_vec, idx_n, idx_a, normed_vec):
        n_scores = torch.mm(n_vec, n_vec.t())
        pos_logits = (n_scores[~torch.eye(n_scores.shape[0], dtype=bool)]
                      .reshape(n_vec.size(0), -1)
                      .view(-1, 1))
        n_a_scores = torch.mm(n_vec, a_vec.t())
        neg_logits = (n_a_scores
                      .repeat(1, (n_vec.size(0) - 1))
                      .view(pos_logits.size(0), -1))
        outs, probs = self.nce_core(pos_logits, neg_logits)
        return outs, probs


# ===================================================================
# Copied from official FeCo: nce_criteria.py
# ===================================================================
class NCECriterion(nn.Module):
    """Official NCE loss.

    loss = -[ log P(1|pos) + sum log P(0|neg) ] / batch_size
    where P uses a uniform noise distribution q = 1/len_neg.
    """

    def __init__(self, len_neg):
        super(NCECriterion, self).__init__()
        self.num_data = len_neg

    def forward(self, x):
        batch_size = x.size(0)
        k = x.size(1) - 1  # number of negative samples

        # uniform noise assumption
        q_noise = 1.0 / self.num_data

        # positive term
        p_p = x.select(1, 0)
        log_D1 = torch.div(p_p, p_p.add(k * q_noise + _EPS)).log_()

        # negative term
        p_n = x.narrow(1, 1, k)
        log_D0 = (torch.div(p_n.clone().fill_(k * q_noise),
                             p_n.add(k * q_noise + _EPS))
                  .log_())

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / batch_size
        return loss


# ===================================================================
# Utility (from official FeCo: utils/utils.py)
# ===================================================================
def _l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x ** 2, dim=dim).unsqueeze(dim))


# ===================================================================
# StreamingModel wrapper
# ===================================================================
class FeCoBaseline(StreamingModel):
    """FeCo (centralized) wrapped as a StreamingModel.

    Follows the official training loop from centralized_main.py:
    - MLP encoder (input -> 128 -> 256 -> latent_dim)
    - ProjectionHead (latent_dim -> 256 -> feature_dim)
    - NCEAverage + NCECriterion contrastive loss
    - Memory-bank based normal vector for cosine-similarity scoring
    - Threshold = percentile of normal validation scores

    No drift detection (centralized FeCo has none).
    """

    def __init__(self,
                 latent_dim: int = 64,
                 feature_dim: int = 128,
                 tau: float = 0.03,
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 memory_bank_size: int = 50,
                 n_train_batch_size: int = 5,
                 a_train_batch_size: int = 200,
                 init_epochs: int = 60,
                 threshold_percentile: float = 5.0,
                 Z_momentum: float = 0.9):
        super().__init__('FeCo')
        # hyper-parameters (official defaults from global_vars.py / centralized_main.py)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.tau = tau
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.memory_bank_size = memory_bank_size
        self.n_train_batch_size = n_train_batch_size
        self.a_train_batch_size = a_train_batch_size
        self.init_epochs = init_epochs
        self.threshold_percentile = threshold_percentile
        self.Z_momentum = Z_momentum

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # state -- populated in initialize()
        self.model = None
        self.model_head = None
        self.nce_average = None
        self.criterion = None
        self.optimizer = None
        self.memory_bank = []
        self.normal_vec = None
        self.threshold = 0.5
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return self.scaler.transform(X).astype(np.float32)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _split_normal_anomaly(self, X: np.ndarray, y: np.ndarray):
        """Split data into normal (y==0) and anomaly (y==1) subsets."""
        normal_mask = (y == 0)
        anomaly_mask = (y == 1)
        X_n = X[normal_mask]
        X_a = X[anomaly_mask]
        return X_n, X_a

    def _build_loaders(self, X_n: np.ndarray, X_a: np.ndarray):
        """Create DataLoader pairs for normal and anomaly data."""
        ds_n = torch.utils.data.TensorDataset(
            self._to_tensor(X_n),
            torch.arange(len(X_n), device=self.device))
        ds_a = torch.utils.data.TensorDataset(
            self._to_tensor(X_a),
            torch.arange(len(X_a), device=self.device))

        loader_n = torch.utils.data.DataLoader(
            ds_n, batch_size=self.n_train_batch_size, shuffle=True,
            drop_last=True)
        loader_a = torch.utils.data.DataLoader(
            ds_a, batch_size=self.a_train_batch_size, shuffle=True,
            drop_last=True)
        return loader_n, loader_a

    def _train_one_epoch(self, loader_n, loader_a):
        """One training epoch following the official train() function."""
        self.model.train()
        self.model_head.train()

        for (normal_data, idx_n), (anormal_data, idx_a) in zip(
                loader_n, loader_a):
            if normal_data.size(0) != self.n_train_batch_size:
                break

            data = torch.cat([normal_data, anormal_data], dim=0)

            # forward
            unnormed_vec, normed_vec = self.model(data)
            vec = self.model_head(unnormed_vec)

            n_vec = vec[:self.n_train_batch_size]
            a_vec = vec[self.n_train_batch_size:]

            outs, probs = self.nce_average(
                n_vec, a_vec, idx_n, idx_a,
                normed_vec[:self.n_train_batch_size])
            loss = self.criterion(outs)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update memory bank (official logic)
            self.model.eval()
            _, n = self.model(normal_data)
            n = n.detach()
            average = torch.mean(n, dim=0, keepdim=True)
            if len(self.memory_bank) < self.memory_bank_size:
                self.memory_bank.append(average)
            else:
                self.memory_bank.pop(0)
                self.memory_bank.append(average)
            self.model.train()

    def _compute_normal_vec(self):
        """Compute normal_vec as mean of memory bank vectors (L2-normalised)."""
        if len(self.memory_bank) == 0:
            # Fallback: unit vector matching MLP output dim (latent_dim)
            self.normal_vec = torch.zeros(1, self.latent_dim, device=self.device)
            self.normal_vec[0, 0] = 1.0
            return
        stacked = torch.cat(self.memory_bank, dim=0)
        mean_vec = torch.mean(stacked, dim=0, keepdim=True)
        self.normal_vec = _l2_normalize(mean_vec)

    def _compute_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Cosine similarity between test features and normal_vec."""
        self.model.eval()
        X_t = self._to_tensor(X_scaled)
        with torch.no_grad():
            _, normed = self.model(X_t)
            sim = torch.mm(normed, self.normal_vec.t()).squeeze(-1)
        return sim.cpu().numpy()

    def _calibrate_threshold(self, X_normal_scaled: np.ndarray):
        """Set threshold as a low percentile of normal-sample scores.

        Samples scoring *below* this threshold are classified as anomalous
        (they are far from the normal prototype).
        """
        if len(X_normal_scaled) == 0:
            self.threshold = 0.5
            return
        scores = self._compute_scores(X_normal_scaled)
        self.threshold = float(
            np.percentile(scores, self.threshold_percentile))

    # ------------------------------------------------------------------
    # StreamingModel interface
    # ------------------------------------------------------------------
    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        t0 = time.perf_counter()

        # scale
        X_s = self._scale(X_init)
        input_dim = X_s.shape[1]

        # split
        X_n, X_a = self._split_normal_anomaly(X_s, y_init)

        # handle edge-case: no anomaly samples
        if len(X_a) == 0:
            X_a = X_n[:1]  # dummy; NCE still needs negatives
        if len(X_n) < self.n_train_batch_size:
            # replicate so we have at least one batch
            reps = (self.n_train_batch_size // len(X_n)) + 1
            X_n = np.tile(X_n, (reps, 1))[:self.n_train_batch_size * 2]

        # build models (official architecture)
        self.model = MLP(
            input_size=input_dim,
            layer1_size=128,
            layer2_size=256,
            output_size=self.latent_dim,
        ).to(self.device)

        self.model_head = ProjectionHead(
            self.latent_dim, self.feature_dim
        ).to(self.device)

        # NCE components
        len_neg = len(X_a)
        len_pos = len(X_n)
        self.nce_average = NCEAverage(
            self.feature_dim, len_neg, len_pos,
            self.tau, self.Z_momentum
        ).to(self.device)
        self.criterion = NCECriterion(len_neg).to(self.device)

        # optimiser (official: SGD)
        all_params = (list(self.model.parameters())
                      + list(self.model_head.parameters()))
        self.optimizer = torch.optim.SGD(
            all_params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # data loaders
        loader_n, loader_a = self._build_loaders(X_n, X_a)

        # training loop
        self.memory_bank = []
        for _ in range(self.init_epochs):
            self._train_one_epoch(loader_n, loader_a)

        # compute normal vector from memory bank
        self._compute_normal_vec()

        # calibrate threshold on normal validation scores
        self._calibrate_threshold(X_n)

        self._is_initialized = True
        self._last_timing['init_ms'] = (time.perf_counter() - t0) * 1000

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X_s = self._scale(X)
        scores = self._compute_scores(X_s)
        # samples with similarity <= threshold are anomalous (label 1)
        preds = (scores <= self.threshold).astype(np.int64)
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def detect_drift(self, X: np.ndarray) -> tuple:
        """FeCo centralized has no drift detection."""
        return False, 0.0

    def update(self, X: np.ndarray, y: np.ndarray):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_n, X_a = self._split_normal_anomaly(X_s, y)

        # need both normal and anomaly samples for NCE training
        if len(X_n) < self.n_train_batch_size or len(X_a) == 0:
            # not enough data for a proper NCE batch; skip training,
            # but still update memory bank if we have normal samples
            if len(X_n) > 0:
                self.model.eval()
                with torch.no_grad():
                    _, normed = self.model(self._to_tensor(X_n))
                    avg = torch.mean(normed, dim=0, keepdim=True)
                if len(self.memory_bank) < self.memory_bank_size:
                    self.memory_bank.append(avg)
                else:
                    self.memory_bank.pop(0)
                    self.memory_bank.append(avg)
                self._compute_normal_vec()
            self._last_timing['training_ms'] = (
                (time.perf_counter() - t0) * 1000)
            return

        # rebuild NCE components for the new data sizes
        len_neg = len(X_a)
        len_pos = len(X_n)
        self.nce_average = NCEAverage(
            self.feature_dim, len_neg, len_pos,
            self.tau, self.Z_momentum
        ).to(self.device)
        self.criterion = NCECriterion(len_neg).to(self.device)

        loader_n, loader_a = self._build_loaders(X_n, X_a)

        # one epoch of NCE training (incremental update)
        self._train_one_epoch(loader_n, loader_a)

        # recompute normal vector
        self._compute_normal_vec()

        # recalibrate threshold on new normal data
        self._calibrate_threshold(X_n)

        self._last_timing['training_ms'] = (
            (time.perf_counter() - t0) * 1000)

    def get_timing(self) -> dict:
        return dict(self._last_timing)

    def reset(self):
        super().reset()
        self.model = None
        self.model_head = None
        self.nce_average = None
        self.criterion = None
        self.optimizer = None
        self.memory_bank = []
        self.normal_vec = None
        self.threshold = 0.5
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}

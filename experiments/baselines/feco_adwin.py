"""FeCo + ADWIN baseline: contrastive learning model with ADWIN drift detection.

FeCo is a federated contrastive learning model. In our single-node streaming
evaluation, we use the contrastive learning component + classifier, and mount
ADWIN for drift detection triggering retraining.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from river.drift import ADWIN
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import LEARNING_RATE, BATCH_SIZE, DROPOUT, ADWIN_DELTA


class ContrastiveEncoder(nn.Module):
    """Encoder with projection head for contrastive learning."""
    def __init__(self, input_dim, latent_dim=64, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        proj = self.projection(z)
        logits = self.classifier(z)
        return logits, F.normalize(proj, dim=1), z


class FeCoADWINBaseline(StreamingModel):
    """FeCo + ADWIN: contrastive learning with ADWIN drift-triggered retraining."""

    def __init__(self, adwin_delta=ADWIN_DELTA, contrastive_weight=0.3, temperature=0.5):
        super().__init__('FeCo+ADWIN')
        self.adwin_delta = adwin_delta
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.adwin = ADWIN(delta=adwin_delta)
        self._last_timing = {}
        self._drift_detected = False

    def _scale(self, X):
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return np.clip(self.scaler.transform(X), -5, 5)

    def _contrastive_loss(self, projections, labels):
        """Supervised contrastive loss."""
        n = len(labels)
        if n < 2:
            return torch.tensor(0.0, device=self.device)
        sim = torch.mm(projections, projections.t()) / self.temperature
        # Mask: same class pairs
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = labels_eq.float() - torch.eye(n, device=self.device)
        # Exclude self-similarity
        logits_mask = 1.0 - torch.eye(n, device=self.device)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        # Mean of positive pairs
        pos_count = mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = ContrastiveEncoder(X_s.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self._train(X_s, y_init, epochs=10)
        self._is_initialized = True

    def predict(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def update(self, X, y):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        if self._drift_detected:
            self._train(X_s, y, epochs=5)
            self._drift_detected = False
        else:
            # Light update
            self._train(X_s, y, epochs=1)
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0].cpu().numpy()

        detected = False
        for p in max_probs:
            self.adwin.update(float(p))
            if self.adwin.drift_detected:
                detected = True

        self._drift_detected = detected
        confidence = 1.0 if detected else 0.0
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def get_timing(self):
        return dict(self._last_timing)

    def _train(self, X_s, y, epochs=1):
        self.model.train()
        X_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_t)), shuffle=True)
        for _ in range(epochs):
            for bf, bl in dl:
                self.optimizer.zero_grad()
                logits, proj, _ = self.model(bf)
                cls_loss = nn.CrossEntropyLoss()(logits, bl)
                con_loss = self._contrastive_loss(proj, bl)
                loss = cls_loss + self.contrastive_weight * con_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def reset(self):
        super().reset()
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.adwin = ADWIN(delta=self.adwin_delta)
        self._drift_detected = False
        self._last_timing = {}

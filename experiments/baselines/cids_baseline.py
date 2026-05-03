"""CIDS baseline: Contrastive cross-entropy loss for static NID.

A static NID model using supervised contrastive learning loss combined
with standard cross-entropy. No continual learning or drift detection.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import HIDDEN_DIMS, LEARNING_RATE, BATCH_SIZE, DROPOUT


class CIDSModel(nn.Module):
    """MLP with contrastive projection head."""
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, n_classes=2,
                 proj_dim=32, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hd)])
            prev = hd
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)
        self.projector = nn.Linear(prev, proj_dim)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        proj = F.normalize(self.projector(feat), dim=1)
        return logits, proj


class CIDSBaseline(StreamingModel):
    """CIDS: Contrastive NID — static model, no drift detection or adaptation."""

    def __init__(self, contrastive_weight=0.3, temperature=0.5):
        super().__init__('CIDS')
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}

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
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = labels_eq.float() - torch.eye(n, device=self.device)
        logits_mask = 1.0 - torch.eye(n, device=self.device)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        pos_count = torch.clamp(mask.sum(dim=1), min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = CIDSModel(X_s.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self._train(X_s, y_init, epochs=10)
        self._is_initialized = True

    def predict(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def update(self, X, y):
        """Static model: minimal update, no CL mechanism."""
        t0 = time.perf_counter()
        X_s = self._scale(X)
        self._train(X_s, y, epochs=1)
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        # Static model: no drift detection
        return False, 0.0

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
                logits, proj = self.model(bf)
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
        self._last_timing = {}

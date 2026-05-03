"""CARD baseline: Cross-domain Adversarial Robust Detection.

Uses adversarial training (domain discriminator) to learn domain-invariant
features for robust NID across network environments. Static model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import HIDDEN_DIMS, LEARNING_RATE, BATCH_SIZE, DROPOUT


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CARDModel(nn.Module):
    """Feature extractor + classifier + domain discriminator."""
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, n_classes=2, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hd)])
            prev = hd
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)
        # Domain discriminator
        self.domain_disc = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),  # 2 domains: current vs historical
        )

    def forward(self, x, alpha=0.1):
        feat = self.features(x)
        logits = self.classifier(feat)
        # Gradient reversal for domain adversarial
        reversed_feat = GradientReversalFunction.apply(feat, alpha)
        domain_logits = self.domain_disc(reversed_feat)
        return logits, domain_logits, feat


class CARDBaseline(StreamingModel):
    """CARD: adversarial domain adaptation for NID. Static model."""

    def __init__(self, adv_weight=0.1):
        super().__init__('CARD')
        self.adv_weight = adv_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}
        self._prev_features = None

    def _scale(self, X):
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return np.clip(self.scaler.transform(X), -5, 5)

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = CARDModel(X_s.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self._train(X_s, y_init, epochs=10, use_adv=False)
        self._prev_features = X_s
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
        self._train(X_s, y, epochs=1, use_adv=True)
        self._prev_features = X_s
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        # Static model: no drift detection
        return False, 0.0

    def get_timing(self):
        return dict(self._last_timing)

    def _train(self, X_s, y, epochs=1, use_adv=False):
        self.model.train()
        X_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        # Create domain labels: current=0, historical=1
        domain_labels = torch.zeros(len(y_t), dtype=torch.long, device=self.device)
        if use_adv and self._prev_features is not None:
            prev_t = torch.FloatTensor(self._prev_features).to(self.device)
            n_prev = min(len(prev_t), len(X_t))
            if n_prev > 0:
                combined_X = torch.cat([X_t, prev_t[:n_prev]])
                combined_domain = torch.cat([
                    torch.zeros(len(X_t), dtype=torch.long, device=self.device),
                    torch.ones(n_prev, dtype=torch.long, device=self.device),
                ])
                combined_y = torch.cat([y_t, y_t[:n_prev]])  # labels not used for prev but needed for shape
            else:
                combined_X = X_t
                combined_domain = domain_labels
                combined_y = y_t
        else:
            combined_X = X_t
            combined_domain = domain_labels
            combined_y = y_t

        ds = torch.utils.data.TensorDataset(combined_X, combined_y, combined_domain)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(combined_X)), shuffle=True)
        for _ in range(epochs):
            for bf, bl, bd in dl:
                self.optimizer.zero_grad()
                logits, domain_logits, _ = self.model(bf)
                # Only use classification loss for current domain samples
                current_mask = bd == 0
                if current_mask.any():
                    cls_loss = nn.CrossEntropyLoss()(logits[current_mask], bl[current_mask])
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)
                if use_adv:
                    adv_loss = nn.CrossEntropyLoss()(domain_logits, bd)
                    loss = cls_loss + self.adv_weight * adv_loss
                else:
                    loss = cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def reset(self):
        super().reset()
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._prev_features = None
        self._last_timing = {}

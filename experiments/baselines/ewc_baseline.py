"""EWC (Elastic Weight Consolidation) baseline for continual learning NID."""
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import HIDDEN_DIMS, LEARNING_RATE, BATCH_SIZE, DROPOUT, EWC_LAMBDA


class EWCModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, n_classes=2, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hd)])
            prev = hd
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)
        self.fisher_dict = {}
        self.optpar_dict = {}

    def forward(self, x):
        return self.classifier(self.features(x))

    def compute_ewc_loss(self, ewc_lambda):
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return ewc_lambda * loss

    def update_fisher(self, data_loader, device):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out = self(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
        for n in fisher:
            fisher[n] /= max(len(data_loader), 1)
            # Accumulate with old fisher
            if n in self.fisher_dict:
                self.fisher_dict[n] = 0.5 * self.fisher_dict[n] + 0.5 * fisher[n]
            else:
                self.fisher_dict[n] = fisher[n]
        self.optpar_dict = {n: p.data.clone() for n, p in self.named_parameters()}


class EWCBaseline(StreamingModel):
    """Pure EWC continual learning baseline (no drift detection, just EWC regularization)."""

    def __init__(self, ewc_lambda=EWC_LAMBDA):
        super().__init__('EWC')
        self.ewc_lambda = ewc_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}
        self._task_count = 0

    def _scale(self, X):
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return np.clip(self.scaler.transform(X), -5, 5)

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = EWCModel(X_s.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self._train(X_s, y_init, epochs=5)
        self._update_fisher(X_s, y_init)
        self._is_initialized = True

    def predict(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = torch.argmax(self.model(X_t), dim=1).cpu().numpy()
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def update(self, X, y):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        self._train(X_s, y, epochs=1)
        # Update Fisher every 5 windows
        self._task_count += 1
        if self._task_count % 5 == 0:
            self._update_fisher(X_s, y)
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        # EWC has no drift detection - always returns False
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
                out = self.model(bf)
                loss = nn.CrossEntropyLoss()(out, bl)
                if self._task_count > 0:
                    loss += self.model.compute_ewc_loss(self.ewc_lambda)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def _update_fisher(self, X_s, y):
        X_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_t)))
        self.model.update_fisher(dl, self.device)

    def reset(self):
        super().reset()
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._task_count = 0
        self._last_timing = {}

"""LwF (Learning without Forgetting) baseline for continual learning NID."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import HIDDEN_DIMS, LEARNING_RATE, BATCH_SIZE, DROPOUT


class LwFBaseline(StreamingModel):
    """Learning without Forgetting: uses knowledge distillation from previous model."""

    def __init__(self, temperature=2.0, distill_weight=0.5):
        super().__init__('LwF')
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.old_model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._last_timing = {}
        self._window_count = 0

    def _build_model(self, input_dim):
        layers = []
        prev = input_dim
        for hd in HIDDEN_DIMS:
            layers.extend([nn.Linear(prev, hd), nn.ReLU(), nn.Dropout(DROPOUT), nn.BatchNorm1d(hd)])
            prev = hd
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers).to(self.device)

    def _scale(self, X):
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return np.clip(self.scaler.transform(X), -5, 5)

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = self._build_model(X_s.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self._train(X_s, y_init, epochs=5, use_distill=False)
        self._snapshot()
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
        self._train(X_s, y, epochs=1, use_distill=True)
        self._window_count += 1
        if self._window_count % 10 == 0:
            self._snapshot()
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        return False, 0.0

    def get_timing(self):
        return dict(self._last_timing)

    def _train(self, X_s, y, epochs=1, use_distill=False):
        self.model.train()
        X_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        # Get old model outputs for distillation
        old_logits = None
        if use_distill and self.old_model is not None:
            self.old_model.eval()
            with torch.no_grad():
                old_logits = self.old_model(X_t)

        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_t)), shuffle=True)

        for _ in range(epochs):
            for i, (bf, bl) in enumerate(dl):
                self.optimizer.zero_grad()
                new_logits = self.model(bf)
                # Task loss
                task_loss = nn.CrossEntropyLoss()(new_logits, bl)

                # Distillation loss
                if old_logits is not None:
                    start = i * BATCH_SIZE
                    end = min(start + len(bf), len(old_logits))
                    if start < len(old_logits):
                        old_batch = old_logits[start:end]
                        if len(old_batch) == len(bf):
                            soft_old = F.softmax(old_batch / self.temperature, dim=1)
                            soft_new = F.log_softmax(new_logits / self.temperature, dim=1)
                            distill_loss = F.kl_div(soft_new, soft_old, reduction='batchmean') * (self.temperature ** 2)
                            task_loss = (1 - self.distill_weight) * task_loss + self.distill_weight * distill_loss

                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def _snapshot(self):
        """Save current model as old model for future distillation."""
        import copy
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()

    def reset(self):
        super().reset()
        self.model = None
        self.old_model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._window_count = 0
        self._last_timing = {}

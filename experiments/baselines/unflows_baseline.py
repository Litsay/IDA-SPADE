"""unFlowS baseline: Pure static autoencoder-based NID (no ADWIN).

Static unsupervised model evaluated as a baseline without drift detection,
per paper Table 1 requirements.
"""
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import LEARNING_RATE, BATCH_SIZE, DROPOUT


class FlowAutoEncoder(nn.Module):
    """Autoencoder for flow-based anomaly detection."""
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class UnFlowsBaseline(StreamingModel):
    """Pure unFlowS: AE + classifier, no drift detection."""

    def __init__(self, recon_threshold_pct=95):
        super().__init__('unFlowS')
        self.recon_threshold_pct = recon_threshold_pct
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classifier = None
        self.cls_optimizer = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.recon_threshold = None
        self._last_timing = {}
        self._classifier_ready = False

    def _scale(self, X):
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        return np.clip(self.scaler.transform(X), -5, 5)

    def _train_ae(self, X_s, epochs=20):
        self.model.train()
        X_t = torch.FloatTensor(X_s).to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, X_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(X_t)), shuffle=True)
        for _ in range(epochs):
            for bx, _ in dl:
                self.optimizer.zero_grad()
                recon, _ = self.model(bx)
                loss = nn.MSELoss()(recon, bx)
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X_t)
            errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
            self.recon_threshold = np.percentile(errors, self.recon_threshold_pct)

    def _train_classifier(self, X_s, y, epochs=5):
        self.classifier.train()
        self.model.eval()
        X_t = torch.FloatTensor(X_s).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            _, z = self.model(X_t)
        ds = torch.utils.data.TensorDataset(z, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(BATCH_SIZE, len(z)), shuffle=True)
        for _ in range(epochs):
            for bz, bl in dl:
                self.cls_optimizer.zero_grad()
                out = self.classifier(bz)
                loss = nn.CrossEntropyLoss()(out, bl)
                loss.backward()
                self.cls_optimizer.step()

    def initialize(self, X_init, y_init):
        X_s = self._scale(X_init)
        self.model = FlowAutoEncoder(X_s.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self._train_ae(X_s, epochs=20)
        latent_dim = 16
        self.classifier = SimpleClassifier(latent_dim).to(self.device)
        self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=LEARNING_RATE)
        if len(np.unique(y_init)) >= 2:
            self._train_classifier(X_s, y_init, epochs=10)
            self._classifier_ready = True
        self._is_initialized = True

    def predict(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()
        if self._classifier_ready:
            self.classifier.eval()
            with torch.no_grad():
                _, z = self.model(X_t)
                logits = self.classifier(z)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
        else:
            with torch.no_grad():
                recon, _ = self.model(X_t)
                errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
            if self.recon_threshold is not None:
                preds = (errors > self.recon_threshold).astype(np.int64)
            else:
                preds = np.zeros(len(X), dtype=np.int64)
        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return preds

    def update(self, X, y):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        if not self._classifier_ready and len(np.unique(y)) >= 2:
            self._train_classifier(X_s, y, epochs=10)
            self._classifier_ready = True
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        return False, 0.0

    def get_timing(self):
        return dict(self._last_timing)

    def reset(self):
        super().reset()
        self.model = None
        self.classifier = None
        self.cls_optimizer = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.recon_threshold = None
        self._classifier_ready = False
        self._last_timing = {}

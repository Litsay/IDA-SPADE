"""unFlowS + ADWIN baseline: static autoencoder, retrain on ADWIN drift detection.

unFlowS is a purely static unsupervised model. In streaming evaluation,
we mount ADWIN as a post-hoc drift detector. When ADWIN fires, the model
is retrained on the current window.
"""
import numpy as np
import torch
import torch.nn as nn
import time
from river.drift import ADWIN
from sklearn.preprocessing import MinMaxScaler

from ..streaming_interface import StreamingModel
from ..config import LEARNING_RATE, BATCH_SIZE, DROPOUT, ADWIN_DELTA


class FlowAutoEncoder(nn.Module):
    """Autoencoder for flow-based anomaly detection (unFlowS-style)."""
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
    """Simple MLP classifier for supervised fallback."""
    def __init__(self, input_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class UnFlowsADWINBaseline(StreamingModel):
    """unFlowS + ADWIN: AE feature extraction + classifier, retrain on drift.

    Uses AE for representation learning, then a classifier on latent features.
    ADWIN monitors prediction confidence for drift detection.
    """

    def __init__(self, adwin_delta=ADWIN_DELTA, recon_threshold_pct=95):
        super().__init__('unFlowS+ADWIN')
        self.adwin_delta = adwin_delta
        self.recon_threshold_pct = recon_threshold_pct
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classifier = None
        self.cls_optimizer = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.adwin = ADWIN(delta=adwin_delta)
        self.recon_threshold = None
        self._last_timing = {}
        self._drift_detected = False

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
        # Set reconstruction threshold
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X_t)
            errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
            self.recon_threshold = np.percentile(errors, self.recon_threshold_pct)

    def _train_classifier(self, X_s, y, epochs=5):
        """Train classifier on AE latent features."""
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
        # Train classifier on AE latent features
        latent_dim = 16  # must match FlowAutoEncoder
        self.classifier = SimpleClassifier(latent_dim).to(self.device)
        self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=LEARNING_RATE)
        # Only train classifier if we have both classes
        if len(np.unique(y_init)) >= 2:
            self._train_classifier(X_s, y_init, epochs=10)
            self._classifier_ready = True
        else:
            self._classifier_ready = False
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
            # Fallback: use reconstruction error
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
        # Train classifier if not ready yet and we now have both classes
        if not self._classifier_ready and len(np.unique(y)) >= 2:
            self._train_classifier(X_s, y, epochs=10)
            self._classifier_ready = True
        # Retrain if ADWIN detected drift
        if self._drift_detected:
            self._train_ae(X_s, epochs=5)
            if len(np.unique(y)) >= 2:
                self._train_classifier(X_s, y, epochs=3)
            self._drift_detected = False
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    def detect_drift(self, X):
        t0 = time.perf_counter()
        X_s = self._scale(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.model.eval()

        detected = False
        if self._classifier_ready:
            self.classifier.eval()
            with torch.no_grad():
                _, z = self.model(X_t)
                logits = self.classifier(z)
                probs = torch.softmax(logits, dim=1)
                max_probs = probs.max(dim=1)[0].cpu().numpy()
            for p in max_probs:
                self.adwin.update(float(p))
                if self.adwin.drift_detected:
                    detected = True
        else:
            # Use reconstruction error for ADWIN
            with torch.no_grad():
                recon, _ = self.model(X_t)
                errors = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
            for e in errors:
                self.adwin.update(float(e))
                if self.adwin.drift_detected:
                    detected = True

        self._drift_detected = detected
        confidence = 1.0 if detected else 0.0
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

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
        self.adwin = ADWIN(delta=self.adwin_delta)
        self.recon_threshold = None
        self._drift_detected = False
        self._classifier_ready = False
        self._last_timing = {}

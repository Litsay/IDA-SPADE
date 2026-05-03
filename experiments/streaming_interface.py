"""Abstract interface for all streaming models in the experiment framework."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Window:
    """A single data window in the stream."""
    index: int
    X: np.ndarray              # feature matrix (n_samples, n_features)
    y_binary: np.ndarray       # binary labels (0/1)
    y_multiclass: np.ndarray   # multi-class labels (string array)
    metadata: Dict = field(default_factory=dict)  # day, anomaly_ratio, etc.


@dataclass
class WindowResult:
    """Result from processing a single window."""
    window_index: int
    predictions: np.ndarray
    true_labels: np.ndarray
    drift_detected: bool = False
    drift_confidence: float = 0.0
    timing: Dict = field(default_factory=dict)  # component-wise timing in ms
    extra: Dict = field(default_factory=dict)    # model-specific extra info


class StreamingModel(ABC):
    """Abstract base class for all streaming NID models."""

    def __init__(self, name: str):
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        """Initial training on the first batch of data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for a window."""
        pass

    def predict_evaluate(self, X: np.ndarray, y: np.ndarray):
        """Return (predictions, true_labels) at the model's natural evaluation granularity.

        Default: connection-level (same as predict + raw labels).
        Models operating at a different granularity (e.g. entity-level)
        should override this to return predictions and labels at that level.
        """
        return self.predict(X), y

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability scores for positive class. Override for AUC computation.

        Returns None if not supported (default). Shape should match predict_evaluate output.
        """
        return None

    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray):
        """Incremental update on a new window (Train step in Test-then-Train)."""
        pass

    @abstractmethod
    def detect_drift(self, X: np.ndarray) -> tuple:
        """Detect concept drift. Returns (detected: bool, confidence: float)."""
        pass

    @abstractmethod
    def get_timing(self) -> Dict[str, float]:
        """Return timing breakdown for the last window in ms."""
        pass

    def reset(self):
        """Reset model state for a fresh run."""
        self._is_initialized = False

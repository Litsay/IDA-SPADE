"""Prequential (Test-then-Train) evaluation with fading factor."""
import numpy as np
import time
from typing import Dict, List, Iterator, Optional
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .streaming_interface import StreamingModel, Window, WindowResult
from .config import FADING_FACTOR


@dataclass
class PrequentialMetrics:
    """Per-window metrics record."""
    window_index: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fading_f1: float
    auc: float
    anomaly_ratio: float
    drift_detected: bool
    drift_confidence: float
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def compute_fading_average(values: List[float], alpha: float = FADING_FACTOR) -> float:
    """Compute exponentially weighted fading average."""
    if not values:
        return 0.0
    n = len(values)
    weights = np.array([alpha ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, values) / weights.sum())


def prequential_evaluate(model: StreamingModel, stream: Iterator[Window],
                         X_init: np.ndarray, y_init: np.ndarray,
                         alpha: float = FADING_FACTOR,
                         verbose: bool = True) -> List[PrequentialMetrics]:
    """Run prequential (test-then-train) evaluation.

    1. Initialize model on X_init, y_init
    2. For each window: predict → record metrics → update
    """
    # Initialize
    model.initialize(X_init, y_init)

    results = []
    f1_history = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Prequential Evaluation: {model.name}")
        print(f"{'Win':<6} {'F1':<8} {'Fading F1':<11} {'Drift':<8} {'Anom%':<8}")
        print(f"{'-'*70}")

    for window in stream:
        t0 = time.perf_counter()

        # TEST: predict at model's natural evaluation granularity
        preds, y_true = model.predict_evaluate(window.X, window.y_binary)
        t_predict = (time.perf_counter() - t0) * 1000

        # Compute metrics
        if len(preds) != len(y_true):
            min_len = min(len(preds), len(y_true))
            preds = preds[:min_len]
            y_true = y_true[:min_len]

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=1)
        rec = recall_score(y_true, preds, zero_division=1)
        f1 = f1_score(y_true, preds, zero_division=1)
        f1_history.append(f1)
        fading_f1 = compute_fading_average(f1_history, alpha)

        # AUC: requires probability scores and both classes present
        auc = 1.0  # default for windows with single class
        scores = model.predict_proba(window.X) if hasattr(model, 'predict_proba') else None
        if scores is not None and len(scores) == len(y_true) and len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = 1.0
        elif len(np.unique(y_true)) <= 1:
            # Single-class window: perfect if predictions match
            auc = 1.0 if accuracy_score(y_true, preds) == 1.0 else 0.0

        # Detect drift
        t1 = time.perf_counter()
        drift_det, drift_conf = model.detect_drift(window.X)
        t_drift = (time.perf_counter() - t1) * 1000

        # TRAIN: update model on current window
        t2 = time.perf_counter()
        model.update(window.X, window.y_binary)
        t_train = (time.perf_counter() - t2) * 1000

        timing = model.get_timing()
        timing['predict_ms'] = t_predict
        timing['drift_detect_ms'] = t_drift
        timing['train_ms'] = t_train
        timing['total_ms'] = (time.perf_counter() - t0) * 1000

        metrics = PrequentialMetrics(
            window_index=window.index,
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            fading_f1=fading_f1, auc=auc,
            anomaly_ratio=window.metadata.get('anomaly_ratio', 0),
            drift_detected=drift_det,
            drift_confidence=drift_conf,
            timing=timing,
            metadata=window.metadata,
        )
        results.append(metrics)

        if verbose and (window.index % 10 == 0 or window.index < 5):
            drift_str = f"Y({drift_conf:.2f})" if drift_det else "N"
            print(f"{window.index:<6} {f1:<8.4f} {fading_f1:<11.4f} {drift_str:<8} "
                  f"{window.metadata.get('anomaly_ratio', 0)*100:<7.1f}%")

    if verbose:
        print(f"{'='*70}")
        final_fading = results[-1].fading_f1 if results else 0
        avg_f1 = np.mean([r.f1 for r in results]) if results else 0
        std_f1 = np.std([r.f1 for r in results]) if results else 0
        print(f"Windows: {len(results)} | Avg F1: {avg_f1:.4f} | Std F1: {std_f1:.4f} | "
              f"Fading F1: {final_fading:.4f}")

    return results


def aggregate_metrics(results: List[PrequentialMetrics], alpha: float = FADING_FACTOR) -> Dict:
    """Compute aggregated metrics from prequential results."""
    if not results:
        return {}
    f1s = [r.f1 for r in results]
    accs = [r.accuracy for r in results]
    precs = [r.precision for r in results]
    recs = [r.recall for r in results]
    aucs = [r.auc for r in results]

    return {
        'accuracy': compute_fading_average(accs, alpha),
        'precision': compute_fading_average(precs, alpha),
        'recall': compute_fading_average(recs, alpha),
        'f1': compute_fading_average(f1s, alpha),
        'auc': compute_fading_average(aucs, alpha),
        'f1_std': float(np.std(f1s)),
        'f1_mean': float(np.mean(f1s)),
        'n_windows': len(results),
        'n_drifts': sum(1 for r in results if r.drift_detected),
    }

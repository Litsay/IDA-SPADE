"""Phase 2: Concept Drift Perception — redesigned.

Ground truth drift identification from distribution changes (anomaly ratio),
not from model F1 drops. Includes:
- Drift detection quality metrics (Precision / Recall / F1)
- Per-window F1 time series for IDA-SPADE vs Reactive
- Opportunity window ΔT and F1 recovery speed comparison
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .evaluator import PrequentialMetrics
from .config import F1_RECOVERY_THRESHOLD


@dataclass
class DriftEvent:
    """A single drift event with time anchors and recovery info."""
    description: str
    T_true: int
    T_alert: int = -1        # first proactive detection near T_true
    T_react: int = -1        # first reactive detection near T_true
    f1_before_proactive: float = 0.0
    f1_before_reactive: float = 0.0
    f1_dip_proactive: float = 0.0
    f1_dip_reactive: float = 0.0
    recovery_windows_proactive: int = -1  # windows to recover (proactive)
    recovery_windows_reactive: int = -1   # windows to recover (reactive)
    max_f1_drop_proactive: float = 0.0
    max_f1_drop_reactive: float = 0.0
    opportunity_window: int = 0           # T_react - T_alert
    anomaly_ratio_before: float = 0.0
    anomaly_ratio_after: float = 0.0


def identify_drift_points(windows, threshold: float = 0.15,
                          min_gap: int = 8) -> List[int]:
    """Identify ground truth drift points from anomaly ratio changes.

    A drift is defined as a window where the anomaly ratio differs from
    the recent average by more than `threshold`. Consecutive drifts within
    `min_gap` are merged (only the first is kept).
    """
    if len(windows) < 5:
        return []

    ratios = np.array([w.metadata.get('anomaly_ratio', 0) for w in windows])

    # Smooth with rolling mean (window=3) to reduce noise
    smoothed = np.convolve(ratios, np.ones(3) / 3, mode='same')

    drift_candidates = []
    for i in range(3, len(ratios)):
        recent_avg = np.mean(smoothed[max(0, i - 5):i])
        delta = abs(smoothed[i] - recent_avg)
        if delta > threshold:
            drift_candidates.append(i)

    # Merge consecutive: keep first in each cluster
    if not drift_candidates:
        return []
    merged = [drift_candidates[0]]
    for idx in drift_candidates[1:]:
        if idx - merged[-1] >= min_gap:
            merged.append(idx)

    return merged


def extract_time_anchors(results_proactive: List[PrequentialMetrics],
                         results_reactive: List[PrequentialMetrics],
                         T_true: int, windows,
                         lookback: int = 5,
                         lookahead: int = 15) -> DriftEvent:
    """Extract T_alert, T_react, and F1 recovery for one drift event."""

    ratios = [w.metadata.get('anomaly_ratio', 0) for w in windows]
    event = DriftEvent(
        description='',
        T_true=T_true,
        anomaly_ratio_before=np.mean(ratios[max(0, T_true - lookback):T_true]) if T_true > 0 else 0,
        anomaly_ratio_after=np.mean(ratios[T_true:min(T_true + lookback, len(ratios))]),
    )

    n_pro = len(results_proactive)
    n_rea = len(results_reactive)

    # --- F1 before drift (both methods) ---
    start = max(0, T_true - lookback)
    if start < n_pro:
        f1s_before_p = [results_proactive[i].f1 for i in range(start, min(T_true, n_pro))]
        event.f1_before_proactive = np.mean(f1s_before_p) if f1s_before_p else 0.0
    if start < n_rea:
        f1s_before_r = [results_reactive[i].f1 for i in range(start, min(T_true, n_rea))]
        event.f1_before_reactive = np.mean(f1s_before_r) if f1s_before_r else 0.0

    # --- T_alert: first proactive detection within search range ---
    search_start = max(0, T_true - lookback * 3)
    search_end = min(T_true + lookahead, n_pro)
    for i in range(search_start, search_end):
        if results_proactive[i].drift_detected:
            event.T_alert = i
            break

    # --- T_react: first reactive detection within search range ---
    search_end_r = min(T_true + lookahead, n_rea)
    for i in range(search_start, search_end_r):
        if results_reactive[i].drift_detected:
            event.T_react = i
            break

    # --- Opportunity window ---
    if event.T_alert >= 0 and event.T_react >= 0:
        event.opportunity_window = event.T_react - event.T_alert

    # --- F1 dip and recovery (proactive) ---
    end = min(T_true + lookahead, n_pro)
    if T_true < n_pro:
        f1s_after = [results_proactive[i].f1 for i in range(T_true, end)]
        if f1s_after:
            event.f1_dip_proactive = min(f1s_after)
            event.max_f1_drop_proactive = max(0, event.f1_before_proactive - event.f1_dip_proactive)
            recovery_level = F1_RECOVERY_THRESHOLD * event.f1_before_proactive
            for j, f1 in enumerate(f1s_after):
                if f1 >= recovery_level and j > 0:
                    event.recovery_windows_proactive = j
                    break

    # --- F1 dip and recovery (reactive) ---
    end_r = min(T_true + lookahead, n_rea)
    if T_true < n_rea:
        f1s_after_r = [results_reactive[i].f1 for i in range(T_true, end_r)]
        if f1s_after_r:
            event.f1_dip_reactive = min(f1s_after_r)
            event.max_f1_drop_reactive = max(0, event.f1_before_reactive - event.f1_dip_reactive)
            recovery_level = F1_RECOVERY_THRESHOLD * event.f1_before_reactive
            for j, f1 in enumerate(f1s_after_r):
                if f1 >= recovery_level and j > 0:
                    event.recovery_windows_reactive = j
                    break

    return event


def compute_detection_quality(results: List[PrequentialMetrics],
                              true_drift_windows: List[int],
                              tolerance: int = 3) -> Dict:
    """Compute Precision / Recall / F1 for drift detection.

    A detection at window k is a true positive if any T_true is within
    [k - tolerance, k + tolerance]. Each T_true can only match one detection.
    """
    detected_windows = [i for i, r in enumerate(results) if r.drift_detected]

    if not detected_windows and not true_drift_windows:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'n_detected': 0, 'n_true': 0, 'tp': 0, 'fp': 0, 'fn': 0}

    # Match detections to true drifts
    matched_true = set()
    tp = 0
    fp = 0
    for d in detected_windows:
        matched = False
        for t in true_drift_windows:
            if abs(d - t) <= tolerance and t not in matched_true:
                matched_true.add(t)
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(true_drift_windows) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'n_detected': len(detected_windows), 'n_true': len(true_drift_windows),
        'tp': tp, 'fp': fp, 'fn': fn,
    }


def compute_phase2_summary(events: List[DriftEvent], model_name: str) -> Dict:
    """Compute Phase 2 summary statistics across all drift events."""
    if not events:
        return {'model': model_name, 'n_events': 0}

    valid_events = [e for e in events if e.T_alert >= 0 and e.T_react >= 0]
    return {
        'model': model_name,
        'n_events': len(events),
        'n_with_both_detections': len(valid_events),
        'avg_opportunity_window': float(np.mean([e.opportunity_window for e in valid_events])) if valid_events else 0,
        'avg_f1_drop_proactive': float(np.mean([e.max_f1_drop_proactive for e in events])),
        'avg_f1_drop_reactive': float(np.mean([e.max_f1_drop_reactive for e in events])),
        'avg_recovery_proactive': float(np.mean([
            e.recovery_windows_proactive for e in events if e.recovery_windows_proactive >= 0
        ])) if any(e.recovery_windows_proactive >= 0 for e in events) else -1,
        'avg_recovery_reactive': float(np.mean([
            e.recovery_windows_reactive for e in events if e.recovery_windows_reactive >= 0
        ])) if any(e.recovery_windows_reactive >= 0 for e in events) else -1,
    }


# Keep old function name for backward compatibility
def find_natural_drift_points(results, f1_drop_threshold=0.1, lookback=5):
    drift_points = []
    f1s = [r.f1 for r in results]
    for i in range(lookback, len(f1s)):
        avg_before = np.mean(f1s[max(0, i - lookback):i])
        if avg_before - f1s[i] > f1_drop_threshold:
            drift_points.append(i)
    return drift_points

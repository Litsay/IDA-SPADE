"""Phase 3: Knowledge Retention (Catastrophic Forgetting Resistance).

Measures per-attack-category recall during stationary phases to validate
EWC/L_stable effectiveness.
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics import recall_score

from .streaming_interface import Window
from .evaluator import PrequentialMetrics
from .config import STATIONARY_MIN_WINDOWS


def identify_stationary_phases(results: List[PrequentialMetrics],
                               drift_windows: List[int],
                               min_gap: int = STATIONARY_MIN_WINDOWS) -> List[Tuple[int, int]]:
    """Identify stationary (non-drift) phases from drift event windows.

    Returns list of (start_window, end_window) tuples for stationary phases.
    """
    if not results:
        return []

    all_windows = [r.window_index for r in results]
    if not all_windows:
        return []

    # Sort drift windows
    sorted_drifts = sorted(set(drift_windows))
    if not sorted_drifts:
        # No drift = entire stream is stationary
        return [(all_windows[0], all_windows[-1])]

    phases = []
    # Before first drift
    if sorted_drifts[0] - all_windows[0] >= min_gap:
        phases.append((all_windows[0], sorted_drifts[0] - 1))

    # Between drifts
    for i in range(len(sorted_drifts) - 1):
        gap_start = sorted_drifts[i] + min_gap
        gap_end = sorted_drifts[i + 1] - 1
        if gap_end - gap_start >= min_gap:
            phases.append((gap_start, gap_end))

    # After last drift
    if all_windows[-1] - sorted_drifts[-1] >= min_gap:
        phases.append((sorted_drifts[-1] + min_gap, all_windows[-1]))

    return phases


def compute_per_class_recall(windows: List[Window],
                             predictions_per_window: Dict[int, np.ndarray],
                             stationary_phases: List[Tuple[int, int]],
                             attack_classes: List[str]) -> Dict:
    """Compute per-attack-class recall during stationary phases.

    Args:
        windows: list of Window objects with y_multiclass labels
        predictions_per_window: {window_index: predicted_labels_array}
        stationary_phases: list of (start, end) window index tuples
        attack_classes: list of attack category names

    Returns:
        Dict with per-class per-window recall data
    """
    # Build window lookup
    window_map = {w.index: w for w in windows}

    # Per-class, per-window recall
    class_recall = {cls: [] for cls in attack_classes}
    window_indices = []

    for phase_start, phase_end in stationary_phases:
        for wid in range(phase_start, phase_end + 1):
            if wid not in window_map or wid not in predictions_per_window:
                continue
            w = window_map[wid]
            preds = predictions_per_window[wid]
            y_mc = w.y_multiclass
            y_bin = w.y_binary

            if len(preds) != len(y_mc):
                min_len = min(len(preds), len(y_mc))
                preds = preds[:min_len]
                y_mc = y_mc[:min_len]
                y_bin = y_bin[:min_len]

            window_indices.append(wid)

            for cls in attack_classes:
                mask = y_mc == cls
                if mask.sum() == 0:
                    class_recall[cls].append(np.nan)
                else:
                    # For attack classes: recall = correctly predicted as 1
                    # For normal class: recall = correctly predicted as 0
                    if cls.lower() == 'normal':
                        cls_true = y_bin[mask]
                        cls_pred = preds[mask]
                        rec = float((cls_pred == 0).sum()) / max(mask.sum(), 1)
                    else:
                        cls_true = y_bin[mask]
                        cls_pred = preds[mask]
                        rec = float((cls_pred == 1).sum()) / max(mask.sum(), 1)
                    class_recall[cls].append(rec)

    return {
        'window_indices': window_indices,
        'per_class_recall': class_recall,
        'stationary_phases': stationary_phases,
    }


def compute_retention_summary(class_recall_data: Dict, attack_classes: List[str]) -> Dict:
    """Compute summary statistics for knowledge retention."""
    per_class = class_recall_data['per_class_recall']
    summary = {}
    for cls in attack_classes:
        vals = [v for v in per_class.get(cls, []) if not np.isnan(v)]
        if vals:
            summary[cls] = {
                'mean_recall': float(np.mean(vals)),
                'std_recall': float(np.std(vals)),
                'min_recall': float(np.min(vals)),
                'n_windows': len(vals),
            }
        else:
            summary[cls] = {'mean_recall': np.nan, 'std_recall': np.nan, 'min_recall': np.nan, 'n_windows': 0}
    return summary

"""Phase 4: Ablation variants and compute-benefit analysis.

Paper Table 3 ablation variants (all use real ECBA + entity-level processing):
- IDA-SPADE-Reactive: PC-DriftForecasting -> KS-test on entity features
- IDA-SPADE-Global: ECBA entity identification -> random grouping
- IDA-SPADE-Statistical: PC-DriftForecasting -> sliding window z-score
"""
import numpy as np
import pandas as pd
import time
from scipy.stats import ks_2samp
from collections import deque
from sklearn.preprocessing import MinMaxScaler

from .streaming_interface import StreamingModel
from .ida_spade_wrapper import IDASpadeStreaming
from .config import HIDDEN_DIMS, BASELINE_DRIFT_THRESHOLD


class IDASpadeReactive(IDASpadeStreaming):
    """Ablation: replace PC-DriftForecasting with KS-test on entity features (reactive)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-Reactive')
        kwargs['use_causal'] = False
        super().__init__(**kwargs)
        self._ks_threshold = BASELINE_DRIFT_THRESHOLD
        self._prev_entity_features = None

    def detect_drift(self, X):
        t0 = time.perf_counter()
        detected = False
        confidence = 0.0

        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)

        if feat_tensor is not None and self._prev_entity_features is not None:
            current = feat_tensor.numpy()
            prev = self._prev_entity_features
            n_feats = min(current.shape[1], prev.shape[1], 20)
            p_vals = []
            for i in range(n_feats):
                try:
                    _, p = ks_2samp(prev[:, i], current[:, i])
                    p_vals.append(p)
                except Exception:
                    p_vals.append(1.0)
            min_p = min(p_vals) if p_vals else 1.0
            detected = min_p < self._ks_threshold
            confidence = 1.0 - min_p if detected else 0.0

        if feat_tensor is not None:
            self._prev_entity_features = feat_tensor.numpy()

        self._drift_state = 'drift' if detected else 'stable'
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def reset(self):
        super().reset()
        self._prev_entity_features = None


class IDASpadeGlobal(IDASpadeStreaming):
    """Ablation: replace ECBA entity identification with random grouping."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-Global')
        super().__init__(**kwargs)

    def _aggregate_and_prepare(self, X):
        """Random grouping instead of real entity identification."""
        df = self._numpy_to_dataframe(X)

        n_groups = min(15, max(3, len(df) // 100))
        random_ids = [f"random_entity_{np.random.randint(0, n_groups)}"
                      for _ in range(len(df))]

        df_copy = df.copy()
        df_copy['_entity_id'] = random_ids

        entity_feats = {}
        for eid, group in df_copy.groupby('_entity_id'):
            if len(group) == 0:
                continue
            agg = {}
            for feat, funcs in self._preset['aggregation_rules'].items():
                if feat in group.columns:
                    feat_aggs = self._aggregator._aggregate_feature(group[feat], funcs)
                    agg.update(feat_aggs)
            agg['entity_size'] = len(group)
            entity_feats[str(eid)] = agg

        if not entity_feats:
            return None, None, None

        feats_list = []
        entity_order = []
        for eid, rec in entity_feats.items():
            vec = [float(rec.get(key, 0.0)) if isinstance(rec.get(key, 0.0), (int, float)) else 0.0
                   for key in self._feature_template]
            feats_list.append(vec)
            entity_order.append(eid)

        if not feats_list:
            return None, None, None

        raw = np.array(feats_list, dtype=np.float32)

        if not self.scaler_fitted:
            self.scaler.fit(raw)
            self.scaler_fitted = True
        scaled = self.scaler.transform(raw)

        feat_tensor = torch.FloatTensor(scaled).clamp(-5, 5)
        feat_tensor[torch.isnan(feat_tensor)] = 0
        feat_tensor[torch.isinf(feat_tensor)] = 0

        return feat_tensor, entity_order, random_ids


# Need torch for IDASpadeGlobal
import torch


class IDASpadeStatistical(IDASpadeStreaming):
    """Ablation: replace PC-DriftForecasting with sliding window z-score monitoring."""

    def __init__(self, var_threshold=2.0, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-Statistical')
        kwargs['use_causal'] = False
        super().__init__(**kwargs)
        self._mean_history = deque(maxlen=20)
        self._var_history = deque(maxlen=20)
        self._var_threshold = var_threshold

    def detect_drift(self, X):
        t0 = time.perf_counter()

        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)

        detected = False
        confidence = 0.0

        if feat_tensor is not None:
            entity_feats = feat_tensor.numpy()
            current_mean = entity_feats.mean(axis=0)
            current_var = entity_feats.var(axis=0)

            self._mean_history.append(current_mean.mean())
            self._var_history.append(current_var.mean())

            if len(self._mean_history) >= 5:
                recent = list(self._mean_history)[-5:]
                older = list(self._mean_history)[:-5] if len(self._mean_history) > 5 else recent
                if older:
                    mean_change = abs(np.mean(recent) - np.mean(older))
                    std_older = np.std(older) + 1e-8
                    z_score = mean_change / std_older
                    if z_score > self._var_threshold:
                        detected = True
                        confidence = min(z_score / (self._var_threshold * 2), 1.0)

        self._drift_state = 'drift' if detected else 'stable'
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def reset(self):
        super().reset()
        self._mean_history.clear()
        self._var_history.clear()


class IDASpadeNoCL(IDASpadeStreaming):
    """Ablation: remove contrastive loss (keep prototype + layered adaptation)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-NoCL')
        super().__init__(**kwargs)
        self._use_contrastive = False


class IDASpadeNoProto(IDASpadeStreaming):
    """Ablation: remove prototype module (keep contrastive + layered adaptation)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-NoProto')
        super().__init__(**kwargs)
        self._use_prototype = False


class IDASpadeNoTemporal(IDASpadeStreaming):
    """Ablation: remove temporal context buffer (use current-window features only)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-NoTemporal')
        super().__init__(**kwargs)
        self._use_temporal = False


class IDASpadeNoFSM(IDASpadeStreaming):
    """Ablation: bypass the three-state FSM and always operate in 'drift' mode.

    Forces `_drift_state = 'drift'` after every detect_drift() call so the
    update() method always takes the drift branch (unfreeze + LR x2 + epochs=2,
    no EWC). The PC-DriftForecasting matrix is still computed and reused for
    SupCon weighting; only the FSM state assignment is overridden.

    Built on the no-temporal architecture (matches current 'Full' baseline).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-NoFSM')
        super().__init__(**kwargs)
        self._use_temporal = False  # match current Full baseline

    def detect_drift(self, X):
        result = super().detect_drift(X)
        # Override the FSM state regardless of pc/trend/novelty signals.
        self._drift_state = 'drift'
        return result


class IDASpadeNoLayered(IDASpadeStreaming):
    """Ablation: keep FSM and other layered behaviours, but use a constant EWC
    strength (= EWC_LAMBDA) across stable / pre_drift / drift states.

    Disables the EWC-scaling differentiation only: pre_drift no longer halves
    EWC (EWC_PRE_DRIFT_SCALE bypass) and drift no longer drops EWC entirely
    (use_ewc=False bypass). Other layered behaviours (LR scaling, prototype EMA
    decay, conditional backbone freeze) remain active.

    Built on the no-temporal architecture (matches current 'Full' baseline).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-NoLayered')
        super().__init__(**kwargs)
        self._use_temporal = False  # match current Full baseline
        self._use_layered_ewc = False


def profile_components(model: StreamingModel, X: np.ndarray, y: np.ndarray,
                       n_rounds: int = 5) -> dict:
    """Profile per-component timing for a model over multiple rounds."""
    timings = []
    for _ in range(n_rounds):
        model.predict(X)
        model.detect_drift(X)
        model.update(X, y)
        timings.append(model.get_timing())

    all_keys = set()
    for t in timings:
        all_keys.update(t.keys())

    result = {}
    for key in all_keys:
        vals = [t.get(key, 0) for t in timings]
        result[key] = {
            'mean_ms': float(np.mean(vals)),
            'std_ms': float(np.std(vals)),
        }
    return result

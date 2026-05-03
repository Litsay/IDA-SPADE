"""IDA-SPADE B1 variant: alert/quiet binary state + constant EWC.

B1 differs from the canonical (three-state FSM) IDA-SPADE in three ways:

1. The three-state FSM (stable / pre_drift / drift) is replaced with a binary
   alert / quiet signal driven by Drift_Alert(k); pre_drift no longer exists.
   Once an alert fires, the model stays in "hot" mode for K_hot windows; after
   the cooldown it returns to "quiet" mode.
2. EWC strength is constant across all windows (no pre_drift halving, no drift
   anchor drop). Equivalent to setting `_use_layered_ewc = False` in the parent.
3. Prototype EMA decay beta_k and backbone freeze are gated by the alert
   cooldown counter, not by the FSM state.

PC-DriftForecasting still produces the alert and the coupling matrix used by
the manifold-guided supervised contrastive loss; only the FSM-driven loss
schedule is removed.

Variants in this file
    IDASpadeB1            : the new "Full" baseline for the B1 paper version
    IDASpadeB1NoPCAlert   : B1 with PC channel removed from alert signal
                            (only entity novelty + risk trend trigger alerts;
                             PC matrix still supplies SupCon weights)
    IDASpadeB1Reactive    : B1 with KS-test instead of PC for alerts
    IDASpadeB1Statistical : B1 with sliding z-score instead of PC
    IDASpadeB1Global      : B1 with random entity grouping (NoECBA)
    IDASpadeB1NoCL        : B1 without supervised contrastive head
    IDASpadeB1NoProto     : B1 without prototype module
"""
import time
from collections import deque

import numpy as np
import torch
from scipy.stats import ks_2samp

from .ida_spade_wrapper import IDASpadeStreaming
from .config import BASELINE_DRIFT_THRESHOLD


B1_K_HOT_DEFAULT = 3       # windows of hot mode after each alert
B1_N_QUIET_FREEZE = 15     # consecutive quiet windows before backbone freeze


class IDASpadeB1(IDASpadeStreaming):
    """B1: alert / quiet binary + constant EWC.

    Compared with `IDASpadeStreaming`:
      - `detect_drift()` reuses the parent's three-channel alert (PC OR
        novelty OR trend) but collapses the resulting state to a binary
        cooldown counter; the next K_hot windows are "hot"
        (`_drift_state == 'drift'`) and the rest are "quiet"
        (`_drift_state == 'stable'`).
      - `_use_layered_ewc = False` so the parent `update()` keeps EWC enabled
        in both hot and quiet branches. Drift's `use_ewc=False` and pre_drift's
        `EWC_PRE_DRIFT_SCALE` are both bypassed.
      - prototype beta_k still alternates 0.99 (quiet) / 0.90 (hot), gated by
        the cooldown via `_drift_state` (the parent's `_train_batch` reads it
        directly).
      - backbone freeze still triggers after `_freeze_after_stable` consecutive
        quiet windows, with the counter reset whenever the cooldown is active.
    """

    def __init__(self, K_hot: int = B1_K_HOT_DEFAULT,
                 N_quiet_freeze: int = B1_N_QUIET_FREEZE,
                 **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1')
        super().__init__(**kwargs)
        self.K_hot = int(K_hot)
        self._alert_cooldown = 0
        self._use_layered_ewc = False        # constant EWC across windows
        self._freeze_after_stable = int(N_quiet_freeze)
        self._use_temporal = False           # match current Full architecture

    def _b1_state_remap(self, detected: bool):
        """Update cooldown and remap _drift_state to 'drift' / 'stable'."""
        if detected:
            self._alert_cooldown = self.K_hot
        elif self._alert_cooldown > 0:
            self._alert_cooldown -= 1
        self._drift_state = 'drift' if self._alert_cooldown > 0 else 'stable'

    def detect_drift(self, X):
        # Run parent's three-channel detection (PC + novelty + trend) and
        # collapse the resulting state assignment to binary B1 semantics.
        detected, confidence = super().detect_drift(X)
        self._b1_state_remap(detected)
        return detected, confidence

    def reset(self):
        super().reset()
        self._alert_cooldown = 0


class IDASpadeB1NoPCAlert(IDASpadeB1):
    """B1 with PC channel suppressed from the alert signal.

    Only entity novelty and risk trend can fire alerts. The PC matrix is still
    computed each window and reused as the manifold-guided weighting in
    L_SupCon; only the PC -> alert link is severed. This isolates PC's
    contribution to alert quality from its contribution to representation
    weighting.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-NoPCAlert')
        super().__init__(**kwargs)

    def detect_drift(self, X):
        t0 = time.perf_counter()
        # Update PC's internal state (entity history + risk score) so the
        # SupCon weighting and risk-trend channel still see the latest data,
        # but ignore PC's drift_detected flag.
        if self.use_causal:
            self._ensure_entity_history_updated(X)
            try:
                result = self._pc_forecaster.forecast_drift()
                phi_k = result.get('risk_score', 0.0)
                self._risk_history.append(phi_k)
            except Exception:
                pass

        novelty_ratio = self._compute_entity_novelty(X)
        trend_alert = self._extrapolate_risk_trend()
        detected = bool(trend_alert) or (novelty_ratio > 0.4)
        confidence = 0.3 if detected else 0.0
        self._b1_state_remap(detected)
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence


class IDASpadeB1Reactive(IDASpadeB1):
    """B1 architecture + KS-test alert (replaces PC-DriftForecasting)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-Reactive')
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
            confidence = (1.0 - min_p) if detected else 0.0
        if feat_tensor is not None:
            self._prev_entity_features = feat_tensor.numpy()
        self._b1_state_remap(detected)
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def reset(self):
        super().reset()
        self._prev_entity_features = None


class IDASpadeB1Statistical(IDASpadeB1):
    """B1 architecture + sliding-window z-score alert (replaces PC)."""

    def __init__(self, var_threshold: float = 2.0, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-Statistical')
        kwargs['use_causal'] = False
        super().__init__(**kwargs)
        self._mean_history = deque(maxlen=20)
        self._var_history = deque(maxlen=20)
        self._var_threshold = float(var_threshold)

    def detect_drift(self, X):
        t0 = time.perf_counter()
        detected = False
        confidence = 0.0
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)
        if feat_tensor is not None:
            ent_feats = feat_tensor.numpy()
            current_mean = ent_feats.mean(axis=0)
            current_var = ent_feats.var(axis=0)
            self._mean_history.append(float(current_mean.mean()))
            self._var_history.append(float(current_var.mean()))
            if len(self._mean_history) >= 5:
                recent = list(self._mean_history)[-5:]
                older = list(self._mean_history)[:-5] if len(self._mean_history) > 5 else recent
                if older:
                    mean_change = abs(np.mean(recent) - np.mean(older))
                    std_older = np.std(older) + 1e-8
                    z_score = mean_change / std_older
                    if z_score > self._var_threshold:
                        detected = True
                        confidence = float(min(z_score / (self._var_threshold * 2), 1.0))
        self._b1_state_remap(detected)
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def reset(self):
        super().reset()
        self._mean_history.clear()
        self._var_history.clear()


class IDASpadeB1Global(IDASpadeB1):
    """B1 architecture with random entity grouping (replaces ECBA)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-Global')
        super().__init__(**kwargs)

    def _aggregate_and_prepare(self, X):
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


class IDASpadeB1NoCL(IDASpadeB1):
    """B1 architecture without the supervised contrastive head."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-NoCL')
        super().__init__(**kwargs)
        self._use_contrastive = False


class IDASpadeB1NoProto(IDASpadeB1):
    """B1 architecture without the prototype module."""

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-NoProto')
        super().__init__(**kwargs)
        self._use_prototype = False


class IDASpadeB1ReactiveNoProto(IDASpadeB1Reactive):
    """B1 architecture with KS-test alert AND prototype removed.

    This is the {Reactive alert} x {NoProto} cell of the 2x2 ablation, used to
    test whether PC-DriftForecasting and the prototype module make INDEPENDENT
    contributions or whether one carries the other through the alert-gated
    EMA-decay coupling.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'IDA-SPADE-B1-Reactive-NoProto')
        super().__init__(**kwargs)
        self._use_prototype = False


# Convenience factory: name -> class
B1_VARIANT_CLASSES = {
    'B1':                  IDASpadeB1,
    'B1-NoPCAlert':        IDASpadeB1NoPCAlert,
    'B1-Reactive':         IDASpadeB1Reactive,
    'B1-Statistical':      IDASpadeB1Statistical,
    'B1-Global':           IDASpadeB1Global,
    'B1-NoCL':             IDASpadeB1NoCL,
    'B1-NoProto':          IDASpadeB1NoProto,
    'B1-Reactive-NoProto': IDASpadeB1ReactiveNoProto,
}


def make_b1_variant(name: str, *, feature_cols=None, dataset_name='NSL-KDD',
                    K_hot: int = B1_K_HOT_DEFAULT, **extra):
    """Construct a B1 variant by name.

    Examples
        make_b1_variant('B1', feature_cols=fc, dataset_name='UNSW-NB15')
        make_b1_variant('B1-NoPCAlert', K_hot=5, ...)
    """
    if name not in B1_VARIANT_CLASSES:
        raise ValueError(f'Unknown B1 variant: {name!r}; '
                         f'must be one of {list(B1_VARIANT_CLASSES)}')
    cls = B1_VARIANT_CLASSES[name]
    return cls(feature_cols=feature_cols, dataset_name=dataset_name,
               K_hot=K_hot, **extra)

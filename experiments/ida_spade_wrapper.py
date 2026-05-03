"""IDA-SPADE v6 wrapped as a StreamingModel for the experiment framework.

Manifold-Guided Contrastive Continual Learning:
- NetworkEntityAggregator for ECBA entity aggregation
- PCDriftForecaster for Takens embedding + causal graph drift detection
- ContrastiveContinualMLP: temporal attention + contrastive + prototypes
- Layered adaptation: stable/pre_drift/drift with backbone freeze/unfreeze
"""
import sys
import os
import time
import importlib.util as _ilu
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy as sp_entropy

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Import real components from IDA-SPADE.py
_spec = _ilu.spec_from_file_location(
    "ida_standalone", os.path.join(_parent, "IDA-SPADE.py"))
_ida = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ida)
NetworkEntityAggregator = _ida.NetworkEntityAggregator
PCDriftForecaster = _ida.PCDriftForecaster
STANDALONE_PRESETS = _ida.DATASET_PRESETS

from .streaming_interface import StreamingModel
from .contrastive_modules import (
    ContrastiveContinualMLP, EntityFeatureBuffer, sup_con_loss
)
from .config import (HIDDEN_DIMS, LEARNING_RATE, BATCH_SIZE, DROPOUT, EWC_LAMBDA,
                     PC_E, PC_TAU, PC_ALPHA, PC_BETA, PC_GAMMA,
                     PC_SIGMA, PC_BASELINE_WINDOW, PC_THETA_LID, PC_K_LID,
                     PC_LAMBDA_DECAY, PC_ETA, MIN_WINDOWS_CAUSALITY,
                     BASELINE_DRIFT_THRESHOLD, WINDOW_SIZE, DATASET_CONFIGS,
                     CONTRASTIVE_ALPHA, CONTRASTIVE_TAU, CONTRASTIVE_PROJ_DIM,
                     PROTOTYPE_BETA_STABLE, PROTOTYPE_BETA_DRIFT,
                     PROTOTYPE_WEIGHT_BASE, PROTOTYPE_WEIGHT_REVERSAL,
                     TEMPORAL_BUFFER_LEN,
                     LR_PRE_DRIFT_SCALE, LR_DRIFT_SCALE, EWC_PRE_DRIFT_SCALE)


class IDASpadeStreaming(StreamingModel):
    """IDA-SPADE v6 as a StreamingModel using ECBA + PCDriftForecaster +
    ContrastiveContinualMLP with temporal context, prototypes, and layered adaptation.

    Receives connection-level data (same as all baselines), internally performs
    entity aggregation (ECBA), and outputs entity-level predictions.
    """

    def __init__(self, input_dim: int = None, dataset_name: str = 'NSL-KDD',
                 feature_cols: list = None,
                 use_causal: bool = True, use_ewc: bool = True,
                 name: str = 'IDA-SPADE'):
        super().__init__(name)
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.feature_cols = feature_cols
        self.use_causal = use_causal
        self.use_ewc = use_ewc
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.training_step = 0
        self._last_timing = {}
        self._window_count = 0

        # Resolve dataset preset
        self._preset = self._resolve_preset()

        # ECBA: real entity aggregator
        self._aggregator = NetworkEntityAggregator(
            window_size=WINDOW_SIZE,
            aggregation_rules=self._preset['aggregation_rules'],
            label_col=self._preset['label_col'],
            label_positive=self._preset['label_positive'],
            proto_prefix=self._preset.get('proto_prefix', 'proto_'),
            service_prefix=self._preset.get('service_prefix', 'service_'),
            trend_features=self._preset.get('trend_features', []),
        )

        # PC-DriftForecasting: real forecaster
        self._pc_forecaster = PCDriftForecaster(
            E=PC_E, tau=PC_TAU,
            alpha=PC_ALPHA, beta=PC_BETA, gamma=PC_GAMMA,
            sigma=self._preset.get('pc_sigma', PC_SIGMA),
            theta_lid=PC_THETA_LID, K_lid=PC_K_LID,
            lambda_decay=PC_LAMBDA_DECAY, eta=PC_ETA,
            baseline_window=self._preset.get('pc_baseline_window', PC_BASELINE_WINDOW),
            causal_features=self._preset.get('causal_features', []),
            min_history=MIN_WINDOWS_CAUSALITY,
        )

        # Feature template (fixed dimension)
        self._feature_template = self._build_feature_template()

        # Drift state: 'stable' | 'pre_drift' | 'drift'
        self._drift_state = 'stable'
        self._last_drift_result = (False, 0.0)
        self._prediction_threshold = 0.5

        # Proactive perception: entity novelty tracking
        self._known_entities = set()
        self._entity_novelty_history = deque(maxlen=20)

        # Proactive perception: risk trend tracking
        self._risk_history = deque(maxlen=30)

        # Label distribution reversal detection
        self._attack_ratio_history = deque(maxlen=10)

        # Class-stratified replay buffer for EWC Fisher
        self._replay_pos_X = deque(maxlen=250)
        self._replay_pos_y = deque(maxlen=250)
        self._replay_neg_X = deque(maxlen=250)
        self._replay_neg_y = deque(maxlen=250)

        # Cache for ECBA results (avoid 3x aggregation per window)
        self._cached_window_id = -1
        self._cached_ecba = None

        # --- v6: New components ---
        # Temporal context buffer
        self._temporal_buffer = EntityFeatureBuffer(buffer_len=TEMPORAL_BUFFER_LEN)

        # Contrastive learning state
        self._contrastive_alpha = CONTRASTIVE_ALPHA
        self._contrastive_tau = CONTRASTIVE_TAU

        # Prototype fusion weight (adapts during reversal)
        self._prototype_weight = PROTOTYPE_WEIGHT_BASE

        # Ablation flags (overridden by NoCL/NoProto/NoLayered variants)
        self._use_contrastive = True
        self._use_prototype = True
        # When False, EWC strength is constant (= EWC_LAMBDA) across all FSM
        # states; pre_drift's EWC_PRE_DRIFT_SCALE and drift's use_ewc=False are
        # both bypassed, isolating the contribution of layered EWC scheduling.
        self._use_layered_ewc = True

        # Current entity order for contrastive loss (set during update)
        self._current_entity_order = None

        # Adaptive backbone freeze: only freeze after N consecutive stable windows
        self._consecutive_stable = 0
        self._freeze_after_stable = 15  # require 15 consecutive stable windows (tuned)

        # Reversal cooldown: prototype-only mode for N windows after reversal
        self._reversal_cooldown = 0

        # Temporal context flag (overridden by NoTemporal variant)
        self._use_temporal = True

        # PC consecutive confirmation state
        self._prev_pc_detected = False

    # =========================================================================
    # Setup helpers
    # =========================================================================

    def _resolve_preset(self):
        """Get dataset preset from standalone DATASET_PRESETS."""
        mapping = {
            'NSL-KDD': 'NSL-KDD',
            'UNSW-NB15': 'UNSW-NB15',
            'CIC-IDS-2017': 'CIC-IDS-2017',
        }
        key = mapping.get(self.dataset_name, self.dataset_name)
        if key in STANDALONE_PRESETS:
            return STANDALONE_PRESETS[key]
        return STANDALONE_PRESETS.get('NSL-KDD', {})

    def _build_feature_template(self):
        """Build fixed-order feature name list from aggregation rules."""
        template = []
        for feat in sorted(self._preset['aggregation_rules'].keys()):
            for func in self._preset['aggregation_rules'][feat]:
                template.append(f"{feat}_{func}")
        template.append('entity_size')
        for tf in sorted(self._preset.get('trend_features', [])):
            template.append(f'{tf}_trend')
        return template

    def _numpy_to_dataframe(self, X):
        """Convert numpy array back to DataFrame using feature_cols."""
        if self.feature_cols is not None and len(self.feature_cols) == X.shape[1]:
            return pd.DataFrame(X, columns=self.feature_cols)
        return pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    # =========================================================================
    # ECBA pipeline
    # =========================================================================

    def _aggregate_and_prepare(self, X):
        """Core ECBA pipeline: numpy -> DataFrame -> entity groups -> aggregate -> template features."""
        df = self._numpy_to_dataframe(X)
        entity_ids = self._aggregator._assign_entity_ids(df)
        entity_feats = self._aggregator.aggregate_features(df, window_id=self._window_count)

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

        # Scale (frozen MinMaxScaler — fitted once on init)
        if not self.scaler_fitted:
            self.scaler.fit(raw)
            self.scaler_fitted = True
        scaled = self.scaler.transform(raw)

        feat_tensor = torch.FloatTensor(scaled).clamp(-5, 5)
        feat_tensor[torch.isnan(feat_tensor)] = 0
        feat_tensor[torch.isinf(feat_tensor)] = 0

        return feat_tensor, entity_order, entity_ids

    def _extract_entity_labels(self, y, entity_ids, entity_order):
        """Compute entity-level labels from connection-level labels (OR aggregation)."""
        entity_label_map = defaultdict(int)
        for i, eid in enumerate(entity_ids):
            if i < len(y) and y[i] == 1:
                entity_label_map[eid] = 1
        labels = []
        for eid in entity_order:
            labels.append(entity_label_map.get(eid, 0))
        return torch.LongTensor(labels)

    def _get_cached_ecba(self, X):
        """Get ECBA results, using cache if available for current window."""
        if self._cached_window_id == self._window_count and self._cached_ecba is not None:
            return self._cached_ecba
        result = self._aggregate_and_prepare(X)
        self._cached_ecba = result
        self._cached_window_id = self._window_count
        return result

    # =========================================================================
    # Replay buffer
    # =========================================================================

    def _get_replay_data(self):
        """Get balanced replay data from stratified buffers."""
        all_X = list(self._replay_pos_X) + list(self._replay_neg_X)
        all_y = list(self._replay_pos_y) + list(self._replay_neg_y)
        if len(all_X) < 2:
            return None, None
        return torch.stack(all_X), torch.stack(all_y)

    def _store_replay_samples(self, features, labels):
        """Store samples into class-stratified replay buffer."""
        labels_cpu = labels.detach().cpu()
        features_cpu = features.detach().cpu()
        pos_mask = labels_cpu == 1
        neg_mask = labels_cpu == 0

        for i in pos_mask.nonzero(as_tuple=True)[0]:
            self._replay_pos_X.append(features_cpu[i])
            self._replay_pos_y.append(labels_cpu[i])

        neg_idx = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_idx) > 0:
            n_store = max(1, min(len(neg_idx), int(len(features) * 0.3)))
            chosen = neg_idx[torch.randperm(len(neg_idx))[:n_store]]
            for i in chosen:
                self._replay_neg_X.append(features_cpu[i])
                self._replay_neg_y.append(labels_cpu[i])

    # =========================================================================
    # Model init + helpers
    # =========================================================================

    def _init_model(self, input_dim):
        self.input_dim = input_dim
        self.model = ContrastiveContinualMLP(
            input_dim=input_dim,
            hidden_dims=HIDDEN_DIMS,  # [128, 64, 32] full backbone
            n_classes=2,
            dropout=DROPOUT,
            proj_dim=CONTRASTIVE_PROJ_DIM,
            ewc_lambda=EWC_LAMBDA,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    def _scale(self, X):
        """Scale raw connection-level features (fallback for non-ECBA paths)."""
        if not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True
        scaled = self.scaler.transform(X)
        return np.clip(scaled, -5, 5)

    def _class_weights(self, labels):
        unique, counts = torch.unique(labels, return_counts=True)
        w = torch.ones(2, device=self.device)
        if len(unique) > 1:
            total = len(labels)
            for i, lbl in enumerate(unique):
                w[lbl] = total / (2.0 * counts[i])
        return torch.clamp(w, 0.3, 8.0)

    def _set_lr(self, lr):
        """Set optimizer learning rate."""
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _calibrate_threshold(self, feat_tensor=None, entity_labels=None):
        """Calibrate prediction threshold on available labeled entity data."""
        if self.model is None:
            return

        if feat_tensor is not None and entity_labels is not None:
            X = feat_tensor
            y = entity_labels.numpy() if hasattr(entity_labels, 'numpy') else entity_labels
        else:
            replay_X, replay_y = self._get_replay_data()
            if replay_X is None or len(replay_X) < 32:
                return
            X = replay_X
            y = replay_y.numpy()

        if len(X) < 10:
            return

        unique = np.unique(y)
        if len(unique) < 2:
            return

        self.model.eval()
        with torch.no_grad():
            logits, feat = self.model(X.to(self.device))
            if self._use_prototype:
                scores = self.model.predict_with_prototypes(
                    feat, logits, self._prototype_weight).cpu().numpy()
            else:
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        from sklearn.metrics import f1_score as _f1
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.20, 0.70, 0.01):
            preds = (scores >= t).astype(int)
            f1 = _f1(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, round(t, 2)
        self._prediction_threshold = best_t

    # =========================================================================
    # Initialize
    # =========================================================================

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray):
        """Initial training: aggregate -> entity features -> train model -> warm up PC history."""
        feat_tensor, entity_order, entity_ids = self._aggregate_and_prepare(X_init)

        init_epochs = self._preset.get('init_epochs', 5)

        if feat_tensor is None:
            X_scaled = self._scale(X_init)
            self._init_model(X_scaled.shape[1])
            self._train_batch(
                torch.FloatTensor(X_scaled).to(self.device),
                torch.LongTensor(y_init).to(self.device),
                epochs=init_epochs, use_ewc=False)
            self._is_initialized = True
            return

        self._init_model(len(self._feature_template))

        entity_labels = self._extract_entity_labels(y_init, entity_ids, entity_order)

        # Store initial entity order for contrastive loss during init
        self._current_entity_order = entity_order
        self._train_batch(
            feat_tensor.to(self.device),
            entity_labels.to(self.device),
            epochs=init_epochs, use_ewc=False)
        self._current_entity_order = None

        self._calibrate_threshold(feat_tensor, entity_labels)

        # Pure benign init — lower threshold for attack detection
        benign_threshold = self._preset.get('benign_init_threshold', None)
        if benign_threshold is not None and len(np.unique(entity_labels.numpy())) == 1:
            self._prediction_threshold = benign_threshold

        # Seed temporal buffer with init data
        self._temporal_buffer.update(entity_order, feat_tensor)

        # Store initial replay samples
        self._store_replay_samples(feat_tensor.to(self.device), entity_labels.to(self.device))

        # Warm up PC entity history
        n_init = len(X_init)
        chunk_size = WINDOW_SIZE
        wid = 0
        for start in range(0, n_init, chunk_size):
            end = min(start + chunk_size, n_init)
            if end - start < chunk_size // 2:
                break
            chunk_df = self._numpy_to_dataframe(X_init[start:end])
            chunk_feats = self._aggregator.aggregate_features(chunk_df, window_id=wid)
            if chunk_feats:
                self._pc_forecaster.update_entity_history(chunk_feats)
            wid += 1

        self._is_initialized = True

    # =========================================================================
    # Predict (with temporal context + prototype fusion)
    # =========================================================================

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(0, dtype=np.int64)

        t0 = time.perf_counter()
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)

        if feat_tensor is None:
            self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
            return np.zeros(0, dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            current = feat_tensor.to(self.device)
            if self._use_temporal:
                history = self._temporal_buffer.get_history_tensor(
                    entity_order, feat_tensor, self.device)
                logits, feat, _ = self.model.forward_temporal(
                    current, history, return_all=True)
            else:
                logits, feat, _ = self.model(current, return_all=True)

            # Prototype-augmented decision
            if self._use_prototype:
                scores = self.model.predict_with_prototypes(
                    feat, logits, proto_weight=self._prototype_weight)
            else:
                scores = torch.softmax(logits, dim=1)[:, 1]

            entity_preds = (scores.cpu().numpy() >= self._prediction_threshold).astype(np.int64)

        self._last_timing['inference_ms'] = (time.perf_counter() - t0) * 1000
        return entity_preds

    def predict_evaluate(self, X: np.ndarray, y: np.ndarray):
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)
        if feat_tensor is None or self.model is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        entity_preds = self.predict(X)
        entity_labels = self._extract_entity_labels(y, entity_ids, entity_order)
        return entity_preds, entity_labels.numpy()

    def predict_proba(self, X: np.ndarray):
        """Return entity-level probability scores for positive class."""
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)
        if feat_tensor is None or self.model is None:
            return None
        self.model.eval()
        with torch.no_grad():
            current = feat_tensor.to(self.device)
            if self._use_temporal:
                history = self._temporal_buffer.get_history_tensor(
                    entity_order, feat_tensor, self.device)
                logits, feat, _ = self.model.forward_temporal(
                    current, history, return_all=True)
            else:
                logits, feat, _ = self.model(current, return_all=True)
            if self._use_prototype:
                scores = self.model.predict_with_prototypes(
                    feat, logits, proto_weight=self._prototype_weight)
            else:
                scores = torch.softmax(logits, dim=1)[:, 1]
        return scores.cpu().numpy()

    # =========================================================================
    # Train batch (CE + contrastive + EWC)
    # =========================================================================

    def _train_batch(self, X_t, y_t, epochs=1, use_ewc=True, ewc_scale=1.0,
                     skip_fisher=False):
        """Train with CE + contrastive + EWC, with layered adaptation."""
        if len(X_t) < 2:
            return
        self.model.train()
        cw = self._class_weights(y_t)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=min(BATCH_SIZE, len(X_t)), shuffle=True,
            drop_last=(len(X_t) > BATCH_SIZE))

        # Get PC matrix for manifold-guided contrastive (may be None)
        pc_mat_np, pc_emap = None, None
        if self._use_contrastive and self.use_causal:
            pc_mat_np, pc_emap = self._pc_forecaster.get_causal_coupling()

        pc_mat = None
        if pc_mat_np is not None:
            pc_mat = torch.FloatTensor(pc_mat_np).to(self.device)

        for _ in range(epochs):
            for bf, bl in dl:
                if len(bf) < 2:
                    continue
                self.optimizer.zero_grad()

                # Forward
                logits, feat, proj = self.model(bf, return_all=True)

                # CE loss
                loss_ce = nn.CrossEntropyLoss(weight=cw)(logits, bl)

                # Contrastive loss (vanilla SupCon per mini-batch;
                # PC guidance applied only when batch covers full entity set)
                loss_con = torch.tensor(0.0, device=self.device)
                if self._use_contrastive and len(bf) >= 4:
                    # Use manifold guidance only if mini-batch == full batch
                    # AND entity_order length matches X_t length (i.e. no
                    # replay/combined batch where entity_order would be short)
                    use_pc = (pc_mat is not None and
                              len(bf) == len(X_t) and
                              self._current_entity_order is not None and
                              len(self._current_entity_order) == len(X_t))
                    loss_con = sup_con_loss(
                        proj, bl, temperature=self._contrastive_tau,
                        pc_matrix=pc_mat if use_pc else None,
                        entity_order=self._current_entity_order if use_pc else None,
                        pc_entity_map=pc_emap if use_pc else None)

                # Total loss
                loss = loss_ce + self._contrastive_alpha * loss_con

                # EWC
                if use_ewc and self.training_step > 0 and ewc_scale > 0:
                    loss = loss + ewc_scale * self.model.compute_ewc_loss()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        # Update prototypes after training
        if self._use_prototype:
            self.model.eval()
            with torch.no_grad():
                _, feat_all = self.model(X_t)
                beta = (PROTOTYPE_BETA_DRIFT
                        if self._drift_state == 'drift'
                        else PROTOTYPE_BETA_STABLE)
                self.model.prototype.update(feat_all, y_t, beta_override=beta)

        # Fisher on balanced replay buffer
        if use_ewc and not skip_fisher:
            replay_X, replay_y = self._get_replay_data()
            if replay_X is not None and len(replay_X) >= 32:
                replay_X = replay_X.to(self.device)
                replay_y = replay_y.to(self.device)
                bs = min(BATCH_SIZE, len(replay_X))
                ds_f = torch.utils.data.TensorDataset(replay_X, replay_y)
                dl_f = torch.utils.data.DataLoader(ds_f, batch_size=bs)
                self.model.update_fisher(dl_f, self.device)

        self.training_step += 1

    # =========================================================================
    # Update (layered adaptation: stable/pre_drift/drift)
    # =========================================================================

    def update(self, X: np.ndarray, y: np.ndarray):
        """Update with layered adaptation strategy."""
        t0 = time.perf_counter()

        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)

        if feat_tensor is None:
            self._window_count += 1
            self._cached_ecba = None
            self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000
            return

        if self.model is None:
            self._init_model(len(self._feature_template))

        entity_labels = self._extract_entity_labels(y, entity_ids, entity_order)
        self._current_entity_order = entity_order  # for contrastive loss

        feat_dev = feat_tensor.to(self.device)
        labels_dev = entity_labels.to(self.device)

        # --- Label distribution reversal detection ---
        current_attack_ratio = float(entity_labels.float().mean())
        self._attack_ratio_history.append(current_attack_ratio)
        reversal_triggered = False

        if len(self._attack_ratio_history) >= 5:
            recent_avg = np.mean(list(self._attack_ratio_history)[-6:-1])
            if abs(current_attack_ratio - recent_avg) > 0.3:
                # Reversal: reset Fisher + classifier, boost prototype weight
                self.model.fisher_dict = {}
                self.model.optpar_dict = {}
                nn.init.xavier_uniform_(self.model.classifier.weight)
                nn.init.zeros_(self.model.classifier.bias)
                self.model.unfreeze_backbone()
                self._prototype_weight = PROTOTYPE_WEIGHT_REVERSAL

                replay_X, replay_y = self._get_replay_data()
                if replay_X is not None and len(replay_X) >= 16:
                    combined_X = torch.cat([feat_dev.cpu(), replay_X]).to(self.device)
                    combined_y = torch.cat([labels_dev.cpu(), replay_y]).to(self.device)
                    self._train_batch(combined_X, combined_y, epochs=5, use_ewc=False)
                else:
                    self._train_batch(feat_dev, labels_dev, epochs=5, use_ewc=False)
                self._prediction_threshold = 0.5
                self._calibrate_threshold()
                reversal_triggered = True

        # --- Layered adaptation (if no reversal) ---
        if not reversal_triggered:
            if self._drift_state == 'drift':
                # Full adaptation: unfreeze backbone, aggressive LR, no EWC
                self.model.unfreeze_backbone()
                self._consecutive_stable = 0
                self._set_lr(LEARNING_RATE * LR_DRIFT_SCALE)
                drift_use_ewc = False if self._use_layered_ewc else self.use_ewc
                self._train_batch(feat_dev, labels_dev, epochs=2, use_ewc=drift_use_ewc)
                self._prototype_weight = PROTOTYPE_WEIGHT_BASE
            elif self._drift_state == 'pre_drift':
                # Preparation: unfreeze backbone, contrastive warm-up, reduced EWC
                self.model.unfreeze_backbone()
                self._consecutive_stable = 0
                self._set_lr(LEARNING_RATE * LR_PRE_DRIFT_SCALE)
                pre_drift_ewc_scale = EWC_PRE_DRIFT_SCALE if self._use_layered_ewc else 1.0
                self._train_batch(feat_dev, labels_dev, epochs=1,
                                  use_ewc=self.use_ewc, ewc_scale=pre_drift_ewc_scale)
            elif current_attack_ratio < 0.05:
                # Benign window: conditionally freeze backbone, skip Fisher
                self._consecutive_stable += 1
                if self._consecutive_stable >= self._freeze_after_stable:
                    self.model.freeze_backbone()
                self._set_lr(LEARNING_RATE)
                self._train_batch(feat_dev, labels_dev, epochs=2,
                                  use_ewc=self.use_ewc, skip_fisher=True)
            else:
                # Stable: conditionally freeze backbone, full EWC
                self._consecutive_stable += 1
                if self._consecutive_stable >= self._freeze_after_stable:
                    self.model.freeze_backbone()
                self._set_lr(LEARNING_RATE)
                self._train_batch(feat_dev, labels_dev, epochs=2, use_ewc=self.use_ewc)

            # Decay prototype weight back toward base
            self._prototype_weight = max(
                PROTOTYPE_WEIGHT_BASE,
                self._prototype_weight * 0.9)

        self._drift_state = 'stable'  # reset for next window

        # Update PC entity history
        self._ensure_entity_history_updated(X)

        # Update temporal buffer (AFTER training, so next window uses these features)
        self._temporal_buffer.update(entity_order, feat_tensor)

        # Store replay samples
        self._store_replay_samples(feat_dev, labels_dev)

        # Dynamic threshold recalibration
        if not reversal_triggered:
            if current_attack_ratio < 0.05:
                self.model.eval()
                with torch.no_grad():
                    logits, feat = self.model(feat_dev)
                    if self._use_prototype:
                        scores = self.model.predict_with_prototypes(
                            feat, logits, self._prototype_weight).cpu().numpy()
                    else:
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                max_score = float(scores.max()) if len(scores) > 0 else 0.5
                self._prediction_threshold = max(self._prediction_threshold,
                                                  min(max_score + 0.02, 0.99))
            elif self._window_count < 20 or self._window_count % 5 == 4:
                self._calibrate_threshold()

        self._window_count += 1
        self._cached_ecba = None
        self._current_entity_order = None
        self._last_timing['training_ms'] = (time.perf_counter() - t0) * 1000

    # =========================================================================
    # Drift detection (three-level)
    # =========================================================================

    def detect_drift(self, X: np.ndarray) -> tuple:
        """Three-level drift detection with proactive pre-alert and layered response.

        Level 1 (stable): No drift signals -> freeze backbone, full EWC
        Level 2 (pre_drift): Early signals -> unfreeze backbone, contrastive warm-up
        Level 3 (drift): PC confirms -> full adaptation, no EWC
        """
        t0 = time.perf_counter()

        pc_detected = False
        confidence = 0.0
        pre_alert = False

        if self.use_causal:
            self._ensure_entity_history_updated(X)

            # Channel 1: Entity novelty signal
            novelty_ratio = self._compute_entity_novelty(X)

            # Channel 2: PC-DriftForecasting
            result = self._pc_forecaster.forecast_drift()
            pc_detected = result.get('drift_detected', False)
            confidence = result.get('confidence', 0.0)
            phi_k = result.get('risk_score', 0.0)

            self._risk_history.append(phi_k)

            # Channel 3: Risk trend extrapolation
            trend_alert = self._extrapolate_risk_trend()

            # Combine into three-level state
            if pc_detected:
                self._drift_state = 'drift'
            elif trend_alert or novelty_ratio > 0.4:
                self._drift_state = 'pre_drift'
                pre_alert = True
                confidence = max(confidence, 0.3)
            else:
                self._drift_state = 'stable'
        else:
            self._drift_state = 'stable'

        detected = pc_detected or pre_alert
        self._last_drift_result = (detected, confidence)
        self._last_timing['drift_detection_ms'] = (time.perf_counter() - t0) * 1000
        return detected, confidence

    def _compute_entity_novelty(self, X):
        """Compute ratio of unseen entities in current window."""
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)
        if entity_order is None:
            return 0.0

        current_entities = set(entity_order)
        if not self._known_entities:
            self._known_entities = current_entities.copy()
            self._entity_novelty_history.append(0.0)
            return 0.0

        new_entities = current_entities - self._known_entities
        novelty_ratio = len(new_entities) / max(len(current_entities), 1)

        self._known_entities.update(current_entities)
        self._entity_novelty_history.append(novelty_ratio)

        if len(self._entity_novelty_history) >= 3:
            recent_avg = np.mean(list(self._entity_novelty_history)[-5:])
            if novelty_ratio > 2 * recent_avg + 0.1:
                return novelty_ratio

        return novelty_ratio

    def _extrapolate_risk_trend(self):
        """Predict if risk score will exceed threshold in the next window."""
        if len(self._risk_history) < 3:
            return False

        recent = list(self._risk_history)[-3:]
        slope = (recent[-1] - recent[-3]) / 2.0
        phi_extrapolated = recent[-1] + slope

        threshold = self._pc_forecaster.adaptive_threshold()

        return phi_extrapolated > threshold and slope > 0

    def _ensure_entity_history_updated(self, X):
        """Update PC entity history once per window (idempotent)."""
        if self._window_count == getattr(self, '_last_history_window', -1):
            return
        feat_tensor, entity_order, entity_ids = self._get_cached_ecba(X)
        if feat_tensor is not None:
            df = self._numpy_to_dataframe(X)
            entity_feats = self._aggregator.aggregate_features(df, window_id=self._window_count)
            self._pc_forecaster.update_entity_history(entity_feats)
        self._last_history_window = self._window_count

    # =========================================================================
    # Misc
    # =========================================================================

    def get_timing(self) -> dict:
        return dict(self._last_timing)

    def reset(self):
        super().reset()
        self.model = None
        self.optimizer = None
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.training_step = 0
        self._last_timing = {}
        self._window_count = 0
        self._drift_state = 'stable'
        self._last_drift_result = (False, 0.0)
        self._cached_ecba = None
        self._cached_window_id = -1
        self._prediction_threshold = 0.5
        self._known_entities = set()
        self._entity_novelty_history = deque(maxlen=20)
        self._risk_history = deque(maxlen=30)
        self._attack_ratio_history = deque(maxlen=10)
        self._replay_pos_X = deque(maxlen=250)
        self._replay_pos_y = deque(maxlen=250)
        self._replay_neg_X = deque(maxlen=250)
        self._replay_neg_y = deque(maxlen=250)
        # v6 state
        self._consecutive_stable = 0
        self._reversal_cooldown = 0
        self._prev_pc_detected = False
        self._temporal_buffer = EntityFeatureBuffer(buffer_len=TEMPORAL_BUFFER_LEN)
        self._prototype_weight = PROTOTYPE_WEIGHT_BASE
        self._current_entity_order = None
        # Rebuild PC forecaster
        self._pc_forecaster = PCDriftForecaster(
            E=PC_E, tau=PC_TAU,
            alpha=PC_ALPHA, beta=PC_BETA, gamma=PC_GAMMA,
            sigma=self._preset.get('pc_sigma', PC_SIGMA),
            theta_lid=PC_THETA_LID, K_lid=PC_K_LID,
            lambda_decay=PC_LAMBDA_DECAY, eta=PC_ETA,
            baseline_window=self._preset.get('pc_baseline_window', PC_BASELINE_WINDOW),
            causal_features=self._preset.get('causal_features', []),
            min_history=MIN_WINDOWS_CAUSALITY,
        )

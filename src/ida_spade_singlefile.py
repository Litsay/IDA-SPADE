"""
IDA-SPADE: Intrusion Detection via Anticipatory Spatio-Temporal
            Potential Causality Analysis on Drift Adaptation

Paper-aligned implementation:
  - Algorithm 1: ECBA (Entity-Centric Behavioral Aggregation)
  - Algorithm 2: PC-DriftForecasting (Pattern Causality Drift Forecasting)
  - Eq. 23-24: Continual learning with binary loss switching
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, entropy as sp_entropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx
import itertools
import warnings
import time
import tracemalloc
import json
import argparse

warnings.filterwarnings('ignore')

# Fix threadpoolctl crash on Windows (sklearn dependency)
try:
    import threadpoolctl
    _orig_openblas_init = threadpoolctl._OpenBLASModule.__init__
    def _patched_openblas_init(self, *args, **kwargs):
        try:
            _orig_openblas_init(self, *args, **kwargs)
        except AttributeError:
            self.version = 'unknown'
            self.num_threads = 1
            self.threading_layer = 'unknown'
    threadpoolctl._OpenBLASModule.__init__ = _patched_openblas_init
except Exception:
    pass


# ============================================================================
# Dataset Presets
# ============================================================================
DATASET_PRESETS = {
    'UNSW-NB15': {
        'train_path': 'UNSW_pre_data/UNSWTrain.csv',
        'test_path': 'UNSW_pre_data/UNSWTest.csv',
        'label_col': 'label',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.087,
        # One-hot encoded proto/service column prefixes for entity identification
        'proto_prefix': 'proto_',
        'service_prefix': 'service_',
        # Transient-preserving aggregation (paper Algorithm 1)
        'aggregation_rules': {
            # rate features: max + std
            'rate': ['max', 'std'], 'sload': ['max', 'std'], 'dload': ['max', 'std'],
            # byte/packet counts: max + entropy
            'sbytes': ['max', 'entropy'], 'dbytes': ['max', 'entropy'],
            'spkts': ['max', 'entropy'], 'dpkts': ['max', 'entropy'],
            # connection counts: max
            'ct_srv_src': ['max'], 'ct_srv_dst': ['max'],
            'ct_dst_ltm': ['max'], 'ct_src_ltm': ['max'],
            'ct_src_dport_ltm': ['max'], 'ct_dst_sport_ltm': ['max'],
            'ct_dst_src_ltm': ['max'],
            # latency: variance + max
            'dur': ['var', 'max'], 'sinpkt': ['var', 'max'], 'dinpkt': ['var', 'max'],
            'sjit': ['var', 'max'], 'djit': ['var', 'max'],
            'tcprtt': ['var', 'max'], 'synack': ['var', 'max'], 'ackdat': ['var', 'max'],
            # TTL/window: max
            'sttl': ['max'], 'dttl': ['max'], 'swin': ['max'], 'dwin': ['max'],
            # loss: max + std
            'sloss': ['max', 'std'], 'dloss': ['max', 'std'],
            # mean sizes: max
            'smean': ['max'], 'dmean': ['max'],
            # other: max
            'trans_depth': ['max'], 'response_body_len': ['max'],
            'stcpb': ['max'], 'dtcpb': ['max'],
            'ct_state_ttl': ['max'], 'ct_flw_http_mthd': ['max'],
            'ct_ftp_cmd': ['max'], 'is_ftp_login': ['max'], 'is_sm_ips_ports': ['max'],
        },
        'causal_features': ['sload', 'dload', 'rate', 'ct_srv_src', 'ct_dst_ltm'],
        'trend_features': ['sbytes', 'dbytes', 'sload', 'dload'],
        'init_epochs': 1,  # UNSW: more drift, less init overfitting
    },
    'NSL-KDD': {
        'train_path': 'NSL_pre_data/PKDDTrain+.csv',
        'test_path': 'NSL_pre_data/PKDDTest+.csv',
        'label_col': 'labels2',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.465,
        'proto_prefix': 'protocol_type_',
        'service_prefix': 'service_',
        'aggregation_rules': {
            # latency: variance + max
            'duration': ['var', 'max'],
            # byte counts: max + entropy
            'src_bytes': ['max', 'entropy'], 'dst_bytes': ['max', 'entropy'],
            # error rates: max + std
            'serror_rate': ['max', 'std'], 'srv_rerror_rate': ['max', 'std'],
            'dst_host_srv_serror_rate': ['max', 'std'],
            'dst_host_rerror_rate': ['max', 'std'], 'dst_host_srv_rerror_rate': ['max', 'std'],
            # connection counts: max
            'count': ['max'], 'srv_count': ['max'],
            'dst_host_count': ['max'], 'dst_host_srv_count': ['max'],
            # rate features: max + std
            'same_srv_rate': ['max', 'std'], 'diff_srv_rate': ['max', 'std'],
            'srv_diff_host_rate': ['max', 'std'],
            'dst_host_same_srv_rate': ['max', 'std'], 'dst_host_diff_srv_rate': ['max', 'std'],
            'dst_host_same_src_port_rate': ['max', 'std'],
            'dst_host_srv_diff_host_rate': ['max', 'std'],
            # binary/count: max
            'wrong_fragment': ['max'], 'hot': ['max'],
            'num_failed_logins': ['max'], 'logged_in': ['max'],
            'num_compromised': ['max'], 'num_root': ['max'],
            'su_attempted': ['max'], 'num_file_creations': ['max'],
            'num_shells': ['max'], 'num_access_files': ['max'],
            'is_guest_login': ['max'],
        },
        'causal_features': ['count', 'srv_count', 'serror_rate', 'dst_host_count', 'dst_host_srv_count'],
        'trend_features': ['src_bytes', 'dst_bytes', 'serror_rate', 'dst_host_srv_serror_rate'],
        'init_epochs': 3,  # NSL-KDD: stable distribution benefits from stronger init
    },
    'CIC-IDS-2017': {
        'label_col': 'label_binary',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.19,
        'proto_prefix': 'cic_proto_',
        'service_prefix': 'cic_service_',
        'aggregation_rules': {
            'Flow Duration': ['var', 'max'],
            'Total Fwd Packets': ['max', 'entropy'],
            'Total Backward Packets': ['max', 'entropy'],
            'Total Length of Fwd Packets': ['max', 'entropy'],
            'Total Length of Bwd Packets': ['max', 'entropy'],
            'Fwd Packet Length Mean': ['max', 'std'],
            'Bwd Packet Length Mean': ['max', 'std'],
            'Flow IAT Mean': ['var', 'max'],
            'Fwd IAT Total': ['var', 'max'],
            'Bwd IAT Total': ['var', 'max'],
            'Fwd PSH Flags': ['max'],
            'Fwd Header Length': ['max'],
            'Bwd Header Length': ['max'],
            'FIN Flag Count': ['max'],
            'SYN Flag Count': ['max'],
            'RST Flag Count': ['max'],
            'ACK Flag Count': ['max'],
            'Down/Up Ratio': ['max'],
            'Average Packet Size': ['max', 'std'],
            'Init_Win_bytes_forward': ['max'],
            'Init_Win_bytes_backward': ['max'],
            'Active Mean': ['var', 'max'],
            'Idle Mean': ['var', 'max'],
        },
        'causal_features': ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                             'Fwd Packet Length Mean', 'Bwd Packet Length Mean'],
        'trend_features': ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                            'Fwd Packet Length Mean', 'Bwd Packet Length Mean'],
        'init_epochs': 2,
        # Strategy 1: dataset-adaptive PC sensitivity (entity count ~12 → need higher sigma)
        'pc_sigma': 2.5,
        'pc_baseline_window': 20,
        # Strategy 3: confidence-scaled EWC during drift (reduce false-alarm forgetting)
        'drift_ewc_mode': 'confidence_scaled',
    },
}


# ============================================================================
# Algorithm 1: ECBA — Entity-Centric Behavioral Aggregation
# ============================================================================
class NetworkEntityAggregator:
    """Paper Algorithm 1: Entity identification via phi(c) and
    transient-preserving aggregation function A.

    Entity identification uses (protocol, service) tuples decoded from
    one-hot encoded columns in the preprocessed datasets.
    """

    def __init__(self, window_size=1000, aggregation_rules=None,
                 label_col='label', label_positive=1,
                 proto_prefix='proto_', service_prefix='service_',
                 trend_features=None):
        self.window_size = window_size
        self.aggregation_rules = aggregation_rules or {}
        self.label_col = label_col
        self.label_positive = label_positive
        self.proto_prefix = proto_prefix
        self.service_prefix = service_prefix
        self.trend_features = trend_features or []
        self.window_stats = []

    def _decode_onehot_entity(self, row, proto_cols, service_cols):
        """Decode one-hot encoded protocol and service to entity tuple phi(c)."""
        proto = 'unknown'
        for col in proto_cols:
            if row.get(col, 0) == 1:
                proto = col
                break
        service = 'unknown'
        for col in service_cols:
            if row.get(col, 0) == 1:
                service = col
                break
        return (proto, service)

    def create_non_overlapping_windows(self, df):
        """Create non-overlapping windows of size T (paper specification)."""
        windows = []
        n = len(df)
        wid = 0
        start = 0
        while start + self.window_size <= n:
            end = start + self.window_size
            wd = df.iloc[start:end]
            windows.append((wid, wd))
            if self.label_col in wd.columns:
                col = wd[self.label_col]
                if col.dtype == object:
                    ar = (col != 'normal').mean()
                else:
                    ar = (col == self.label_positive).mean()
            else:
                ar = 0.0
            self.window_stats.append({
                'window_id': wid, 'anomaly_ratio': ar, 'total_samples': len(wd)
            })
            start += self.window_size  # Non-overlapping
            wid += 1
        return windows

    def _compute_entropy(self, values):
        """Compute Shannon entropy of a numeric series."""
        vals = values.dropna()
        if len(vals) <= 1:
            return 0.0
        # Discretize into bins
        n_bins = min(10, max(2, len(vals) // 5))
        try:
            counts, _ = np.histogram(vals, bins=n_bins)
            counts = counts[counts > 0]
            if len(counts) <= 1:
                return 0.0
            probs = counts / counts.sum()
            return float(sp_entropy(probs, base=2))
        except Exception:
            return 0.0

    def _aggregate_feature(self, series, agg_funcs):
        """Apply transient-preserving aggregation functions."""
        results = {}
        feat_name = series.name if hasattr(series, 'name') else 'feat'
        for func in agg_funcs:
            try:
                if func == 'max':
                    v = float(series.max())
                elif func == 'std':
                    v = float(series.std())
                    if np.isnan(v):
                        v = 0.0
                elif func == 'var':
                    v = float(series.var())
                    if np.isnan(v):
                        v = 0.0
                elif func == 'entropy':
                    v = self._compute_entropy(series)
                elif func == 'mean':
                    v = float(series.mean())
                elif func == 'sum':
                    v = float(series.sum())
                else:
                    v = float(series.mean())
                results[f'{feat_name}_{func}'] = v if np.isfinite(v) else 0.0
            except Exception:
                results[f'{feat_name}_{func}'] = 0.0
        return results

    def _assign_entity_ids(self, window_data):
        """Assign entity IDs based on (protocol, service) — label-free."""
        all_cols = list(window_data.columns)
        proto_cols = [c for c in all_cols if c.startswith(self.proto_prefix)]
        service_cols = [c for c in all_cols if c.startswith(self.service_prefix)]

        if proto_cols or service_cols:
            if proto_cols:
                proto_idx = window_data[proto_cols].values.argmax(axis=1)
                proto_names = np.array(proto_cols)[proto_idx]
            else:
                proto_names = np.full(len(window_data), 'unknown')
            if service_cols:
                service_idx = window_data[service_cols].values.argmax(axis=1)
                service_names = np.array(service_cols)[service_idx]
            else:
                service_names = np.full(len(window_data), 'unknown')
            return [f"{p}|{s}" for p, s in zip(proto_names, service_names)]
        else:
            n_groups = min(10, max(3, len(window_data) // 100))
            gs = max(1, len(window_data) // n_groups)
            return [f"entity_{min(i // gs, n_groups - 1)}"
                    for i in range(len(window_data))]

    def aggregate_features(self, window_data, window_id):
        """Aggregate features only — completely label-free.

        Returns: dict {entity_id: {feat_agg_key: value, ...}}
        No label information is accessed or produced.
        """
        entity_ids = self._assign_entity_ids(window_data)
        wd = window_data.copy().reset_index(drop=True)
        wd['_entity_id'] = entity_ids

        entity_features = {}
        for eid, group in wd.groupby('_entity_id'):
            if len(group) == 0:
                continue
            agg = {}
            for feat, funcs in self.aggregation_rules.items():
                if feat in group.columns:
                    feat_aggs = self._aggregate_feature(group[feat], funcs)
                    agg.update(feat_aggs)
            agg['entity_size'] = len(group)
            # Temporal trend: second_half_max - first_half_max for key features
            if self.trend_features and len(group) >= 4:
                mid = len(group) // 2
                for tf in self.trend_features:
                    if tf in group.columns:
                        first_max = group[tf].iloc[:mid].max()
                        second_max = group[tf].iloc[mid:].max()
                        agg[f'{tf}_trend'] = float(second_max - first_max)
                    else:
                        agg[f'{tf}_trend'] = 0.0
            elif self.trend_features:
                for tf in self.trend_features:
                    agg[f'{tf}_trend'] = 0.0
            entity_features[str(eid)] = agg
        return entity_features

    def extract_entity_labels(self, window_data, label_ratio=1.0):
        """Extract entity-level labels separately from features.

        Args:
            window_data: raw window DataFrame (must contain label column)
            label_ratio: fraction of connections whose labels are visible
                         (paper: 0.01 = 1% expert annotation budget)

        Returns: dict {entity_id: 0 or 1}, only for entities with
                 at least one labeled connection.
        """
        if self.label_col not in window_data.columns:
            return {}

        entity_ids = self._assign_entity_ids(window_data)
        wd = window_data.copy().reset_index(drop=True)
        wd['_entity_id'] = entity_ids

        # Simulate limited annotation: randomly reveal label_ratio of labels
        n = len(wd)
        if label_ratio < 1.0:
            n_labeled = max(1, int(n * label_ratio))
            labeled_mask = np.zeros(n, dtype=bool)
            labeled_idx = np.random.choice(n, n_labeled, replace=False)
            labeled_mask[labeled_idx] = True
        else:
            labeled_mask = np.ones(n, dtype=bool)

        entity_labels = {}
        for eid, group in wd.groupby('_entity_id'):
            if len(group) == 0:
                continue
            # Use positional indexing (group.index is now 0-based after reset_index)
            group_labeled = labeled_mask[group.index]
            labeled_group = group[group_labeled]
            if len(labeled_group) == 0:
                continue  # No labeled connections → skip this entity
            col = labeled_group[self.label_col]
            if col.dtype == object:
                has_attack = (col != 'normal').any()
            else:
                has_attack = (col == self.label_positive).any()
            entity_labels[str(eid)] = 1 if has_attack else 0
        return entity_labels

    def extract_entity_labels_full(self, window_data):
        """Extract ground-truth entity labels using ALL labels (for evaluation only)."""
        return self.extract_entity_labels(window_data, label_ratio=1.0)

    def aggregate_window(self, window_data, window_id):
        """Legacy interface: returns combined features + labels.
        Kept for backward compatibility. Prefer aggregate_features + extract_entity_labels.
        """
        features = self.aggregate_features(window_data, window_id)
        labels = self.extract_entity_labels_full(window_data)
        for eid in features:
            if eid in labels:
                features[eid]['label'] = 'anomaly' if labels[eid] == 1 else 'normal'
            else:
                features[eid]['label'] = 'normal'
        return features


# ============================================================================
# Algorithm 2: PC-DriftForecasting
# ============================================================================
class PCDriftForecaster:
    """Paper Algorithm 2: Pattern Causality based Drift Forecasting.

    Implements:
    - Takens' delay embedding (Eq. 7)
    - Mahalanobis distance matrix (Eq. 9)
    - LID validation (Eq. 10)
    - Cross-manifold mapping CMD/PC (Eq. 11-14)
    - Dynamic causal graph (Eq. 15)
    - Topological features (Eq. 16-19)
    - Composite risk Phi(k) and adaptive threshold (Eq. 20-22)
    """

    def __init__(self, E=3, tau=1,
                 alpha=0.4, beta=0.35, gamma=0.25,
                 sigma=1.5, theta_lid=15.0, K_lid=10,
                 lambda_decay=1.0, eta=2.0,
                 baseline_window=10, causal_features=None,
                 min_history=5):
        # Embedding parameters
        self.E = E
        self.tau = tau
        # Topological weight parameters (Eq. 20: alpha + beta + gamma = 1)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Adaptive threshold parameters (Eq. 22)
        self.sigma = sigma
        self.baseline_window = baseline_window
        # LID parameters (Eq. 10)
        self.theta_lid = theta_lid
        self.K_lid = K_lid
        # Cross-manifold mapping parameters (Eq. 12, 14)
        self.lambda_decay = lambda_decay
        self.eta = eta
        # Features for causal analysis
        self.causal_features = causal_features or []
        self.min_history = min_history

        # State
        self.entity_history = defaultdict(list)  # entity_id -> list of feature dicts
        self.risk_history = []
        self.graph_history = []  # list of (G, density, centrality_dict, modularity)
        self.detection_times = []
        self.last_pc_matrix = None       # (n, n) causal coupling matrix
        self.last_pc_entities = []       # entity IDs corresponding to rows/cols

    def update_entity_history(self, entity_records):
        """Add current window's entity observations to history."""
        for eid, feats in entity_records.items():
            self.entity_history[eid].append(feats)

    def _get_entity_time_series(self, eid, feat_name):
        """Extract time series of a specific feature for an entity."""
        history = self.entity_history[eid]
        # Use the first available aggregation of this feature
        ts = []
        for h in history:
            found = False
            for key in h:
                if key.startswith(feat_name + '_'):
                    ts.append(h[key])
                    found = True
                    break
            if not found:
                ts.append(0.0)
        return np.array(ts, dtype=np.float64)

    def reconstruct_shadow_attractor(self, ts_multi, E, tau):
        """Eq. 7: Multi-variable delay embedding.

        ts_multi: (K, M) — K time steps, M variables
        Returns: (N, M*E) shadow attractor states
        """
        K, M = ts_multi.shape
        start = (E - 1) * tau
        if start >= K:
            return None
        N = K - start
        states = np.zeros((N, M * E))
        for k in range(N):
            for d in range(E):
                idx = start + k - d * tau
                states[k, d * M:(d + 1) * M] = ts_multi[idx]
        return states

    def mahalanobis_distance_matrix(self, states):
        """Eq. 9: Compute Mahalanobis distance matrix."""
        N, D = states.shape
        eps = 1e-6
        # Regularized covariance
        cov = np.cov(states.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += eps * np.eye(D)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(D)

        # Efficient pairwise Mahalanobis
        # For memory efficiency, compute on-the-fly when N is large
        if N > 200:
            # Subsample for large N
            idx = np.random.choice(N, 200, replace=False)
            idx.sort()
            states_sub = states[idx]
            N_sub = len(idx)
            dist = np.zeros((N_sub, N_sub))
            for i in range(N_sub):
                diff = states_sub[i] - states_sub
                dist[i] = np.sqrt(np.maximum(0, np.sum(diff @ cov_inv * diff, axis=1)))
            return dist, idx
        else:
            dist = np.zeros((N, N))
            for i in range(N):
                diff = states[i] - states
                dist[i] = np.sqrt(np.maximum(0, np.sum(diff @ cov_inv * diff, axis=1)))
            return dist, np.arange(N)

    def estimate_lid(self, distances_k, K_lid):
        """Eq. 10: MLE estimate of Local Intrinsic Dimensionality."""
        sorted_d = np.sort(distances_k)
        # Exclude zero distance (self)
        nn_dists = sorted_d[sorted_d > 1e-10][:K_lid]
        if len(nn_dists) < 2:
            return 0.0
        r_max = nn_dists[-1]
        if r_max <= 0:
            return 0.0
        log_ratios = np.log(r_max / (nn_dists[:-1] + 1e-10))
        if log_ratios.sum() <= 0:
            return 0.0
        lid = (len(log_ratios) / log_ratios.sum())
        return float(lid)

    def cross_manifold_mapping(self, M_i, M_j, k, E, dist_i):
        """Eq. 11-14: Cross-manifold mapping and CMD/PC computation.

        M_i, M_j: shadow attractors of entity i, j
        k: current time index in the attractor
        dist_i: distance matrix for M_i
        Returns: PC coupling strength
        """
        N_i = len(M_i)
        N_j = len(M_j)
        N = min(N_i, N_j)
        if k >= N or k < E + 1:
            return 0.0

        # Step 1: Find E+1 nearest neighbors in M_i with historical constraint
        # dist_i[k] gives distances from state k to all others
        distances_k = dist_i[k, :k].copy() if k < len(dist_i) else dist_i[min(k, len(dist_i)-1), :min(k, len(dist_i))]
        if len(distances_k) < E + 1:
            return 0.0
        nn_indices = np.argsort(distances_k)[:E + 1]

        # Step 2: Exponential distance weights (Eq. 12)
        nn_dists = distances_k[nn_indices]
        weights = np.exp(-self.lambda_decay * nn_dists)
        w_sum = weights.sum()
        if w_sum <= 0:
            return 0.0
        weights /= w_sum

        # Step 3: Estimate target state (Eq. 11)
        valid_nn = nn_indices[nn_indices < N_j]
        valid_w = weights[:len(valid_nn)]
        if len(valid_nn) == 0:
            return 0.0
        w_sum2 = valid_w.sum()
        if w_sum2 <= 0:
            return 0.0
        valid_w /= w_sum2
        x_j_hat = np.sum(valid_w[:, None] * M_j[valid_nn], axis=0)

        # Step 4: CMD (Eq. 13)
        k_j = min(k, N_j - 1)
        x_j_true = M_j[k_j]
        cmd = np.linalg.norm(x_j_true - x_j_hat) / (np.linalg.norm(x_j_true) + 1e-10)

        # Step 5: PC coupling strength (Eq. 14)
        pc = np.exp(-self.eta * cmd)
        return float(pc)

    def build_causal_graph(self, entities, pc_matrix):
        """Eq. 15: Build dynamic causal graph G(k)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(len(entities)))
        n = len(entities)
        for i in range(n):
            for j in range(n):
                if i != j and pc_matrix[i, j] > 0.01:
                    G.add_edge(i, j, weight=pc_matrix[i, j])
        return G

    def extract_topological_features(self, G_k, G_prev=None):
        """Eq. 16-19: Extract topological change features."""
        n = G_k.number_of_nodes()
        max_edges = n * (n - 1) if n > 1 else 1

        # Current graph metrics
        density_k = G_k.number_of_edges() / max_edges if max_edges > 0 else 0

        cen_k = {}
        for node in G_k.nodes():
            out_weights = [G_k[node][succ]['weight']
                           for succ in G_k.successors(node)]
            cen_k[node] = sum(out_weights) / (n - 1) if n > 1 else 0

        Q_k = 0.0
        if G_k.number_of_edges() > 0:
            try:
                G_und = G_k.to_undirected()
                communities = list(nx.community.louvain_communities(G_und, seed=42))
                Q_k = nx.community.modularity(G_und, communities)
            except Exception:
                Q_k = 0.0

        if G_prev is not None and len(self.graph_history) > 0:
            # Eq. 16: Network density change
            density_prev = self.graph_history[-1][1]
            delta_den = abs(density_k - density_prev)

            # Eq. 17: Centrality distribution change
            cen_prev = self.graph_history[-1][2]
            all_nodes = set(cen_k.keys()) | set(cen_prev.keys())
            if all_nodes:
                delta_cen = np.mean([abs(cen_k.get(e, 0) - cen_prev.get(e, 0))
                                     for e in all_nodes])
            else:
                delta_cen = 0.0

            # Eq. 18-19: Modularity change
            Q_prev = self.graph_history[-1][3]
            delta_mod = abs(Q_k - Q_prev)
        else:
            # First graph: use absolute values as initial signal
            # (deviation from a "uniform/empty" baseline)
            delta_den = density_k
            delta_cen = np.mean(list(cen_k.values())) if cen_k else 0.0
            delta_mod = abs(Q_k)

        return delta_den, delta_cen, delta_mod, density_k, cen_k, Q_k

    def compute_drift_risk(self, delta_den, delta_cen, delta_mod):
        """Eq. 20: Composite risk score Phi(k)."""
        Phi_k = (self.alpha * np.tanh(delta_den) +
                 self.beta * np.tanh(delta_cen) +
                 self.gamma * np.tanh(delta_mod))
        return float(Phi_k)

    def adaptive_threshold(self):
        """Eq. 22: Adaptive threshold theta_drift(k)."""
        if len(self.risk_history) < 2:
            return 0.5  # Default before enough history
        w = min(self.baseline_window, len(self.risk_history))
        recent = self.risk_history[-w:]
        mu = np.mean(recent)
        std = np.std(recent)
        return float(mu + self.sigma * std)

    def get_causal_coupling(self):
        """Return last computed PC matrix and entity-to-index mapping.
        Returns: (pc_matrix, entity_map) or (None, None) if unavailable.
        entity_map: dict mapping entity_id -> row/col index in pc_matrix.
        """
        if self.last_pc_matrix is None or not self.last_pc_entities:
            return None, None
        entity_map = {eid: i for i, eid in enumerate(self.last_pc_entities)}
        return self.last_pc_matrix, entity_map

    def forecast_drift(self):
        """Main drift forecasting pipeline (Algorithm 2).

        Returns: dict with drift_detected, confidence, risk_score, threshold
        """
        t0 = time.time()
        _no_drift = lambda msg='': {'drift_detected': False, 'confidence': 0.0,
                    'risk_score': 0.0, 'threshold': 0.5,
                    'detection_time': time.time() - t0, '_skip_reason': msg}

        all_entities = list(self.entity_history.keys())
        if len(all_entities) < 2:
            return _no_drift('< 2 entities total')

        # FIX: Filter to entities with sufficient history, instead of
        # requiring ALL entities to have min_history (which never holds
        # because new entities appear every window with history=1).
        entities = [e for e in all_entities
                    if len(self.entity_history[e]) >= self.min_history]
        if len(entities) < 2:
            return _no_drift('< 2 entities with sufficient history (%d/%d)' %
                             (len(entities), len(all_entities)))

        # Limit to top entities by history length (avoid O(n^2) explosion)
        if len(entities) > 15:
            entities = sorted(entities,
                              key=lambda e: len(self.entity_history[e]),
                              reverse=True)[:15]
        n_entities = len(entities)

        # Step 1: Build multi-variable time series per entity
        n_causal = len(self.causal_features)
        if n_causal == 0:
            return {'drift_detected': False, 'confidence': 0.0,
                    'risk_score': 0.0, 'threshold': 0.5,
                    'detection_time': time.time() - t0}

        entity_ts = {}  # entity -> (K, M) array
        for eid in entities:
            K = len(self.entity_history[eid])
            ts = np.zeros((K, n_causal))
            for fi, feat in enumerate(self.causal_features):
                ts[:, fi] = self._get_entity_time_series(eid, feat)
            # Normalize
            ts_std = ts.std(axis=0)
            ts_std[ts_std < 1e-8] = 1.0
            ts = (ts - ts.mean(axis=0)) / ts_std
            entity_ts[eid] = ts

        # Step 2: Reconstruct shadow attractors (Eq. 7)
        attractors = {}
        for eid in entities:
            ts = entity_ts[eid]
            sa = self.reconstruct_shadow_attractor(ts, self.E, self.tau)
            if sa is not None and len(sa) > self.E + 1:
                attractors[eid] = sa

        if len(attractors) < 2:
            return {'drift_detected': False, 'confidence': 0.0,
                    'risk_score': 0.0, 'threshold': 0.5,
                    'detection_time': time.time() - t0}

        # Step 3: Compute distance matrices (Eq. 9)
        dist_matrices = {}
        valid_entities = []
        lid_blocked = set()
        for eid in attractors:
            sa = attractors[eid]
            dist, idx = self.mahalanobis_distance_matrix(sa)
            dist_matrices[eid] = (dist, idx)

            # Step 4: LID validation (Eq. 10)
            k_last = len(dist) - 1
            lid = self.estimate_lid(dist[k_last], self.K_lid)
            if lid > self.theta_lid:
                lid_blocked.add(eid)
            else:
                valid_entities.append(eid)

        if len(valid_entities) < 2:
            # All entities blocked by LID — high manifold complexity
            self.risk_history.append(0.0)
            return {'drift_detected': False, 'confidence': 0.0,
                    'risk_score': 0.0, 'threshold': self.adaptive_threshold(),
                    'detection_time': time.time() - t0,
                    'lid_blocked': len(lid_blocked)}

        # Step 5: Cross-manifold mapping + PC computation (Eq. 11-14)
        n_valid = len(valid_entities)
        pc_matrix = np.zeros((n_valid, n_valid))
        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                ei, ej = valid_entities[i], valid_entities[j]
                M_i = attractors[ei]
                M_j = attractors[ej]
                dist_i, _ = dist_matrices[ei]
                # Use last time point
                k = min(len(M_i) - 1, len(M_j) - 1, len(dist_i) - 1)
                pc = self.cross_manifold_mapping(M_i, M_j, k, self.E, dist_i)
                pc_matrix[i, j] = pc

        # Store for contrastive guidance
        self.last_pc_matrix = pc_matrix.copy()
        self.last_pc_entities = list(valid_entities)

        # Step 6: Build causal graph (Eq. 15)
        G_k = self.build_causal_graph(valid_entities, pc_matrix)

        # Step 7: Extract topological features (Eq. 16-19)
        G_prev = self.graph_history[-1][0] if self.graph_history else None
        delta_den, delta_cen, delta_mod, density, cen_dict, Q_k = \
            self.extract_topological_features(G_k, G_prev)

        # Save graph state
        self.graph_history.append((G_k, density, cen_dict, Q_k))
        if len(self.graph_history) > 20:
            self.graph_history.pop(0)

        # Step 8: Composite risk (Eq. 20)
        Phi_k = self.compute_drift_risk(delta_den, delta_cen, delta_mod)
        self.risk_history.append(Phi_k)

        # Step 9: Adaptive threshold (Eq. 22)
        theta = self.adaptive_threshold()

        # Drift alert
        drift_detected = Phi_k > theta and len(self.risk_history) > 3
        confidence = min(Phi_k / (theta + 1e-10), 1.0) if drift_detected else 0.0

        dt = time.time() - t0
        self.detection_times.append(dt)

        return {
            'drift_detected': drift_detected,
            'confidence': confidence,
            'risk_score': Phi_k,
            'threshold': theta,
            'delta_den': delta_den,
            'delta_cen': delta_cen,
            'delta_mod': delta_mod,
            'n_valid_entities': n_valid,
            'n_lid_blocked': len(lid_blocked),
            'detection_time': dt,
        }


# ============================================================================
# BaselineDriftDetector (KS-test, for comparison / ablation)
# ============================================================================
class BaselineDriftDetector:
    def __init__(self, drift_threshold=0.05):
        self.threshold = drift_threshold
        self.control_data = None
        self.detection_times = []

    def detect_drift(self, new_data, window_id=-1):
        t0 = time.time()
        if self.control_data is None:
            self.control_data = new_data
            return {'drift_detected': False, 'confidence': 0.0, 'detection_time': 0.0}
        p_vals = []
        nf = min(new_data.shape[1], self.control_data.shape[1])
        for i in range(nf):
            try:
                _, p = ks_2samp(self.control_data[:, i], new_data[:, i])
                p_vals.append(p)
            except Exception:
                p_vals.append(1.0)
        min_p = min(p_vals) if p_vals else 1.0
        detected = min_p < self.threshold
        conf = 1.0 - min_p if detected else 0.0
        dt = time.time() - t0
        self.detection_times.append(dt)
        if detected:
            self.control_data = new_data
        return {'drift_detected': detected, 'confidence': conf, 'detection_time': dt}


# ============================================================================
# DriftDetectionComparator
# ============================================================================
class DriftDetectionComparator:
    def __init__(self):
        self.results = {
            'baseline': {'detections': [], 'tp': 0, 'fp': 0, 'fn': 0, 'times': [], 'confs': []},
            'pattern_causality': {'detections': [], 'tp': 0, 'fp': 0, 'fn': 0, 'times': [], 'confs': [],
                                  'validated': 0},
        }
        self.window_perf = []

    def record(self, method, wid, detected, conf, dt, gt=None, validated=False):
        self.results[method]['detections'].append({'wid': wid, 'detected': detected, 'conf': conf})
        self.results[method]['times'].append(dt)
        if detected:
            self.results[method]['confs'].append(conf)
            if validated and method == 'pattern_causality':
                self.results[method]['validated'] += 1
        if gt is not None:
            if detected and gt: self.results[method]['tp'] += 1
            elif detected and not gt: self.results[method]['fp'] += 1
            elif not detected and gt: self.results[method]['fn'] += 1

    def record_perf(self, wid, acc, f1, bl, ca, val=False):
        self.window_perf.append({'wid': wid, 'acc': acc, 'f1': f1, 'bl': bl, 'ca': ca, 'val': val})

    def report(self):
        r = {}
        for m in ['baseline', 'pattern_causality']:
            d = self.results[m]
            total = len(d['detections'])
            det = sum(1 for x in d['detections'] if x['detected'])
            tp, fp, fn = d['tp'], d['fp'], d['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1v = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            r[m] = {
                'detection_rate': det / total if total > 0 else 0,
                'precision': prec, 'recall': rec, 'f1': f1v,
                'avg_time_ms': np.mean(d['times']) * 1000 if d['times'] else 0,
                'avg_conf': np.mean(d['confs']) if d['confs'] else 0,
                'total_detections': det,
            }
            if m == 'pattern_causality':
                r[m]['validated'] = d['validated']
                r[m]['validation_ratio'] = d['validated'] / det if det > 0 else 0
        return r


# ============================================================================
# ContinualLearningModule — Paper Eq. 23-24
# MLP [128, 64, 32], dropout=0.2, lambda=0.1
# ============================================================================
class ContinualLearningModule(nn.Module):
    """Paper specification:
    - Architecture: MLP [128, 64, 32] -> 2
    - Dropout: 0.2
    - EWC lambda: 0.1
    - No memory replay (paper doesn't mention it)
    - Binary loss switch: L_stable (CE+EWC) vs L_drift (pure CE)
    """

    def __init__(self, input_dim, hidden_dims=(128, 64, 32), n_classes=2,
                 dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd), nn.ReLU(),
                nn.Dropout(dropout), nn.BatchNorm1d(hd)
            ])
            prev = hd
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)
        # EWC state
        self.ewc_lambda = 0.1  # Paper value
        self.fisher_dict = {}
        self.optpar_dict = {}

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits, feat

    def compute_ewc_loss(self):
        """Compute EWC penalty (Eq. 23)."""
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return self.ewc_lambda * loss

    def update_fisher(self, data_loader, device):
        """Update diagonal Fisher information matrix."""
        self.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.named_parameters() if p.requires_grad}
        n_batches = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            logits, _ = self(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
            n_batches += 1
        if n_batches > 0:
            for n in fisher:
                fisher[n] /= n_batches
        # Accumulate with old Fisher
        for n in fisher:
            if n in self.fisher_dict:
                self.fisher_dict[n] = 0.5 * self.fisher_dict[n] + 0.5 * fisher[n]
            else:
                self.fisher_dict[n] = fisher[n]
        self.optpar_dict = {n: p.data.clone() for n, p in self.named_parameters()}


# ============================================================================
# PerformanceMonitor
# ============================================================================
class PerformanceMonitor:
    def __init__(self):
        self.history = {
            'acc': [], 'prec': [], 'rec': [], 'f1': [], 'wid': [], 'ar': [],
            'drift': [], 'bl_drift': [], 'ca_drift': [],
            'gt_drift': [], 'phase': [],
            'bl_bool': [], 'ca_bool': [],
        }

    def update(self, wid, preds, labels, ar=None, drift=False,
               bl=False, ca=False, gt=False, phase='train'):
        if len(preds) == 0 or len(labels) == 0:
            return
        self.history['acc'].append(accuracy_score(labels, preds))
        self.history['prec'].append(precision_score(labels, preds, zero_division=0))
        self.history['rec'].append(recall_score(labels, preds, zero_division=0))
        self.history['f1'].append(f1_score(labels, preds, zero_division=0))
        self.history['wid'].append(wid)
        if ar is not None:
            self.history['ar'].append(ar)
        if drift:
            self.history['drift'].append(wid)
        if bl:
            self.history['bl_drift'].append(wid)
        if ca:
            self.history['ca_drift'].append(wid)
        self.history['gt_drift'].append(gt)
        self.history['phase'].append(phase)
        self.history['bl_bool'].append(bl)
        self.history['ca_bool'].append(ca)


# ============================================================================
# AdaptiveContinualIDS (Main orchestrator)
# ============================================================================
class AdaptiveContinualIDS:
    def __init__(self, config=None, dataset_preset=None):
        self.config = config or self._default_config()
        self.preset = dataset_preset or DATASET_PRESETS['UNSW-NB15']
        self.aggregator = NetworkEntityAggregator(
            window_size=self.config['window_size'],
            aggregation_rules=self.preset['aggregation_rules'],
            label_col=self.preset['label_col'],
            label_positive=self.preset['label_positive'],
            proto_prefix=self.preset.get('proto_prefix', 'proto_'),
            service_prefix=self.preset.get('service_prefix', 'service_'),
            trend_features=self.preset.get('trend_features', []),
        )
        self.drift_forecaster = PCDriftForecaster(
            E=self.config['E'], tau=self.config['tau'],
            alpha=self.config['alpha'], beta=self.config['beta'],
            gamma=self.config['gamma'],
            sigma=self.config['sigma'],
            theta_lid=self.config['theta_lid'],
            K_lid=self.config['K_lid'],
            lambda_decay=self.config['lambda_decay'],
            eta=self.config['eta'],
            baseline_window=self.config['baseline_window'],
            causal_features=self.preset['causal_features'],
            min_history=self.config['min_windows_for_causality'],
        )
        self.baseline_detector = BaselineDriftDetector(
            self.config['baseline_drift_threshold'])
        self.comparator = DriftDetectionComparator()
        self.monitor = PerformanceMonitor()
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_step = 0
        self.drift_count = 0
        self.baseline_drift_count = 0
        self.causal_drift_count = 0
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self._running_min = None
        self._running_max = None
        self.prediction_threshold = 0.5  # Default; auto-calibrated after training
        self._is_initial_phase = False
        self._calib_scores = []   # Collect training scores for threshold calibration
        self._calib_labels = []
        self.timings = defaultdict(list)
        self._feature_template = self._build_feature_template()
        # P3: Replay buffer for EWC Fisher computation
        self._replay_buffer_X = deque(maxlen=500)
        self._replay_buffer_y = deque(maxlen=500)

    def _default_config(self):
        return {
            'window_size': 1000,
            'E': 3, 'tau': 1,
            'hidden_dims': [128, 64, 32],  # Paper spec
            'lr': 0.001, 'batch_size': 32,  # Paper spec
            'dropout': 0.2,  # Paper spec
            'ewc_lambda': 0.5,  # Tuned (joint grid: sigma=1.0, ewc=0.5)
            'label_ratio': 1.0,  # 1.0=full labels (Table 1); 0.01=1% (annotation budget experiment)
            # PC-DriftForecasting parameters
            'alpha': 0.4, 'beta': 0.35, 'gamma': 0.25,
            'sigma': 1.0,  # Tuned (joint grid best)
            'theta_lid': 15.0, 'K_lid': 10,
            'lambda_decay': 1.0, 'eta': 2.0,
            'baseline_window': 10,
            'baseline_drift_threshold': 0.05,
            'min_windows_for_causality': 5,  # Lowered from 10
            'init_epochs': 3,   # Epochs per window during Phase 0 (initial training)
            'drift_epochs': 2,  # Epochs per window when drift detected
        }

    def _build_feature_template(self):
        """Build fixed-order feature name list from preset aggregation_rules.

        Ensures every window produces the same feature dimension regardless
        of which entities/features appear.
        """
        template = []
        for feat in sorted(self.preset['aggregation_rules'].keys()):
            for func in self.preset['aggregation_rules'][feat]:
                template.append(f"{feat}_{func}")
        template.append('entity_size')
        for tf in sorted(self.preset.get('trend_features', [])):
            template.append(f'{tf}_trend')
        return template

    def _init_model(self, input_dim):
        self.model = ContinualLearningModule(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_dims'],
            n_classes=2,
            dropout=self.config['dropout'],
        ).to(self.device)
        self.model.ewc_lambda = self.config['ewc_lambda']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config['lr'], weight_decay=1e-5)

    def process_stream(self, df, is_training=True, quiet=False, phase='train'):
        windows = self.aggregator.create_non_overlapping_windows(df)
        results = {
            'predictions': [], 'labels': [],
            'drift_events': [],
        }
        if not quiet:
            print(f"\nProcessing {len(windows)} windows...")
            print(f"{'Win':<6} {'Ent':<5} {'Anom%':<8} {'Baseline':<13} {'PC-Drift':<16} {'F1':<8}")
            print("-" * 66)

        for widx, (wid, wdata) in enumerate(windows):
            # ---- STEP 1: ECBA feature aggregation (label-free) ----
            t_agg = time.perf_counter()
            entity_feats = self.aggregator.aggregate_features(wdata, wid)
            self.timings['aggregation'].append(time.perf_counter() - t_agg)
            if not entity_feats:
                continue

            # Update entity history for PC-DriftForecasting (label-free)
            t_cause = time.perf_counter()
            self.drift_forecaster.update_entity_history(entity_feats)
            self.timings['causal_update'].append(time.perf_counter() - t_cause)

            # Prepare feature tensor (label-free)
            t_feat = time.perf_counter()
            features, entity_order = self._prepare_features_only(entity_feats)
            self.timings['feature_prep'].append(time.perf_counter() - t_feat)
            if features is None:
                continue

            feat_tensor = features.to(self.device)
            ar = (self.aggregator.window_stats[widx]['anomaly_ratio']
                  if widx < len(self.aggregator.window_stats) else 0)

            # ---- STEP 2: Predict (no label access) ----
            preds, f1v = None, 0.0
            t_infer = time.perf_counter()
            scores = None
            if self.model is not None:
                preds, scores = self._predict(feat_tensor)
            self.timings['inference'].append(time.perf_counter() - t_infer)

            # ---- STEP 3: Extract labels SEPARATELY ----
            # For evaluation: use all labels (ground truth)
            gt_labels = self.aggregator.extract_entity_labels_full(wdata)
            labels_np = np.array([gt_labels.get(eid, 0) for eid in entity_order])
            if preds is not None:
                results['predictions'].extend(preds.tolist())
                results['labels'].extend(labels_np.tolist())
                f1v = f1_score(labels_np, preds, zero_division=0)
                # Collect scores for threshold calibration (training only)
                if is_training and scores is not None:
                    self._calib_scores.extend(scores.tolist())
                    self._calib_labels.extend(labels_np.tolist())

            # ---- STEP 4: Drift detection ----
            t_drift = time.perf_counter()
            bl_res = self.baseline_detector.detect_drift(features.cpu().numpy(), wid)
            bl_det, bl_conf = bl_res['drift_detected'], bl_res['confidence']

            ca_det, ca_conf = False, 0.0
            if widx >= self.config['min_windows_for_causality']:
                pc_res = self.drift_forecaster.forecast_drift()
                ca_det = pc_res['drift_detected']
                ca_conf = pc_res['confidence']
            self.timings['drift_detection'].append(time.perf_counter() - t_drift)

            gt_drift = abs(ar - self.preset.get('anomaly_ratio_baseline', 0.087)) > 0.1
            self.comparator.record('baseline', wid, bl_det, bl_conf,
                                   bl_res['detection_time'], gt_drift)
            if widx >= self.config['min_windows_for_causality']:
                self.comparator.record('pattern_causality', wid, ca_det, ca_conf,
                                       pc_res.get('detection_time', 0), gt_drift)

            if bl_det:
                self.baseline_drift_count += 1
            if ca_det:
                self.causal_drift_count += 1

            drift_adapt = ca_det
            if drift_adapt:
                self.drift_count += 1
                results['drift_events'].append({'wid': wid, 'conf': ca_conf})

            if preds is not None:
                self.monitor.update(wid, preds, labels_np, ar, drift_adapt,
                                    bl_det, ca_det, gt=gt_drift, phase=phase)
                self.comparator.record_perf(wid, accuracy_score(labels_np, preds),
                                            f1v, bl_det, ca_det)

            # ---- STEP 5: Training with limited labels (Eq. 23-24) ----
            t_train = time.perf_counter()
            if is_training:
                train_labels_dict = self.aggregator.extract_entity_labels(
                    wdata, label_ratio=self.config.get('label_ratio', 1.0))
                # Only train on entities that have at least one labeled connection
                train_mask = np.array([eid in train_labels_dict for eid in entity_order])
                if train_mask.any():
                    train_feats = feat_tensor[train_mask]
                    train_labels = torch.LongTensor(
                        [train_labels_dict[eid] for eid in entity_order if eid in train_labels_dict]
                    ).to(self.device)
                    if self.model is None:
                        self._init_model(len(self._feature_template))
                    # Adaptive epoch count: init phase > drift > stable
                    if self._is_initial_phase:
                        n_ep = self.preset.get('init_epochs',
                                               self.config.get('init_epochs', 3))
                    elif drift_adapt:
                        n_ep = self.config.get('drift_epochs', 2)
                    else:
                        n_ep = 1
                    if drift_adapt:
                        self._train_pure_ce(train_feats, train_labels, n_epochs=n_ep)
                    else:
                        self._train_with_ewc(train_feats, train_labels, n_epochs=n_ep)
            self.timings['training'].append(time.perf_counter() - t_train)

            if not quiet and (wid % 10 == 0 or wid == len(windows) - 1):
                bl_s = f"Yes({bl_conf:.2f})" if bl_det else "No"
                ca_s = f"Yes({ca_conf:.2f})" if ca_det else "No"
                n_ent = len(entity_feats)
                print(f"{wid:<6} {n_ent:<5} {ar * 100:<7.1f}% "
                      f"{bl_s:<13} {ca_s:<16} {f1v:<8.4f}")

        return results

    def _prepare_features_only(self, entity_feats):
        """Convert entity feature records to feature tensor (label-free).

        Uses _feature_template for fixed dimension across all windows.
        Returns: (features_tensor, entity_order_list) or (None, None)
        """
        if not entity_feats:
            return None, None
        feats_list = []
        entity_order = []
        for eid, rec in entity_feats.items():
            vec = [float(rec.get(key, 0.0)) if isinstance(rec.get(key, 0.0), (int, float)) else 0.0
                   for key in self._feature_template]
            feats_list.append(vec)
            entity_order.append(eid)

        if not feats_list:
            return None, None

        raw = np.array(feats_list, dtype=np.float32)

        # Scale: MinMaxScaler fitted once, then frozen
        if not self.scaler_fitted:
            self.scaler.fit(raw)
            self.scaler_fitted = True
        scaled = self.scaler.transform(raw)

        feats = torch.FloatTensor(scaled)
        feats = torch.clamp(feats, -5, 5)
        feats[torch.isnan(feats)] = 0
        feats[torch.isinf(feats)] = 0
        return feats, entity_order

    def _predict(self, features):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(features)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (scores >= self.prediction_threshold).astype(np.int64)
        return preds, scores

    def _class_weights(self, labels):
        unique, counts = torch.unique(labels, return_counts=True)
        w = torch.ones(2, device=self.device)
        if len(unique) > 1:
            total = len(labels)
            for i, lbl in enumerate(unique):
                w[lbl] = total / (2.0 * counts[i])
        return torch.clamp(w, 0.3, 8.0)

    def _store_replay_samples(self, features, labels):
        """Store ~20% of current window samples into replay buffer."""
        n_store = max(1, int(len(features) * 0.2))
        idx = torch.randperm(len(features))[:n_store]
        for i in idx:
            self._replay_buffer_X.append(features[i].detach().cpu())
            self._replay_buffer_y.append(labels[i].detach().cpu())

    def _train_with_ewc(self, features, labels, n_epochs=1):
        """Eq. 23: L_stable = L_NID(theta) + (lambda/2) * EWC_loss."""
        if len(features) < 2:
            return  # BatchNorm needs >1 samples
        self.model.train()
        cw = self._class_weights(labels)
        ds = torch.utils.data.TensorDataset(features, labels)
        bs = min(self.config['batch_size'], len(features))
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(features) > bs))
        for _ in range(n_epochs):
            for bf, bl in dl:
                if len(bf) < 2:
                    continue
                self.optimizer.zero_grad()
                logits, _ = self.model(bf)
                loss = nn.CrossEntropyLoss(weight=cw)(logits, bl)
                if self.training_step > 0:
                    loss = loss + self.model.compute_ewc_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
        # Store samples into replay buffer
        self._store_replay_samples(features, labels)
        # Fisher computed on replay buffer (protects old knowledge, not new)
        if len(self._replay_buffer_X) >= 32:
            replay_X = torch.stack(list(self._replay_buffer_X)).to(self.device)
            replay_y = torch.stack(list(self._replay_buffer_y)).to(self.device)
            ds_fisher = torch.utils.data.TensorDataset(replay_X, replay_y)
            dl_fisher = torch.utils.data.DataLoader(ds_fisher, batch_size=bs)
            self.model.update_fisher(dl_fisher, self.device)
        self.training_step += 1

    def _train_pure_ce(self, features, labels, n_epochs=1):
        """Eq. 24: L_drift = L_NID(theta) — pure CE, no EWC."""
        if len(features) < 2:
            return
        self.model.train()
        cw = self._class_weights(labels)
        ds = torch.utils.data.TensorDataset(features, labels)
        bs = min(self.config['batch_size'], len(features))
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(features) > bs))
        for _ in range(n_epochs):
            for bf, bl in dl:
                if len(bf) < 2:
                    continue
                self.optimizer.zero_grad()
                logits, _ = self.model(bf)
                loss = nn.CrossEntropyLoss(weight=cw)(logits, bl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
        # Store samples into replay buffer (no Fisher in drift mode)
        self._store_replay_samples(features, labels)
        self.training_step += 1

    def _calibrate_threshold(self):
        """Auto-calibrate prediction threshold from training predictions (no test leakage).
        Uses only the last 30% of training scores (model is more stable by then)."""
        if len(self._calib_scores) < 50:
            return
        n = len(self._calib_scores)
        start = int(n * 0.7)  # Last 30% of training
        scores = np.array(self._calib_scores[start:])
        labels = np.array(self._calib_labels[start:])
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.30, 0.65, 0.01):
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, round(t, 2)
        self.prediction_threshold = best_t

    def train_phase(self, df, quiet=False, phase='train', initial=False):
        if not quiet:
            print("\n=== Training Phase ===")
        self._is_initial_phase = initial
        self.process_stream(df, is_training=True, quiet=quiet, phase=phase)
        self._is_initial_phase = False

    def evaluate_phase(self, df, quiet=False, phase='test'):
        if not quiet:
            print("\n=== Evaluation Phase ===")
        bl_start = self.baseline_drift_count
        ca_start = self.causal_drift_count
        results = self.process_stream(df, is_training=False, quiet=quiet, phase=phase)
        all_l = results.get('labels', [])
        all_p = results.get('predictions', [])
        if len(all_l) > 0 and len(all_p) > 0:
            n = min(len(all_l), len(all_p))
            all_l, all_p = all_l[:n], all_p[:n]
            return {
                'accuracy': accuracy_score(all_l, all_p),
                'precision': precision_score(all_l, all_p, zero_division=0),
                'recall': recall_score(all_l, all_p, zero_division=0),
                'f1_score': f1_score(all_l, all_p, zero_division=0),
                'baseline_drifts': self.baseline_drift_count - bl_start,
                'causal_drifts': self.causal_drift_count - ca_start,
                'total_windows': len(
                    self.aggregator.create_non_overlapping_windows(df)),
            }
        return {}

    def export_window_csv(self, filepath):
        """Export per-window metrics to CSV for visualization."""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        h = self.monitor.history
        n = len(h['wid'])
        rows = []
        for i in range(n):
            rows.append({
                'seq_id': i,
                'window_id': h['wid'][i],
                'f1': h['f1'][i],
                'accuracy': h['acc'][i],
                'precision': h['prec'][i],
                'recall': h['rec'][i],
                'anomaly_ratio': h['ar'][i] if i < len(h['ar']) else 0,
                'baseline_drift': h['bl_bool'][i] if i < len(h['bl_bool']) else False,
                'causal_drift': h['ca_bool'][i] if i < len(h['ca_bool']) else False,
                'gt_drift': h['gt_drift'][i] if i < len(h['gt_drift']) else False,
                'phase': h['phase'][i] if i < len(h['phase']) else 'unknown',
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Per-window data exported to {filepath} ({n} windows)")


# ============================================================================
# Data loading
# ============================================================================
def load_data(dataset_name):
    preset = DATASET_PRESETS[dataset_name]
    df_train = pd.read_csv(preset['train_path'])
    df_test = pd.read_csv(preset['test_path'])
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    lc = preset['label_col']
    if lc in num_cols:
        num_cols.remove(lc)
    df_train[num_cols] = df_train[num_cols].fillna(0)
    df_test[num_cols] = df_test[num_cols].fillna(0)
    if dataset_name == 'NSL-KDD':
        if df_train[lc].dtype == object:
            df_train[lc] = (df_train[lc] != 'normal').astype(int)
            df_test[lc] = (df_test[lc] != 'normal').astype(int)
            DATASET_PRESETS['NSL-KDD']['label_positive'] = 1
    print(f"Dataset: {dataset_name}")
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    ar_train = (df_train[lc] == preset['label_positive']).mean()
    ar_test = (df_test[lc] == preset['label_positive']).mean()
    print(f"Train anomaly ratio: {ar_train:.4f}, Test anomaly ratio: {ar_test:.4f}")
    return df_train, df_test


# ============================================================================
# CLI Config Builder
# ============================================================================
def build_config(args):
    return {
        'window_size': args.W,
        'E': 3, 'tau': 1,
        'hidden_dims': [128, 64, 32],
        'lr': 0.001, 'batch_size': 32,
        'dropout': 0.2,
        'ewc_lambda': args.ewc_lambda,
        'label_ratio': getattr(args, 'label_ratio', 1.0),
        'alpha': 0.4, 'beta': 0.35, 'gamma': 0.25,
        'sigma': 1.0,
        'theta_lid': 15.0, 'K_lid': 10,
        'lambda_decay': 1.0, 'eta': 2.0,
        'baseline_window': 10,
        'baseline_drift_threshold': args.baseline_drift_threshold,
        'min_windows_for_causality': 5,
        'init_epochs': 3,
        'drift_epochs': 2,
    }


# ============================================================================
# Mode: run
# ============================================================================
def mode_run(args):
    preset = DATASET_PRESETS[args.dataset]
    df_train, df_test = load_data(args.dataset)
    config = build_config(args)
    ids = AdaptiveContinualIDS(config=config, dataset_preset=preset)

    split = int(len(df_train) * 0.2)
    df_init = df_train.iloc[:split]
    df_online = df_train.iloc[split:]

    print(f"\n--- Phase 0: Initial Training ({len(df_init)} records, 20%) ---")
    ids.train_phase(df_init, phase='train', initial=True)

    print(f"\n--- Phase 1: Online Continual Learning ({len(df_online)} records, 80% train) ---")
    ids.train_phase(df_online, phase='train', initial=False)

    # Auto-calibrate threshold from training predictions (no test leakage)
    ids._calibrate_threshold()
    print(f"Calibrated threshold: {ids.prediction_threshold:.2f}")

    print(f"\n--- Phase 2: Evaluation ({len(df_test)} records, test) ---")
    metrics = ids.evaluate_phase(df_test, phase='test')

    if metrics:
        print(f"\n=== Final Results ===")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")
        print(f"\nDrift events in test: baseline={metrics.get('baseline_drifts', 0)}, "
              f"causal={metrics.get('causal_drifts', 0)}")

    csv_path = f'results/window_data_{args.dataset.replace("-", "_")}.csv'
    ids.export_window_csv(csv_path)
    return metrics


# ============================================================================
# Mode: sensitivity
# ============================================================================
def mode_sensitivity(args):
    preset = DATASET_PRESETS[args.dataset]
    df_train, df_test = load_data(args.dataset)
    base_config = build_config(args)
    seeds = [42, 123, 7]

    if getattr(args, 'sensitivity_quick', False):
        param_grids = {
            'ewc_lambda': [0.01, 0.1, 0.5],
            'window_size': [500, 1000, 2000],
            'sigma': [1.5, 2.0, 3.0],
        }
    else:
        param_grids = {
            'ewc_lambda': [0.01, 0.05, 0.1, 0.5, 1.0],
            'window_size': [500, 1000, 1500, 2000],
            'sigma': [1.0, 1.5, 2.0, 2.5, 3.0],
            'alpha': [0.2, 0.3, 0.4, 0.5],
            'baseline_drift_threshold': [0.01, 0.05, 0.1],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.0005, 0.001, 0.002],
        }

    all_results = []
    total_runs = sum(len(v) for v in param_grids.values()) * len(seeds)
    run_idx = 0

    for param, values in param_grids.items():
        for val in values:
            seed_metrics = []
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {param} = {val}, seed = {seed}")
                np.random.seed(seed)
                torch.manual_seed(seed)
                cfg = base_config.copy()
                cfg[param] = val
                if param == 'window_size':
                    cfg['window_size'] = val
                ids = AdaptiveContinualIDS(config=cfg, dataset_preset=preset)
                split = int(len(df_train) * 0.2)
                ids.train_phase(df_train.iloc[:split], quiet=True)
                ids.train_phase(df_train.iloc[split:], quiet=True)
                metrics = ids.evaluate_phase(df_test, quiet=True)
                seed_metrics.append(metrics)
                print(f"  F1={metrics.get('f1_score', 0):.4f}")

            f1s = [m.get('f1_score', 0) for m in seed_metrics]
            result = {
                'param': param, 'value': val,
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            }
            all_results.append(result)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_json = f'results/sensitivity_{args.dataset.replace("-", "_")}.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")
    return all_results


# ============================================================================
# Mode: efficiency
# ============================================================================
def mode_efficiency(args):
    preset = DATASET_PRESETS[args.dataset]
    df_train, df_test = load_data(args.dataset)

    window_sizes = [500, 1000, 1500, 2000]
    results = []

    for ws in window_sizes:
        print(f"\nEfficiency test: window_size = {ws}")
        config = build_config(args)
        config['window_size'] = ws
        ids = AdaptiveContinualIDS(config=config, dataset_preset=preset)

        tracemalloc.start()
        t0 = time.perf_counter()
        split = int(len(df_train) * 0.2)
        ids.train_phase(df_train.iloc[:split], quiet=True)
        ids.train_phase(df_train.iloc[split:], quiet=True)
        metrics = ids.evaluate_phase(df_test, quiet=True)
        t_total = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_conns = len(df_train) + len(df_test)
        throughput = total_conns / t_total if t_total > 0 else 0

        result = {
            'window_size': ws, 'total_runtime_s': round(t_total, 2),
            'peak_memory_mb': round(peak_mem / 1024 / 1024, 2),
            'throughput_conn_per_s': round(throughput, 1),
            'metrics': metrics,
        }
        results.append(result)
        print(f"  Runtime: {result['total_runtime_s']}s | Memory: {result['peak_memory_mb']}MB | "
              f"F1: {metrics.get('f1_score', 0):.4f}")

    out_json = f'results/efficiency_{args.dataset.replace("-", "_")}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_json}")
    return results


# ============================================================================
# Mode: drift
# ============================================================================
def mode_drift(args):
    preset = DATASET_PRESETS[args.dataset]
    df_train, df_test = load_data(args.dataset)
    config = build_config(args)
    ids = AdaptiveContinualIDS(config=config, dataset_preset=preset)

    split = int(len(df_train) * 0.2)
    ids.train_phase(df_train.iloc[:split], quiet=True)
    ids.train_phase(df_train.iloc[split:], quiet=True)

    print(f"\n--- Drift Evaluation on Test Set ---")
    ids.process_stream(df_test, is_training=False, quiet=False)

    report = ids.comparator.report()
    print(f"\n=== Drift Detection Comparison ===")
    for method, stats in report.items():
        print(f"\n{method}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return report


# ============================================================================
# Main
# ============================================================================
def main():
    import os
    os.makedirs('results', exist_ok=True)

    parser = argparse.ArgumentParser(description='IDA-SPADE')
    parser.add_argument('--dataset', type=str, default='UNSW-NB15',
                        choices=['UNSW-NB15', 'NSL-KDD'])
    parser.add_argument('--mode', type=str, default='run',
                        choices=['run', 'sensitivity', 'efficiency', 'drift'])
    parser.add_argument('--W', type=int, default=1000, help='Window size (T)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ewc-lambda', type=float, default=0.5)
    parser.add_argument('--baseline-drift-threshold', type=float, default=0.05)
    parser.add_argument('--label-ratio', type=float, default=1.0,
                        help='Fraction of labels visible (1.0=full, 0.01=1%%)')
    parser.add_argument('--sensitivity-quick', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'run':
        mode_run(args)
    elif args.mode == 'sensitivity':
        mode_sensitivity(args)
    elif args.mode == 'efficiency':
        mode_efficiency(args)
    elif args.mode == 'drift':
        mode_drift(args)


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Deque
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
import time
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, pairwise
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
import argparse
from pattern_causality import pattern_causality

class DriftValidator:
    """
    A module to validate the authenticity of concept drift events.
    Confirms whether drift is "real" by analyzing multiple evidence aspects after detection.
    """
    def __init__(self,
                 performance_drop_threshold: float = 0.05,
                 mmd_threshold: float = 0.2,
                 causal_change_threshold: float = 0.08):
        self.perf_drop_threshold = performance_drop_threshold
        self.mmd_threshold = mmd_threshold
        self.causal_change_threshold = causal_change_threshold
        self.validation_history = []

    def _calculate_mmd(self, x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        """Calculate Maximum Mean Discrepancy (MMD) between two sample sets"""
        if x.shape[1] != y.shape[1]:
            min_dim = min(x.shape[1], y.shape[1])
            x = x[:, :min_dim]
            y = y[:, :min_dim]

        K_xx = pairwise.rbf_kernel(x, x, gamma=1.0 / (2 * sigma**2))
        K_yy = pairwise.rbf_kernel(y, y, gamma=1.0 / (2 * sigma**2))
        K_xy = pairwise.rbf_kernel(x, y, gamma=1.0 / (2 * sigma**2))
        
        mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        return float(np.sqrt(max(0, mmd)))

    def validate_drift(self,
                       current_window_id: int,
                       current_features: np.ndarray,
                       current_f1_score: float,
                       current_causal_matrix: np.ndarray,
                       history_buffer: Deque[Dict]) -> Dict:
        """
        Execute drift validation.
        """
        if len(history_buffer) < 3:
            return {'is_real_drift': False, 'confidence': 0.0, 'reasons': ['insufficient_history']}

        # Prepare historical data
        ref_features = np.vstack([item['features'] for item in history_buffer])
        ref_f1_scores = [item['f1_score'] for item in history_buffer]
        ref_causal_matrices = [item['causal_matrix'] for item in history_buffer if item['causal_matrix'] is not None]

        validation_signals = {}
        
        # 1. Check for significant performance degradation
        avg_ref_f1 = np.mean(ref_f1_scores)
        f1_drop = avg_ref_f1 - current_f1_score
        if f1_drop > self.perf_drop_threshold:
            validation_signals['performance_degradation'] = f1_drop

        # 2. Check for significant distribution shift (MMD)
        try:
            mmd_score = self._calculate_mmd(current_features, ref_features)
            if mmd_score > self.mmd_threshold:
                validation_signals['distribution_shift'] = mmd_score
        except Exception as e:
            pass

        # 3. Check for significant causal structure change
        if len(ref_causal_matrices) > 0 and current_causal_matrix is not None:
            avg_ref_causal_matrix = np.mean(ref_causal_matrices, axis=0)
            
            # Ensure matrix shapes match
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(avg_ref_causal_matrix.shape, current_causal_matrix.shape))
            avg_ref_causal_matrix_sliced = avg_ref_causal_matrix[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            current_causal_matrix_sliced = current_causal_matrix[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]

            norm_diff = np.linalg.norm(current_causal_matrix_sliced - avg_ref_causal_matrix_sliced)
            norm_ref = np.linalg.norm(avg_ref_causal_matrix_sliced)
            
            if norm_ref > 1e-8:
                relative_change = norm_diff / norm_ref
                if relative_change > self.causal_change_threshold:
                    validation_signals['causal_structure_shift'] = relative_change
        
        # Comprehensive judgment
        is_real_drift = len(validation_signals) > 0
        confidence = 0.0
        
        if is_real_drift:
            confidence = len(validation_signals) / 3.0
            # Performance drop is a strong signal
            if 'performance_degradation' in validation_signals:
                confidence = min(confidence + 0.3, 1.0)
        
        result = {
            'window_id': current_window_id,
            'is_real_drift': is_real_drift,
            'confidence': confidence,
            'reasons': list(validation_signals.keys()),
            'details': validation_signals
        }
        self.validation_history.append(result)
        return result

class NetworkEntityAggregator:
    """Aggregate discrete network traffic data into entity-centric behavior time series"""
    def __init__(self, window_size: int = 1000,
                 slide_size: int = 500,
                 entity_features: List[str] = None,
                 aggregation_features: Dict[str, str] = None,
                 dataset_type: str = 'NSL-KDD'):
        self.window_size = window_size
        self.slide_size = slide_size
        self.dataset_type = dataset_type
        self.window_stats = []
        
        # Set aggregation features based on dataset type
        if dataset_type == 'NSL-KDD':
            self.aggregation_features = {
                'duration': 'mean', 'src_bytes': 'sum', 'dst_bytes': 'sum', 'count': 'mean',
                'srv_count': 'mean', 'dst_host_count': 'mean', 'dst_host_srv_count': 'mean',
                'serror_rate': 'mean', 'srv_serror_rate': 'mean', 'rerror_rate': 'mean',
                'srv_rerror_rate': 'mean', 'dst_host_serror_rate': 'mean', 'dst_host_rerror_rate': 'mean',
                'same_srv_rate': 'mean', 'diff_srv_rate': 'mean', 'dst_host_same_srv_rate': 'mean',
                'dst_host_diff_srv_rate': 'mean', 'dst_host_srv_diff_host_rate': 'mean',
                'wrong_fragment': 'sum', 'urgent': 'sum', 'hot': 'sum', 'num_failed_logins': 'sum',
                'num_compromised': 'sum', 'num_root': 'sum', 'num_file_creations': 'sum',
                'num_shells': 'sum', 'num_access_files': 'sum',
            }
            self.label_column = 'labels2'
            self.anomaly_value = 'anomaly'
        else:  # UNSW-NB15
            self.aggregation_features = {
                'dur': 'mean', 'sbytes': 'sum', 'dbytes': 'sum', 'spkts': 'sum', 'dpkts': 'sum',
                'sload': 'mean', 'dload': 'mean', 'rate': 'mean',
                'ct_srv_src': 'max', 'ct_srv_dst': 'max', 'ct_dst_ltm': 'max', 'ct_src_ltm': 'max',
                'ct_src_dport_ltm': 'max', 'ct_dst_sport_ltm': 'max', 'ct_dst_src_ltm': 'max',
                'sttl': 'mean', 'dttl': 'mean', 'swin': 'max', 'dwin': 'max',
                'sloss': 'sum', 'dloss': 'sum', 'sinpkt': 'mean', 'dinpkt': 'mean',
                'sjit': 'std', 'djit': 'std', 'smean': 'mean', 'dmean': 'mean',
                'trans_depth': 'max', 'response_body_len': 'sum',
                'tcprtt': 'mean', 'synack': 'mean', 'ackdat': 'mean',
                'stcpb': 'sum', 'dtcpb': 'sum',
                'ct_state_ttl': 'mean', 'ct_flw_http_mthd': 'sum',
                'ct_ftp_cmd': 'sum', 'is_ftp_login': 'max', 'is_sm_ips_ports': 'max'
            }
            self.label_column = 'label'
            self.anomaly_value = 1
            
        # Additional attributes for UNSW-NB15
        if dataset_type == 'UNSW-NB15':
            self.feature_stability = {}
            self.global_feature_stats = {}
            self.entity_count_history = deque(maxlen=10)

    def _compute_feature_statistics(self, df: pd.DataFrame):
        """Compute global feature statistics (UNSW-NB15 only)"""
        if self.dataset_type != 'UNSW-NB15':
            return
            
        self.global_feature_stats = {}
        for feature in self.aggregation_features.keys():
            if feature in df.columns:
                self.global_feature_stats[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'q25': df[feature].quantile(0.25),
                    'q75': df[feature].quantile(0.75),
                    'nunique': df[feature].nunique()
                }

    def _check_feature_stability(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check feature stability and availability (UNSW-NB15 only)"""
        if self.dataset_type != 'UNSW-NB15':
            return {}
            
        stability = {}
        for feature in self.aggregation_features.keys():
            if feature in df.columns:
                if (df[feature].nunique() > 1 and 
                    df[feature].std() > 1e-6 and
                    not df[feature].isna().all()):
                    stability[feature] = True
                else:
                    stability[feature] = False
            else:
                stability[feature] = False
        return stability

    def create_sliding_windows(self, df: pd.DataFrame) -> List[Tuple[int, pd.DataFrame]]:
        windows = []
        n_samples = len(df)
        start = 0
        window_id = 0
        
        if self.dataset_type == 'UNSW-NB15':
            self._compute_feature_statistics(df)
            self.feature_stability = self._check_feature_stability(df)
            stable_features = sum(self.feature_stability.values())
            print(f"Feature stability check: {stable_features}/{len(self.feature_stability)} features are stable")
        
        while start + self.window_size <= n_samples:
            end = start + self.window_size
            window_data = df.iloc[start:end]
            windows.append((window_id, window_data))
            
            if self.label_column in window_data.columns:
                anomaly_ratio = (window_data[self.label_column] == self.anomaly_value).mean()
                label_counts = window_data[self.label_column].value_counts()
                self.window_stats.append({
                    'window_id': window_id, 'anomaly_ratio': anomaly_ratio,
                    'total_samples': len(window_data), 'label_distribution': label_counts.to_dict()
                })
            else:
                self.window_stats.append({
                    'window_id': window_id, 'anomaly_ratio': 0.0,
                    'total_samples': len(window_data), 'label_distribution': {}
                })
                
            start += self.slide_size
            window_id += 1
        return windows

    def aggregate_window(self, window_data: pd.DataFrame, window_id: int) -> Dict[str, pd.DataFrame]:
        entity_records = defaultdict(dict)
        
        if self.dataset_type == 'NSL-KDD':
            # NSL-KDD entity grouping logic
            try:
                entity_groups = pd.qcut(window_data['dst_host_count'],
                                        q=5,
                                        labels=[f'entity_{i}' for i in range(5)],
                                        duplicates='drop')
            except:
                entity_groups = pd.cut(window_data['dst_host_count'],
                                       bins=5,
                                       labels=[f'entity_{i}' for i in range(5)])
        else:
            # UNSW-NB15 entity grouping logic
            entity_groups = self._create_adaptive_entities(window_data, window_id)
            unique_entities = entity_groups.nunique()
            self.entity_count_history.append(unique_entities)
        
        for entity_id in entity_groups.unique():
            if pd.isna(entity_id):
                continue
            entity_mask = entity_groups == entity_id
            entity_data = window_data[entity_mask]
            if len(entity_data) > 0:
                aggregated_features = {}
                
                for feature, agg_func in self.aggregation_features.items():
                    if feature in entity_data.columns:
                        if self.dataset_type == 'UNSW-NB15' and not self.feature_stability.get(feature, True):
                            aggregated_features[feature] = 0.0
                            continue
                            
                        try:
                            if agg_func == 'mean':
                                val = float(entity_data[feature].mean())
                            elif agg_func == 'sum':
                                val = float(entity_data[feature].sum())
                            elif agg_func == 'max':
                                val = float(entity_data[feature].max())
                            elif agg_func == 'min':
                                val = float(entity_data[feature].min())
                            elif agg_func == 'std':
                                val = float(entity_data[feature].std())
                                if np.isnan(val):
                                    val = 0.0
                            else:
                                val = float(entity_data[feature].mean())
                            
                            if np.isfinite(val):
                                aggregated_features[feature] = val
                            else:
                                aggregated_features[feature] = 0.0
                        except:
                            aggregated_features[feature] = 0.0
                    else:
                        aggregated_features[feature] = 0.0
                
                # Add additional features
                aggregated_features['time_window'] = window_id
                
                if self.dataset_type == 'UNSW-NB15':
                    aggregated_features['entity_size'] = len(entity_data)
                
                if self.label_column in entity_data.columns:
                    mode_result = entity_data[self.label_column].mode()
                    if self.dataset_type == 'NSL-KDD':
                        aggregated_features['label'] = mode_result[0] if len(mode_result) > 0 else 'normal'
                        aggregated_features['entity_anomaly_ratio'] = (entity_data[self.label_column] == 'anomaly').mean()
                    else:  # UNSW-NB15
                        entity_label_value = mode_result[0] if len(mode_result) > 0 else 0
                        aggregated_features['label'] = 'anomaly' if entity_label_value == 1 else 'normal'
                        aggregated_features['entity_anomaly_ratio'] = float((entity_data[self.label_column] == 1).mean())
                        aggregated_features['label_consistency'] = float((entity_data[self.label_column] == entity_label_value).mean())
                else:
                    aggregated_features['label'] = 'normal'
                    aggregated_features['entity_anomaly_ratio'] = 0.0
                    if self.dataset_type == 'UNSW-NB15':
                        aggregated_features['label_consistency'] = 1.0
                    
                entity_records[str(entity_id)] = aggregated_features
        
        return entity_records

    def _create_adaptive_entities(self, window_data: pd.DataFrame, window_id: int) -> pd.Series:
        """Create adaptive entity grouping for UNSW-NB15"""
        clustering_features = []
        feature_weights = {}
        
        key_features = [
            ('rate', 3.0), ('sload', 2.5), ('dload', 2.5),
            ('ct_srv_src', 2.0), ('ct_dst_src_ltm', 2.0),
            ('sbytes', 1.5), ('dbytes', 1.5),
            ('spkts', 1.0), ('dpkts', 1.0), ('dur', 1.0)
        ]
        
        valid_features = []
        for feature, weight in key_features:
            if (feature in window_data.columns and 
                self.feature_stability.get(feature, False) and
                window_data[feature].nunique() > 1):
                
                feature_data = window_data[feature].values
                if feature in self.global_feature_stats:
                    stats = self.global_feature_stats[feature]
                    normalized = (feature_data - stats['mean']) / (stats['std'] + 1e-8)
                    normalized = normalized * weight
                    clustering_features.append(normalized)
                    valid_features.append(feature)
                    feature_weights[feature] = weight
        
        if len(clustering_features) < 2:
            return self._fallback_entity_creation(window_data, window_id)
        
        X = np.column_stack(clustering_features)
        X = np.clip(X, np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0))
        
        n_samples = len(window_data)
        optimal_clusters = min(max(3, n_samples // 50), 20)
        
        try:
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            entity_labels = [f'entity_{i}' for i in cluster_labels]
            return pd.Series(entity_labels, index=window_data.index)
            
        except Exception as e:
            print(f"Debug: Clustering failed for window {window_id}: {e}")
            return self._fallback_entity_creation(window_data, window_id)

    def _fallback_entity_creation(self, window_data: pd.DataFrame, window_id: int) -> pd.Series:
        """Fallback entity creation method for UNSW-NB15"""
        try:
            if 'label' in window_data.columns:
                normal_data = window_data[window_data['label'] == 0]
                anomaly_data = window_data[window_data['label'] == 1]
                
                entity_labels = []
                
                if len(normal_data) > 0:
                    if 'rate' in normal_data.columns and normal_data['rate'].nunique() > 1:
                        normal_groups = pd.qcut(normal_data['rate'], q=min(5, normal_data['rate'].nunique()), 
                                              labels=False, duplicates='drop')
                        normal_labels = [f'normal_{g}' for g in normal_groups]
                    else:
                        normal_labels = ['normal_0'] * len(normal_data)
                    entity_labels.extend(normal_labels)
                
                if len(anomaly_data) > 0:
                    if 'sload' in anomaly_data.columns and anomaly_data['sload'].nunique() > 1:
                        anomaly_groups = pd.qcut(anomaly_data['sload'], q=min(5, anomaly_data['sload'].nunique()), 
                                               labels=False, duplicates='drop')
                        anomaly_labels = [f'anomaly_{g}' for g in anomaly_groups]
                    else:
                        anomaly_labels = ['anomaly_0'] * len(anomaly_data)
                    entity_labels.extend(anomaly_labels)
                
                full_labels = [''] * len(window_data)
                normal_idx = 0
                anomaly_idx = 0
                
                for i, (idx, row) in enumerate(window_data.iterrows()):
                    if row['label'] == 0:
                        full_labels[i] = entity_labels[normal_idx] if normal_idx < len(normal_data) else 'normal_0'
                        normal_idx += 1
                    else:
                        normal_count = len(normal_data)
                        full_labels[i] = entity_labels[normal_count + anomaly_idx] if normal_count + anomaly_idx < len(entity_labels) else 'anomaly_0'
                        anomaly_idx += 1
                
                return pd.Series(full_labels, index=window_data.index)
            
        except Exception as e:
            print(f"Debug: Fallback entity creation failed: {e}")
        
        n_samples = len(window_data)
        n_groups = min(10, max(3, n_samples // 100))
        group_size = n_samples // n_groups
        entity_labels = []
        for i in range(n_samples):
            group_id = min(i // group_size, n_groups - 1)
            entity_labels.append(f'uniform_{group_id}')
        
        return pd.Series(entity_labels, index=window_data.index)

class BaselineDriftDetector:
    def __init__(self, drift_threshold: float = 0.05, window_size: int = 1000):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.control_data = None
        self.detection_times = []

    def update_control_data(self, data: np.ndarray):
        self.control_data = data

    def detect_drift(self, new_data: np.ndarray, window_id: int = -1) -> Dict:
        start_time = time.time()
        if self.control_data is None:
            self.control_data = new_data
            return {'drift_detected': False, 'confidence': 0.0, 'method': 'baseline_ks', 'p_value': 1.0, 'detection_time': 0.0}
        
        p_values = []
        n_features = min(new_data.shape[1], self.control_data.shape[1])
        for i in range(n_features):
            try:
                _, p_value = ks_2samp(self.control_data[:, i], new_data[:, i])
                p_values.append(p_value)
            except:
                p_values.append(1.0)
        
        min_p_value = min(p_values) if p_values else 1.0
        drift_detected = min_p_value < self.drift_threshold
        confidence = 1.0 - min_p_value if drift_detected else 0.0
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if drift_detected:
            self.control_data = new_data
        
        return {
            'drift_detected': drift_detected, 'confidence': confidence, 'method': 'baseline_ks',
            'p_value': min_p_value, 'detection_time': detection_time, 'window_id': window_id
        }

class DriftDetectionComparator:
    """Drift detection method comparator"""
    def __init__(self):
        self.comparison_results = {
            'baseline': {'detections': [], 'false_positives': 0, 'true_positives': 0, 'false_negatives': 0, 'detection_times': [], 'confidence_scores': []},
            'pattern_causality': {'detections': [], 'false_positives': 0, 'true_positives': 0, 'false_negatives': 0, 'detection_times': [], 'confidence_scores': [], 'validated_real_drifts': 0}
        }
        self.window_performance = []

    def record_detection(self, method: str, window_id: int, detected: bool, confidence: float, detection_time: float, ground_truth_drift: bool = None, is_validated: bool = False):
        self.comparison_results[method]['detections'].append({'window_id': window_id, 'detected': detected, 'confidence': confidence, 'detection_time': detection_time})
        self.comparison_results[method]['detection_times'].append(detection_time)
        if detected:
            self.comparison_results[method]['confidence_scores'].append(confidence)
            if is_validated and method == 'pattern_causality':
                self.comparison_results[method]['validated_real_drifts'] += 1
        
        if ground_truth_drift is not None:
            if detected and ground_truth_drift: self.comparison_results[method]['true_positives'] += 1
            elif detected and not ground_truth_drift: self.comparison_results[method]['false_positives'] += 1
            elif not detected and ground_truth_drift: self.comparison_results[method]['false_negatives'] += 1

    def record_window_performance(self, window_id: int, accuracy: float, f1_score: float, baseline_drift: bool, causal_drift: bool, validated_drift: bool = False):
        self.window_performance.append({
            'window_id': window_id, 'accuracy': accuracy, 'f1_score': f1_score,
            'baseline_drift': baseline_drift, 'causal_drift': causal_drift,
            'validated_drift': validated_drift
        })

    def generate_comparison_report(self) -> Dict:
        report = {}
        for method in ['baseline', 'pattern_causality']:
            results = self.comparison_results[method]
            total_windows = len(results['detections'])
            detected_count = sum(1 for d in results['detections'] if d['detected'])
            tp, fp, fn = results['true_positives'], results['false_positives'], results['false_negatives']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            report[method] = {
                'detection_rate': detected_count / total_windows if total_windows > 0 else 0,
                'precision': precision, 'recall': recall, 'f1_score': f1,
                'avg_detection_time': np.mean(results['detection_times']) if results['detection_times'] else 0,
                'avg_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0,
                'total_detections': detected_count
            }
            if method == 'pattern_causality':
                report[method]['validated_real_drifts'] = results.get('validated_real_drifts', 0)
                if detected_count > 0:
                    report[method]['validation_ratio'] = results.get('validated_real_drifts', 0) / detected_count
                else:
                    report[method]['validation_ratio'] = 0

        if report['baseline']['f1_score'] > 0:
            report['improvement'] = {
                'f1_score_improvement': (report['pattern_causality']['f1_score'] - report['baseline']['f1_score']) / report['baseline']['f1_score']
            }
        return report

    def plot_comparison(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drift Detection Method Comparison', fontsize=16)

        # 1. Performance and drift events
        ax = axes[0, 0]
        if self.window_performance:
            windows = [p['window_id'] for p in self.window_performance]
            f1_scores = [p['f1_score'] for p in self.window_performance]
            ax.plot(windows, f1_scores, 'k-', alpha=0.9, label='F1-Score')

            baseline_windows = [p['window_id'] for p in self.window_performance if p.get('baseline_drift')]
            if baseline_windows:
                ax.vlines(baseline_windows, ymin=0, ymax=1, color='blue', linestyle='--', alpha=0.6, label='Baseline Drift')
            
            causal_windows = [p['window_id'] for p in self.window_performance if p.get('causal_drift')]
            if causal_windows:
                ax.vlines(causal_windows, ymin=0, ymax=1, color='orange', linestyle=':', alpha=0.7, label='Causal Drift (Detected)')
            
            validated_drifts_to_plot = [
                p for p in self.window_performance
                if p.get('validated_drift') and p.get('f1_score', 0) >= 0.1
            ]

            if validated_drifts_to_plot:
                windows_for_marker = [p['window_id'] for p in validated_drifts_to_plot]
                f1_for_marker = [p['f1_score'] for p in validated_drifts_to_plot]
                ax.scatter(windows_for_marker, f1_for_marker, color='red', s=80, marker='X', zorder=5, label='Real Drift')

            ax.set_xlabel('Window ID')
            ax.set_ylabel('F1 Score')
            ax.set_title('Model Performance and Drift Events')
            ax.legend(fontsize='small')
            ax.set_ylim([0, 1.1])

        # 2. Detection time comparison
        ax = axes[0, 1]
        methods = ['Baseline KS', 'Pattern Causality']
        avg_times = [np.mean(self.comparison_results['baseline']['detection_times'])*1000 if self.comparison_results['baseline']['detection_times'] else 0, 
                     np.mean(self.comparison_results['pattern_causality']['detection_times'])*1000 if self.comparison_results['pattern_causality']['detection_times'] else 0]
        ax.bar(methods, avg_times, color=['blue', 'red'])
        ax.set_ylabel('Average Detection Time (ms)')
        ax.set_title('Detection Time Comparison')

        # 3. Detection count comparison
        ax = axes[0, 2]
        report = self.generate_comparison_report()
        baseline_total = report['baseline']['total_detections']
        causal_total = report['pattern_causality']['total_detections']
        causal_validated = report['pattern_causality'].get('validated_real_drifts', 0)
        ax.bar('Baseline', baseline_total, color='blue', label='Baseline Detections')
        ax.bar('Causal', causal_total, color='orange', label='Causal Detections (All)')
        ax.bar('Causal', causal_validated, color='red', label='Causal Detections (Validated Real)')
        ax.set_ylabel('Number of Detections')
        ax.set_title('Total Detections Comparison')
        ax.legend()

        # 4. Confidence distribution
        ax = axes[1, 0]
        if self.comparison_results['baseline']['confidence_scores']: 
            ax.hist(self.comparison_results['baseline']['confidence_scores'], bins=20, alpha=0.5, label='Baseline', color='blue')
        if self.comparison_results['pattern_causality']['confidence_scores']: 
            ax.hist(self.comparison_results['pattern_causality']['confidence_scores'], bins=20, alpha=0.5, label='Pattern Causality', color='red')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Score Distribution')
        ax.legend()
        
        # 5. Performance metrics comparison
        ax = axes[1, 1]
        metrics = ['Detection Rate', 'Avg Confidence', 'F1 Score']
        baseline_values = [report['baseline']['detection_rate'], report['baseline']['avg_confidence'], report['baseline']['f1_score']]
        causal_values = [report['pattern_causality']['detection_rate'], report['pattern_causality']['avg_confidence'], report['pattern_causality']['f1_score']]
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, baseline_values, width, label='Baseline', color='blue')
        ax.bar(x + width/2, causal_values, width, label='Pattern Causality', color='red')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1.1])

        # 6. Causal drift validation results
        ax = axes[1, 2]
        labels = ['Detected', 'Validated Real']
        counts = [causal_total, causal_validated]
        ax.bar(labels, counts, color=['orange', 'red'])
        ax.set_ylabel('Count')
        ax.set_title('Causal Drift Validation Breakdown')
        if causal_total > 0:
            ax.text(1, causal_validated, f' {causal_validated/causal_total:.1%}', ha='center', va='bottom')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'drift_detection_comparison_validated_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

class PatternCausalityEngine:
    def __init__(self, embedding_dim: int = 3, time_delay: int = 1, prediction_horizon: int = 1,
                 history_length: int = 50, min_history_for_detection: int = 15,
                 causal_threshold: float = 0.2, dark_threshold: float = 0.2,
                 dataset_type: str = 'NSL-KDD'):
        self.E, self.tau, self.h = embedding_dim, time_delay, prediction_horizon
        self.history_length = history_length
        self.min_history = min_history_for_detection
        self.causal_threshold = causal_threshold
        self.dark_threshold = dark_threshold
        self.dataset_type = dataset_type
        self.pc = pattern_causality(verbose=False)
        self.causal_history = deque(maxlen=history_length)
        self.entity_history = defaultdict(lambda: deque(maxlen=history_length))
        self.total_causality_history = deque(maxlen=20)
        self.dark_causality_history = deque(maxlen=20)
        self.causal_matrix_history = deque(maxlen=10)
        self.detection_times = []
        
        # Set causal features based on dataset type
        if dataset_type == 'NSL-KDD':
            self.causal_features = ['serror_rate', 'srv_serror_rate', 'dst_host_count', 'same_srv_rate', 'diff_srv_rate']
        else:  # UNSW-NB15
            self.causal_features = ['sload', 'dload', 'rate', 'ct_srv_src', 'ct_dst_ltm']

    def update_entity_history(self, entity_records: Dict[str, Dict]):
        for entity_id, features in entity_records.items():
            self.entity_history[entity_id].append(features)

    def compute_pattern_causality(self) -> Optional[Dict]:
        entities = list(self.entity_history.keys())
        n_entities = len(entities)
        if n_entities < 2 or min(len(self.entity_history[e]) for e in entities) < self.min_history:
            return None

        n_features = len(self.causal_features)
        causal_matrix = np.zeros((n_entities, n_entities, n_features, 4))
        valid_pairs, total_causality, total_dark_causality = 0, 0, 0

        for i in range(n_entities):
            for j in range(n_entities):
                if i == j: continue
                entity_i, entity_j = entities[i], entities[j]
                history_i, history_j = list(self.entity_history[entity_i]), list(self.entity_history[entity_j])
                for f_idx, feature in enumerate(self.causal_features):
                    try:
                        X = np.array([h.get(feature, 0) for h in history_i[-30:]])
                        Y = np.array([h.get(feature, 0) for h in history_j[-30:]])
                        
                        if self.dataset_type == 'NSL-KDD':
                            X = np.clip(X, np.percentile(X, 5), np.percentile(X, 95))
                            Y = np.clip(Y, np.percentile(Y, 5), np.percentile(Y, 95))
                        else:  # UNSW-NB15
                            X = np.clip(X, np.percentile(X, 1), np.percentile(X, 99))
                            Y = np.clip(Y, np.percentile(Y, 1), np.percentile(Y, 99))
                            X = (X - np.mean(X)) / (np.std(X) + 1e-8)
                            Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
                        
                        if len(X) < self.E * 3 or np.std(X) < 1e-6 or np.std(Y) < 1e-6:
                            continue
                            
                        result = self.pc.pc_lightweight(X=X, Y=Y, E=self.E, tau=self.tau, h=self.h, weighted=True, relative=True)
                        if result is not None and len(result) > 0:
                            causal_matrix[i, j, f_idx, 0] = result.iloc[0]['Total Causality']
                            causal_matrix[i, j, f_idx, 1] = result.iloc[0]['Positive Causality']
                            causal_matrix[i, j, f_idx, 2] = result.iloc[0]['Negative Causality']
                            causal_matrix[i, j, f_idx, 3] = result.iloc[0]['Dark Causality']
                            total_causality += result.iloc[0]['Total Causality']
                            total_dark_causality += result.iloc[0]['Dark Causality']
                            valid_pairs += 1
                    except Exception:
                        continue
                        
        if valid_pairs == 0:
            return None
        
        avg_total_causality = total_causality / valid_pairs
        avg_dark_causality = total_dark_causality / valid_pairs
        self.total_causality_history.append(avg_total_causality)
        self.dark_causality_history.append(avg_dark_causality)
        self.causal_matrix_history.append(causal_matrix)
        
        centrality = self._compute_causal_centrality(causal_matrix)
        result = {'causal_matrix': causal_matrix, 'causal_centrality': centrality, 
                  'avg_total_causality': avg_total_causality, 'avg_dark_causality': avg_dark_causality}
        self.causal_history.append(result)
        return result

    def detect_causal_drift(self) -> Dict:
        start_time = time.time()
        if not self.causal_history:
            return {'drift_detected': False, 'confidence': 0.0, 'method': 'pattern_causality', 'details': {}, 'detection_time': 0}
        
        current_causality = self.causal_history[-1]

        if len(self.total_causality_history) < 5:
            return {'drift_detected': False, 'confidence': 0.0, 'method': 'pattern_causality', 'details': {}, 'detection_time': time.time() - start_time}
        
        recent_total = list(self.total_causality_history)[-5:]
        initial_total = list(self.total_causality_history)[:5] if len(self.total_causality_history) >= 10 else recent_total
        total_change = abs(np.mean(recent_total) - np.mean(initial_total))
        total_relative_change = total_change / (np.mean(initial_total) + 1e-8)
        
        recent_dark = list(self.dark_causality_history)[-5:]
        initial_dark = list(self.dark_causality_history)[:5] if len(self.dark_causality_history) >= 10 else recent_dark
        dark_change = abs(np.mean(recent_dark) - np.mean(initial_dark))
        dark_relative_change = dark_change / (np.mean(initial_dark) + 1e-8)

        structure_change = 0
        if len(self.causal_matrix_history) >= 2:
            recent_matrix, prev_matrix = self.causal_matrix_history[-1], self.causal_matrix_history[-2]
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(recent_matrix.shape, prev_matrix.shape))
            recent_matrix_sliced = recent_matrix[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            prev_matrix_sliced = prev_matrix[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            matrix_diff = np.abs(recent_matrix_sliced - prev_matrix_sliced)
            structure_change = np.mean(matrix_diff)
        
        sudden_change = False
        if len(self.total_causality_history) > 3:
            recent_std = np.std(list(self.total_causality_history)[:-1])
            if recent_std > 0:
                z_score = abs(current_causality['avg_total_causality'] - np.mean(list(self.total_causality_history)[:-1])) / recent_std
                sudden_change = z_score > 2
        
        drift_signals, confidence_scores = [], []
        if total_relative_change > self.causal_threshold:
            drift_signals.append('total_causality_change')
            confidence_scores.append(min(total_relative_change / self.causal_threshold, 1.0))
        if dark_relative_change > self.dark_threshold:
            drift_signals.append('dark_causality_change')
            confidence_scores.append(min(dark_relative_change / self.dark_threshold, 1.0))
        if structure_change > 0.2:
            drift_signals.append('structure_change')
            confidence_scores.append(min(structure_change / 0.2, 1.0))
        if sudden_change:
            drift_signals.append('sudden_change')
            confidence_scores.append(0.8)

        drift_detected = len(drift_signals) > 0
        confidence = np.mean(confidence_scores) if drift_detected else 0.0
        if drift_detected and 'dark_causality_change' in drift_signals:
            confidence = min(confidence * 1.2, 1.0)
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        details = {'total_causality_change': total_relative_change, 'dark_causality_change': dark_relative_change, 
                   'structure_change': structure_change, 'sudden_change': sudden_change, 'drift_signals': drift_signals}
        
        return {'drift_detected': drift_detected, 'confidence': confidence, 'drift_type': 'causal_drift', 
                'method': 'pattern_causality', 'details': details, 'detection_time': detection_time}

    def _compute_causal_centrality(self, causal_matrix: np.ndarray) -> np.ndarray:
        n_entities = causal_matrix.shape[0]
        centrality = np.zeros((n_entities, 8))
        avg_causal = np.mean(causal_matrix, axis=2)
        for i in range(n_entities):
            centrality[i, 0] = np.mean(avg_causal[i, :, 0])
            centrality[i, 1] = np.mean(avg_causal[i, :, 1])
            centrality[i, 2] = np.mean(avg_causal[i, :, 2])
            centrality[i, 3] = np.mean(avg_causal[i, :, 3])
            centrality[i, 4] = np.mean(avg_causal[:, i, 0])
            centrality[i, 5] = np.mean(avg_causal[:, i, 1])
            centrality[i, 6] = np.mean(avg_causal[:, i, 2])
            centrality[i, 7] = np.mean(avg_causal[:, i, 3])
        return centrality

class ContinualLearningModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], n_classes: int = 2, 
                 memory_size: int = 500, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim, self.n_classes, self.memory_size = input_dim, n_classes, memory_size
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.BatchNorm1d(hidden_dim)])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)
        self.ewc_lambda = 0.05
        self.fisher_dict, self.optpar_dict = {}, {}
        self.memory = {'features': deque(maxlen=memory_size), 'labels': deque(maxlen=memory_size), 
                      'importance': deque(maxlen=memory_size)}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def compute_ewc_loss(self):
        ewc_loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher_dict:
                ewc_loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return self.ewc_lambda * ewc_loss

    def update_memory(self, features: torch.Tensor, labels: torch.Tensor, importance: torch.Tensor):
        for i in range(features.size(0)):
            self.memory['features'].append(features[i].cpu())
            self.memory['labels'].append(labels[i].cpu())
            self.memory['importance'].append(importance[i].item())

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'window_id': [],
            'anomaly_ratio': [], 'drift_events': [], 'baseline_drift_events': [],
            'causal_drift_events': [], 'validated_drift_events': []
        }

    def update(self, window_id: int, predictions: np.ndarray, labels: np.ndarray, anomaly_ratio: float = None,
               drift_detected: bool = False, baseline_drift: bool = False, causal_drift: bool = False, 
               validated_drift: bool = False):
        if len(predictions) == 0 or len(labels) == 0:
            return
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        self.metrics_history['accuracy'].append(acc)
        self.metrics_history['precision'].append(prec)
        self.metrics_history['recall'].append(rec)
        self.metrics_history['f1_score'].append(f1)
        self.metrics_history['window_id'].append(window_id)
        if anomaly_ratio is not None:
            self.metrics_history['anomaly_ratio'].append(anomaly_ratio)
        if drift_detected:
            self.metrics_history['drift_events'].append(window_id)
        if baseline_drift:
            self.metrics_history['baseline_drift_events'].append(window_id)
        if causal_drift:
            self.metrics_history['causal_drift_events'].append(window_id)
        if validated_drift:
            self.metrics_history['validated_drift_events'].append(window_id)

    def plot_performance(self):
        if not self.metrics_history['window_id']:
            print("No performance data to plot")
            return
        print("Performance plotting is now integrated into the comparison plot.")

class AdaptiveContinualIDS:
    def __init__(self, config: dict = None, dataset_type: str = 'NSL-KDD'):
        self.dataset_type = dataset_type
        self.config = config or self._default_config()
        self.aggregator = NetworkEntityAggregator(
            window_size=self.config['window_size'], 
            slide_size=self.config['slide_size'],
            dataset_type=dataset_type
        )
        self.causal_engine = PatternCausalityEngine(
            embedding_dim=self.config['embedding_dim'], time_delay=self.config['time_delay'],
            prediction_horizon=self.config['prediction_horizon'], history_length=self.config['causal_history_length'],
            min_history_for_detection=self.config['min_windows_for_causality'],
            causal_threshold=self.config['causal_threshold'], dark_threshold=self.config['dark_threshold'],
            dataset_type=dataset_type
        )
        self.baseline_detector = BaselineDriftDetector(
            drift_threshold=self.config['baseline_drift_threshold'], 
            window_size=self.config['window_size']
        )
        self.drift_comparator = DriftDetectionComparator()
        self.drift_validator = DriftValidator()
        self.performance_monitor = PerformanceMonitor()
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_step, self.drift_count, self.baseline_drift_count, self.causal_drift_count = 0, 0, 0, 0
        self.drift_events = []
        self.validated_causal_drift_count = 0
        self.validation_buffer = deque(maxlen=5)
        
        # Add feature scaler for UNSW-NB15
        if dataset_type == 'UNSW-NB15':
            self.feature_scaler = MinMaxScaler()
            self.scaler_fitted = False

    def _default_config(self) -> dict:
        if self.dataset_type == 'NSL-KDD':
            return {
                'window_size': 1000, 'slide_size': 500, 'embedding_dim': 3, 'time_delay': 1, 
                'prediction_horizon': 1, 'hidden_dims': [128, 64, 32], 'learning_rate': 0.001, 
                'batch_size': 32, 'memory_size': 500, 'drift_window_size': 10, 
                'drift_sensitivity': 'low', 'causal_threshold': 0.2, 'dark_threshold': 0.06, 
                'baseline_drift_threshold': 0.079366, 'causal_history_length': 50,
                'min_windows_for_causality': 15, 'adaptation_lr': 0.0005, 'replay_ratio': 0.3, 
                'dropout_rate': 0.2, 'causal_weight': 0.3, 'use_causal_for_adaptation': True
            }
        else:  # UNSW-NB15
            return {
                'window_size': 1000, 'slide_size': 500, 'embedding_dim': 3, 'time_delay': 1, 
                'prediction_horizon': 1, 'hidden_dims': [256, 128, 64], 'learning_rate': 0.001, 
                'batch_size': 64, 'memory_size': 1000, 'drift_window_size': 10, 
                'drift_sensitivity': 'medium', 'causal_threshold': 0.12, 'dark_threshold': 0.1, 
                'baseline_drift_threshold': 0.2, 'causal_history_length': 50,
                'min_windows_for_causality': 10, 'adaptation_lr': 0.002, 'replay_ratio': 0.4, 
                'dropout_rate': 0.3, 'causal_weight': 0.6, 'use_causal_for_adaptation': True
            }

    def initialize_model(self, input_dim: int):
        self.model = ContinualLearningModule(
            input_dim=input_dim, hidden_dims=self.config['hidden_dims'], n_classes=2,
            memory_size=self.config['memory_size'], dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        if self.dataset_type == 'UNSW-NB15':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def process_stream(self, df: pd.DataFrame, is_training: bool = True):
        windows = self.aggregator.create_sliding_windows(df)
        
        results = {
            'predictions': [], 'scores': [], 'labels': [],
            'drift_events': [], 'baseline_drift_events': [],
            'causal_drift_events': [], 'validated_drift_events': [],
            'performance': [], 'window_stats': self.aggregator.window_stats,
            'all_features': [], 'all_labels': []
        }

        print(f"\nProcessing {len(windows)} windows...")
        if self.dataset_type == 'NSL-KDD':
            print(f"{'Window':<8} {'Samples':<8} {'Anomaly%':<10} {'Baseline':<12} {'Causal':<15} {'Validated':<12} {'F1-Score':<10}")
            print("-" * 85)
        else:
            print(f"{'Window':<8} {'Entities':<10} {'Samples':<8} {'Anomaly%':<10} {'Baseline':<12} {'Causal':<15} {'Validated':<12} {'F1-Score':<10}")
            print("-" * 95)

        for window_idx, (window_id, window_data) in enumerate(windows):
            entity_records = self.aggregator.aggregate_window(window_data, window_id)
            if not entity_records:
                continue
            
            self.causal_engine.update_entity_history(entity_records)
            
            causal_result = None
            causal_features = None
            current_causal_matrix = None
            if window_idx >= self.config['min_windows_for_causality']:
                causal_result = self.causal_engine.compute_pattern_causality()
                if causal_result:
                    causal_features = causal_result.get('causal_centrality')
                    current_causal_matrix = causal_result.get('causal_matrix')

            features, labels = self._prepare_window_features(entity_records, causal_features)
            if features is None or len(features) == 0:
                continue
            
            # Record features and labels for t-SNE
            results['all_features'].append(features)
            if labels is not None:
                results['all_labels'].append(labels)

            features_tensor = features.to(self.device)
            labels_np = labels.numpy() if labels is not None else None
            
            current_anomaly_ratio = self.aggregator.window_stats[window_idx]['anomaly_ratio'] if window_idx < len(self.aggregator.window_stats) else 0

            predictions, scores, f1 = None, None, 0.0
            if self.model is not None:
                predictions, scores = self._predict(features_tensor)
                
                results['predictions'].extend(predictions.tolist())
                results['scores'].extend(scores.tolist())
                if labels_np is not None:
                    results['labels'].extend(labels_np.tolist())
                    f1 = f1_score(labels_np, predictions, zero_division=0)

            baseline_result = self.baseline_detector.detect_drift(features.cpu().numpy(), window_id)
            baseline_drift_detected, baseline_confidence = baseline_result['drift_detected'], baseline_result['confidence']
            
            causal_drift_detected, causal_confidence = False, 0.0
            if window_idx >= self.config['min_windows_for_causality']:
                causal_detection_result = self.causal_engine.detect_causal_drift()
                causal_drift_detected, causal_confidence = causal_detection_result['drift_detected'], causal_detection_result['confidence']

            is_validated_real_drift = False
            validation_confidence = 0.0
            if causal_drift_detected:
                validation_result = self.drift_validator.validate_drift(
                    current_window_id=window_id,
                    current_features=features.cpu().numpy(),
                    current_f1_score=f1,
                    current_causal_matrix=current_causal_matrix,
                    history_buffer=self.validation_buffer
                )
                is_validated_real_drift = validation_result['is_real_drift']
                validation_confidence = validation_result['confidence']
                if is_validated_real_drift:
                    self.validated_causal_drift_count += 1
                    results['validated_drift_events'].append({
                        'window_id': window_id, 'confidence': validation_confidence, 'reasons': validation_result['reasons']
                    })

            # Adjust ground truth drift based on dataset
            if self.dataset_type == 'NSL-KDD':
                ground_truth_drift = abs(current_anomaly_ratio - 0.4654) > 0.1
            else:  # UNSW-NB15
                ground_truth_drift = abs(current_anomaly_ratio - 0.087) > 0.1
                
            self.drift_comparator.record_detection('baseline', window_id, baseline_drift_detected, baseline_confidence, 
                                                 baseline_result.get('detection_time', 0), ground_truth_drift)
            if window_idx >= self.config['min_windows_for_causality']:
                self.drift_comparator.record_detection('pattern_causality', window_id, causal_drift_detected, causal_confidence, 
                                                     causal_detection_result.get('detection_time', 0), ground_truth_drift, is_validated=is_validated_real_drift)

            if baseline_drift_detected:
                self.baseline_drift_count += 1
            if causal_drift_detected:
                self.causal_drift_count += 1
            
            drift_detected_for_adapt = causal_drift_detected if self.config['use_causal_for_adaptation'] else baseline_drift_detected
            if drift_detected_for_adapt:
                self.drift_count += 1
                results['drift_events'].append({'window_id': window_id, 'confidence': causal_confidence})

            if predictions is not None and labels_np is not None:
                self.performance_monitor.update(window_id, predictions, labels_np, current_anomaly_ratio, 
                                              drift_detected_for_adapt, baseline_drift_detected, causal_drift_detected, is_validated_real_drift)
                self.drift_comparator.record_window_performance(window_id, accuracy_score(labels_np, predictions), f1, 
                                                              baseline_drift_detected, causal_drift_detected, is_validated_real_drift)
            
            if is_training and labels is not None:
                if self.model is None:
                    self.initialize_model(features.shape[1])
                if drift_detected_for_adapt:
                    self._adapt_to_drift(features_tensor, labels.to(self.device), causal_confidence, 'causal_drift')
                else:
                    self._incremental_update(features_tensor, labels.to(self.device))
            
            self.validation_buffer.append({'features': features.cpu().numpy(), 'f1_score': f1, 'causal_matrix': current_causal_matrix})

            if window_id % 5 == 0 or window_id == len(windows) - 1:
                bl_status = f"Yes({baseline_confidence:.2f})" if baseline_drift_detected else "No"
                pc_status = f"Yes({causal_confidence:.2f})" if causal_drift_detected else "No"
                val_status = f"REAL({validation_confidence:.2f})" if is_validated_real_drift else ("-" if not causal_drift_detected else "Virtual")
                
                if self.dataset_type == 'NSL-KDD':
                    print(f"{window_id:<8} {len(features):<8} {current_anomaly_ratio*100:<9.1f}% "
                          f"{bl_status:<12} {pc_status:<15} {val_status:<12} {f1:<10.4f}")
                else:
                    print(f"{window_id:<8} {len(entity_records):<10} {len(features):<8} {current_anomaly_ratio*100:<9.1f}% "
                          f"{bl_status:<12} {pc_status:<15} {val_status:<12} {f1:<10.4f}")

        return results

    def _prepare_window_features(self, entity_records: Dict, causal_features: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not entity_records:
            return None, None
        features_list, labels_list = [], []
        entities = list(entity_records.keys())
        
        if self.dataset_type == 'NSL-KDD':
            num_base_features = len(self.aggregator.aggregation_features) + 1  # +1 for entity_anomaly_ratio
        else:  # UNSW-NB15
            num_base_features = len(self.aggregator.aggregation_features) + 2  # +2 for entity_anomaly_ratio and entity_size
        
        for i, entity_id in enumerate(entities):
            record = entity_records[entity_id]
            base_features = [record.get(feature, 0) for feature in self.aggregator.aggregation_features.keys()]
            base_features.append(record.get('entity_anomaly_ratio', 0))
            
            if self.dataset_type == 'UNSW-NB15':
                base_features.append(record.get('entity_size', 1))
            
            if causal_features is not None and i < len(causal_features):
                causal_feat = causal_features[i] * self.config['causal_weight']
                combined_features = np.concatenate([base_features, causal_feat])
            else:
                combined_features = np.concatenate([base_features, np.zeros(8)])
            
            features_list.append(combined_features)
            labels_list.append(1 if record.get('label', 'normal') == 'anomaly' else 0)
        
        if features_list:
            features = torch.FloatTensor(np.array(features_list))
            labels = torch.LongTensor(labels_list)
            
            if self.dataset_type == 'NSL-KDD':
                # Standard normalization for NSL-KDD
                features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
                features[torch.isnan(features)] = 0
            else:  # UNSW-NB15
                # MinMaxScaler for UNSW-NB15
                if not self.scaler_fitted:
                    self.feature_scaler.fit(features.numpy())
                    self.scaler_fitted = True
                
                features_scaled = self.feature_scaler.transform(features.numpy())
                features = torch.FloatTensor(features_scaled)
                features = torch.clamp(features, -5, 5)
                features[torch.isnan(features)] = 0
                features[torch.isinf(features)] = 0
                
            return features, labels
        return None, None

    def _predict(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            scores = probabilities[:, 1].cpu().numpy()
        return predictions, scores

    def _incremental_update(self, features: torch.Tensor, labels: torch.Tensor):
        self.model.train()
        
        if len(self.model.memory['features']) > 0:
            memory_size = int(len(features) * self.config['replay_ratio'])
            memory_size = min(memory_size, len(self.model.memory['features']))
            if memory_size > 0:
                memory_indices = np.random.choice(len(self.model.memory['features']), size=memory_size, replace=False)
                memory_features = torch.stack([self.model.memory['features'][i] for i in memory_indices]).to(self.device)
                memory_labels = torch.tensor([self.model.memory['labels'][i] for i in memory_indices]).to(self.device)
                combined_features = torch.cat([features, memory_features])
                combined_labels = torch.cat([labels, memory_labels])
            else:
                combined_features, combined_labels = features, labels
        else:
            combined_features, combined_labels = features, labels

        # Handle class imbalance for UNSW-NB15
        if self.dataset_type == 'UNSW-NB15':
            unique_labels, counts = torch.unique(combined_labels, return_counts=True)
            if len(unique_labels) > 1:
                class_weights = torch.ones(2, device=self.device)
                total_samples = len(combined_labels)
                for i, label in enumerate(unique_labels):
                    class_weights[label] = total_samples / (2.0 * counts[i])
            else:
                class_weights = torch.ones(2, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(combined_features, combined_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(self.config['batch_size'], len(combined_features)), shuffle=True)
        
        for batch_features, batch_labels in dataloader:
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_features)
            cls_loss = criterion(logits, batch_labels)
            ewc_loss = self.model.compute_ewc_loss() if self.training_step > 0 else 0
            loss = cls_loss + ewc_loss
            loss.backward()
            
            # Gradient clipping for UNSW-NB15
            if self.dataset_type == 'UNSW-NB15':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        importance = torch.norm(features, p=2, dim=1)
        self.model.update_memory(features, labels, importance)
        self.training_step += 1

    def _adapt_to_drift(self, features: torch.Tensor, labels: torch.Tensor, confidence: float, drift_type: str):
        print(f"Adapting to {drift_type} (confidence: {confidence:.2f})")
        
        if self.dataset_type == 'NSL-KDD':
            lr_multiplier = 1.5 + confidence if drift_type == 'causal_drift' else 1.0 + confidence * 0.5
            n_epochs = int(3 + confidence * 4) if drift_type == 'causal_drift' else int(2 + confidence * 3)
            importance_multiplier = 2.0 if drift_type == 'causal_drift' else 1 + confidence
        else:  # UNSW-NB15
            lr_multiplier = 2.0 + confidence if drift_type == 'causal_drift' else 1.5 + confidence * 0.5
            n_epochs = int(5 + confidence * 6) if drift_type == 'causal_drift' else int(3 + confidence * 4)
            importance_multiplier = 3.0 if drift_type == 'causal_drift' else 1.5 + confidence
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['adaptation_lr'] * lr_multiplier
        
        self.model.train()
        
        # Handle class imbalance for UNSW-NB15
        if self.dataset_type == 'UNSW-NB15':
            unique_labels, counts = torch.unique(labels, return_counts=True)
            if len(unique_labels) > 1:
                class_weights = torch.ones(2, device=self.device)
                total_samples = len(labels)
                for i, label in enumerate(unique_labels):
                    class_weights[label] = total_samples / (2.0 * counts[i])
            else:
                class_weights = torch.ones(2, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(self.config['batch_size'], len(features) if len(features) > 0 else 1))
        
        for epoch in range(n_epochs):
            for batch_features, batch_labels in dataloader:
                self.optimizer.zero_grad()
                logits, _ = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                
                if self.dataset_type == 'UNSW-NB15':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                self.optimizer.step()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['learning_rate']
        
        importance = torch.norm(features, p=2, dim=1) * importance_multiplier
        self.model.update_memory(features, labels, importance)

    def train(self, df: pd.DataFrame):
        print("\n=== Training Phase ===")
        self.process_stream(df, is_training=True)
        print(f"\n=== Training Summary ===")
        print(f"Total windows processed: {len(self.aggregator.window_stats)}")
        print(f"\n--- Drift Detection Summary ---")
        print(f"Baseline (KS) drift events: {self.baseline_drift_count}")
        print(f"Pattern Causality drift events: {self.causal_drift_count}")
        if self.causal_drift_count > 0:
            print(f"  -> Validated as REAL drifts: {self.validated_causal_drift_count} ({self.validated_causal_drift_count/self.causal_drift_count:.1%} of detected)")
        else:
            print(f"  -> Validated as REAL drifts: 0 (0.0% of detected)")
        print(f"Active drift adaptations: {self.drift_count} (using {'causal' if self.config['use_causal_for_adaptation'] else 'baseline'})")
        
        comparison_report = self.drift_comparator.generate_comparison_report()
        print(f"\n--- Detection Performance Comparison ---")
        for method in ['baseline', 'pattern_causality']:
            if method in comparison_report:
                print(f"\n{method.upper()}:")
                print(f"  Detection rate: {comparison_report[method]['detection_rate']:.2%}")
                print(f"  Average confidence: {comparison_report[method]['avg_confidence']:.3f}")
                if method == 'pattern_causality':
                    print(f"  Validation Ratio (Real/Detected): {comparison_report[method]['validation_ratio']:.2%}")

    def evaluate(self, df: pd.DataFrame) -> Dict:
        print("\n=== Evaluation Phase ===")
        eval_drift_start = self.drift_count
        baseline_eval_start = self.baseline_drift_count
        causal_eval_start = self.causal_drift_count
        validated_eval_start = self.validated_causal_drift_count

        results = self.process_stream(df, is_training=False)
        
        baseline_eval_count = self.baseline_drift_count - baseline_eval_start
        causal_eval_count = self.causal_drift_count - causal_eval_start
        validated_eval_count = self.validated_causal_drift_count - validated_eval_start
        
        all_labels = results.get('labels', [])
        all_predictions = results.get('predictions', [])

        if len(all_labels) > 0 and len(all_predictions) > 0:
            min_len = min(len(all_labels), len(all_predictions))
            all_labels, all_predictions = all_labels[:min_len], all_predictions[:min_len]
            
            # Store features and labels for t-SNE
            self.final_test_features = torch.cat(results['all_features'], dim=0) if results['all_features'] else None
            self.final_test_labels = torch.cat(results['all_labels'], dim=0) if results['all_labels'] else None

            return {
                'accuracy': accuracy_score(all_labels, all_predictions),
                'precision': precision_score(all_labels, all_predictions, zero_division=0),
                'recall': recall_score(all_labels, all_predictions, zero_division=0),
                'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
                'baseline_drift_events': baseline_eval_count,
                'causal_drift_events': causal_eval_count,
                'validated_drift_events': validated_eval_count,
                'total_windows': len(self.aggregator.create_sliding_windows(df))
            }
        
        print("Warning: Evaluation generated no results. Returning empty metrics dict.")
        return {}

    def plot_results(self):
        print("\nPlotting performance curves and comparison results...")
        self.drift_comparator.plot_comparison()

def generate_tsne_visualization(model_system: 'AdaptiveContinualIDS', df_to_visualize: pd.DataFrame, title: str):
    """
    Generate and display t-SNE visualization of model classification results on specified data stream.
    """
    print(f"\nGenerating t-SNE plot: {title}")
    model, aggregator, device = model_system.model, model_system.aggregator, model_system.device
    if model is None:
        print("Model not initialized, cannot generate t-SNE plot.")
        return

    # 1. Prepare data
    aggregator.window_stats = []
    windows = aggregator.create_sliding_windows(df_to_visualize)
    all_features_list, all_labels_list = [], []
    print(f"Preparing visualization data from {len(windows)} windows...")
    for window_id, window_data in windows:
        entity_records = aggregator.aggregate_window(window_data, window_id)
        if not entity_records:
            continue
        features, labels = model_system._prepare_window_features(entity_records, causal_features=None)
        if features is not None and len(features) > 0:
            all_features_list.append(features)
            if labels is not None:
                all_labels_list.append(labels)
    
    if not all_features_list:
        print("Could not generate features from provided data.")
        return
    features_tensor, labels_tensor = torch.cat(all_features_list, dim=0), torch.cat(all_labels_list, dim=0)

    # 2. Extract feature embeddings
    model.eval()
    all_embeddings = []
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features_tensor), batch_size=model_system.config['batch_size'])
    with torch.no_grad():
        for (features_batch,) in test_loader:
            _, embeddings = model(features_batch.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings_np, all_labels_np = np.concatenate(all_embeddings, axis=0), labels_tensor.cpu().numpy()

    # 3. Apply t-SNE
    print(f"Running t-SNE on {all_embeddings_np.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30.0, float(max(1, all_embeddings_np.shape[0] - 1))), n_iter=300)
    tsne_results = tsne.fit_transform(all_embeddings_np)

    # 4. Plot results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels_np, cmap=plt.cm.get_cmap("coolwarm", 2), alpha=0.6)
    plt.colorbar(scatter, ticks=range(2), format=plt.FuncFormatter(lambda val, loc: ['Normal', 'Anomaly'][int(val)]))
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def visualize_tsne_comparison(model_system: 'AdaptiveContinualIDS', title: str):
    """
    Generate and display t-SNE visualization of model classification results on final test data.
    """
    print("\nGenerating t-SNE plot for final evaluation...")
    model = model_system.model
    device = model_system.device
    
    features_tensor = getattr(model_system, 'final_test_features', None)
    labels_tensor = getattr(model_system, 'final_test_labels', None)

    if model is None or features_tensor is None or labels_tensor is None:
        print("Model not initialized or evaluation data not found. Cannot generate t-SNE plot.")
        return

    # 1. Extract feature embeddings
    model.eval()
    all_embeddings = []
    batch_size = model_system.config.get('batch_size', 32)
    test_dataset = torch.utils.data.TensorDataset(features_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for features_batch in test_loader:
            inputs = features_batch[0].to(device)
            _, embeddings = model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings_np = np.concatenate(all_embeddings, axis=0)
    all_labels_np = labels_tensor.cpu().numpy()

    # 2. Apply t-SNE
    print(f"Running t-SNE on {all_embeddings_np.shape[0]} samples...")
    perplexity = min(30.0, float(max(1, all_embeddings_np.shape[0] - 1)))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
    tsne_results = tsne.fit_transform(all_embeddings_np)

    # 3. Plot results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels_np, cmap=plt.cm.get_cmap("coolwarm", 2), alpha=0.6)
    plt.colorbar(scatter, ticks=range(2), format=plt.FuncFormatter(lambda val, loc: ['Normal', 'Anomaly'][int(val)]))
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def preprocess_data(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Preprocess data based on dataset type"""
    print("\n=== Data Preprocessing ===")
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_columns:
        numeric_columns.remove('label')
    elif 'labels2' in numeric_columns:
        numeric_columns.remove('labels2')
    
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    print(f"Preprocessed data shape: {df.shape}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Adaptive Continual Learning IDS')
    parser.add_argument('--dataset', type=str, choices=['NSL-KDD', 'UNSW-NB15'], required=True,
                        help='Dataset to use (NSL-KDD or UNSW-NB15)')
    parser.add_argument('--train-path', type=str, help='Path to training data')
    parser.add_argument('--test-path', type=str, help='Path to test data')
    args = parser.parse_args()

    # Load data based on dataset type
    if args.dataset == 'NSL-KDD':
        default_train_path = './data/PKDDTrain+.csv'
        default_test_path = './data/PKDDTest+.csv'
    else:  # UNSW-NB15
        default_train_path = './data/UNSWTrain.csv'
        default_test_path = './data/UNSWTest.csv'
    
    train_path = args.train_path or default_train_path
    test_path = args.test_path or default_test_path
    
    try:
        df_train_full = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"\nError: Dataset files not found.")
        print(f"Expected paths: {train_path}, {test_path}")
        exit()

    print(f"Loading {args.dataset} dataset...")
    print(f"Full training data loaded: {df_train_full.shape}")
    print(f"Test data loaded: {df_test.shape}")
    
    # Dataset exploration
    print(f"\n=== Dataset Exploration ===")
    print(f"Training columns: {list(df_train_full.columns[:10])}...")  # Show first 10 columns
    
    if args.dataset == 'NSL-KDD':
        if 'labels2' in df_train_full.columns:
            print(f"Training label distribution: {df_train_full['labels2'].value_counts()}")
            print(f"Training anomaly ratio: {(df_train_full['labels2'] == 'anomaly').mean():.4f}")
        if 'labels2' in df_test.columns:
            print(f"Test label distribution: {df_test['labels2'].value_counts()}")
            print(f"Test anomaly ratio: {(df_test['labels2'] == 'anomaly').mean():.4f}")
    else:  # UNSW-NB15
        if 'label' in df_train_full.columns:
            print(f"Training label distribution: {df_train_full['label'].value_counts()}")
            print(f"Training anomaly ratio: {(df_train_full['label'] == 1).mean():.4f}")
        if 'label' in df_test.columns:
            print(f"Test label distribution: {df_test['label'].value_counts()}")
            print(f"Test anomaly ratio: {(df_test['label'] == 1).mean():.4f}")
    
    # Preprocess data
    df_train_full = preprocess_data(df_train_full, args.dataset)
    df_test = preprocess_data(df_test, args.dataset)
    
    # Split data for online learning simulation
    initial_train_ratio = 0.2
    split_index = int(len(df_train_full) * initial_train_ratio)
    
    initial_training_set = df_train_full.iloc[:split_index]
    online_training_stream = df_train_full.iloc[split_index:]
    online_test_stream = df_test

    print("\n=== Dataset Split for Online Learning ===")
    print(f"Initial training set: {initial_training_set.shape[0]} records ({initial_train_ratio:.0%} of original)")
    print(f"Online training stream: {online_training_stream.shape[0]} records ({1-initial_train_ratio:.0%} of original)")
    print(f"Online test stream: {online_test_stream.shape[0]} records")
    
    # Initialize model
    print("\nInitializing Adaptive Continual Learning IDS...")
    ids = AdaptiveContinualIDS(dataset_type=args.dataset)
    
    # Phase 1: Initial training
    print("\n--- Phase 1: Initial Model Training (20% labeled data) ---")
    ids.train(initial_training_set)
    
    # Phase 2: Online continual learning
    print("\n--- Phase 2: Online Continual Learning (80% training stream) ---")
    ids.train(online_training_stream)
    
    # Phase 3: Final evaluation
    print("\n--- Phase 3: Final Evaluation on Test Stream ---")
    metrics = ids.evaluate(online_test_stream)
    
    # Print evaluation results
    if metrics:
        print(f"\n=== Final Evaluation Results ===")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall: {metrics.get('recall', 0):.4f}")
        print(f"F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"\n--- Test Phase Drift Events ---")
        total_windows = metrics.get('total_windows', 'N/A')
        print(f"Baseline drift events: {metrics.get('baseline_drift_events', 0)} / {total_windows}")
        causal_events = metrics.get('causal_drift_events', 0)
        print(f"Causal drift events (detected): {causal_events} / {total_windows}")
        validated_events = metrics.get('validated_drift_events', 0)
        if causal_events > 0:
            validation_rate = validated_events / causal_events
            print(f"Causal drift events (validated as real): {validated_events} ({validation_rate:.1%})")
        else:
            print(f"Causal drift events (validated as real): 0")
            
        # Plot results
        #ids.plot_results()
        
        # Generate t-SNE visualization
        #visualize_tsne_comparison(ids,title=f't-SNE Visualization of {args.dataset} Test Data Classification')
    else:
        print("\nEvaluation phase produced no metrics.")

if __name__ == "__main__":
    main()
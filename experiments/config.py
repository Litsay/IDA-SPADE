"""Global configuration for the four-phase experiment framework.

Updated to match paper specifications:
- MLP: [128, 64, 32], dropout=0.2, batch=32
- EWC lambda: 0.1
- Non-overlapping windows T=1000
- PC-DriftForecasting parameters
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Python interpreter (CL conda env)
PYTHON_EXE = r"C:\Users\Litsay\anaconda3\envs\CL\python.exe"

# ---- Streaming parameters ----
WINDOW_SIZE = 1000
SLIDE_SIZE = 1000  # Non-overlapping (paper spec)
FADING_FACTOR = 0.998  # alpha for prequential evaluation

# ---- Model parameters (paper Section IV-B) ----
HIDDEN_DIMS = (128, 64, 32)  # Paper specification
LEARNING_RATE = 0.001
BATCH_SIZE = 32  # Paper specification
DROPOUT = 0.2  # Paper specification
EWC_LAMBDA = 0.5  # Tuned (joint grid: sigma=1.0, ewc=0.5)

# ---- PC-DriftForecasting (Algorithm 2) ----
PC_E = 3  # Embedding dimension
PC_TAU = 1  # Time delay
# Topological weights (Eq. 20)
PC_ALPHA = 0.4
PC_BETA = 0.35
PC_GAMMA = 0.25
# Adaptive threshold (Eq. 22)
PC_SIGMA = 1.0  # Tuned (joint grid best)
PC_BASELINE_WINDOW = 10
# LID parameters (Eq. 10)
PC_THETA_LID = 15.0
PC_K_LID = 10
# Cross-manifold mapping (Eq. 12, 14)
PC_LAMBDA_DECAY = 1.0
PC_ETA = 2.0
# Minimum windows before causal analysis
MIN_WINDOWS_CAUSALITY = 5

# ---- Contrastive Learning (Manifold-Guided SupCon) ----
CONTRASTIVE_ALPHA = 0.15      # weight of L_mcon in total loss (tuned)
CONTRASTIVE_TAU = 0.1         # temperature for SupCon
CONTRASTIVE_PROJ_DIM = 32     # projection head output dimension

# ---- Prototype Module ----
PROTOTYPE_BETA_STABLE = 0.99  # EMA decay during stable
PROTOTYPE_BETA_DRIFT = 0.9    # EMA decay during drift
PROTOTYPE_WEIGHT_BASE = 0.2   # base fusion weight for prototype scores
PROTOTYPE_WEIGHT_REVERSAL = 0.6  # fusion weight during reversal

# ---- Temporal Context Buffer ----
TEMPORAL_BUFFER_LEN = 3       # number of past windows to cache per entity

# ---- Layered Adaptation ----
LR_PRE_DRIFT_SCALE = 1.5     # LR multiplier during pre_drift
LR_DRIFT_SCALE = 2.0         # LR multiplier during drift
EWC_PRE_DRIFT_SCALE = 0.125  # EWC lambda multiplier during pre_drift (lambda/8)

# ---- Drift detection ----
BASELINE_DRIFT_THRESHOLD = 0.05
ADWIN_DELTA = 0.002  # ADWIN sensitivity (for baselines)

# ---- Dataset paths ----
DATASET_CONFIGS = {
    'NSL-KDD': {
        'train_path': os.path.join(BASE_DIR, 'NSL_pre_data', 'PKDDTrain+.csv'),
        'test_path': os.path.join(BASE_DIR, 'NSL_pre_data', 'PKDDTest+.csv'),
        'label_col': 'labels2',
        'multiclass_col': 'labels5',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.465,
        'attack_classes': ['normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        'proto_prefix': 'protocol_type_',
        'service_prefix': 'service_',
    },
    'UNSW-NB15': {
        'train_path': os.path.join(BASE_DIR, 'UNSW_pre_data', 'UNSWTrain.csv'),
        'test_path': os.path.join(BASE_DIR, 'UNSW_pre_data', 'UNSWTest.csv'),
        'original_train_path': os.path.join(BASE_DIR, 'UNSW_pre_data', 'UNSW_NB15_training-set.csv'),
        'original_test_path': os.path.join(BASE_DIR, 'UNSW_pre_data', 'UNSW_NB15_testing-set.csv'),
        'label_col': 'label',
        'multiclass_col': 'attack_cat',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.087,
        'attack_classes': ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
                           'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic'],
        'proto_prefix': 'proto_',
        'service_prefix': 'service_',
    },
    'CIC-IDS-2017': {
        'data_dir': r'D:\CIC2017-IDS\MachineLearningCSV',
        'label_col': 'label_binary',
        'label_positive': 1,
        'anomaly_ratio_baseline': 0.19,
        'attack_classes': ['BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk',
                           'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest',
                           'Heartbleed', 'Web Attack', 'Infiltration',
                           'Bot', 'PortScan', 'DDoS'],
        'proto_prefix': 'cic_proto_',
        'service_prefix': 'cic_service_',
    },
}

# ---- Phase 2: Drift injection ----
DRIFT_INJECTION = {
    'NSL-KDD': {
        'injection_points': [
            {'window': 15, 'from_class': 'DoS', 'to_class': 'R2L', 'description': 'DoS->R2L transition'},
            {'window': 30, 'from_class': 'R2L', 'to_class': 'Probe', 'description': 'R2L->Probe transition'},
        ],
    },
    'UNSW-NB15': {
        'injection_points': [
            {'window': 15, 'from_class': 'Generic', 'to_class': 'Exploits', 'description': 'Generic->Exploits'},
            {'window': 30, 'from_class': 'Exploits', 'to_class': 'Reconnaissance', 'description': 'Exploits->Recon'},
        ],
    },
}

# ---- Phase 3: Knowledge retention ----
F1_RECOVERY_THRESHOLD = 0.95
STATIONARY_MIN_WINDOWS = 5

# ---- Phase 4: Ablation ----
V_SCALING_VALUES = [3, 5, 8, 10, 15, 20]
T_SCALING_VALUES = [500, 1000, 2000, 5000]

# ---- Output paths ----
RESULTS_DIR = os.path.join(BASE_DIR, 'experiment_results')
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), '实验数据模板')

# ---- Random seed ----
SEED = 42

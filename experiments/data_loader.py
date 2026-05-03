"""Streaming data loaders for NSL-KDD, UNSW-NB15, and CIC-IDS-2017.

Each loader yields Window objects with binary + multi-class labels.
"""
import os
import numpy as np
import pandas as pd
from typing import Iterator, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

from .config import WINDOW_SIZE, SLIDE_SIZE, DATASET_CONFIGS
from .streaming_interface import Window


def _create_windows(df: pd.DataFrame, feature_cols: list, label_col: str,
                    multiclass_col: str, label_positive, window_size: int,
                    slide_size: int, start_index: int = 0,
                    extra_metadata: Optional[dict] = None) -> Iterator[Window]:
    """Generic sliding window generator."""
    n = len(df)
    wid = start_index
    pos = 0
    while pos + window_size <= n:
        chunk = df.iloc[pos:pos + window_size]
        X = chunk[feature_cols].values.astype(np.float32)
        y_bin = (chunk[label_col] == label_positive).values.astype(np.int64)
        y_mc = chunk[multiclass_col].values if multiclass_col in chunk.columns else np.full(len(chunk), 'unknown')
        meta = {
            'anomaly_ratio': float(y_bin.mean()),
            'size': len(chunk),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        yield Window(index=wid, X=X, y_binary=y_bin, y_multiclass=y_mc, metadata=meta)
        pos += slide_size
        wid += 1


def load_nsl_kdd(window_size: int = WINDOW_SIZE, slide_size: int = SLIDE_SIZE
                 ) -> Tuple[np.ndarray, np.ndarray, Iterator[Window], list]:
    """Load NSL-KDD dataset.

    Returns:
        X_init, y_init: initial training data (first 20% of train)
        stream: Iterator of Window objects (remaining 80% train + test)
        feature_cols: list of feature column names
    """
    cfg = DATASET_CONFIGS['NSL-KDD']
    df_train = pd.read_csv(cfg['train_path'])
    df_test = pd.read_csv(cfg['test_path'])

    # Handle label conversion
    lc = cfg['label_col']
    mc = cfg['multiclass_col']
    if df_train[lc].dtype == object:
        df_train[lc] = (df_train[lc] != 'normal').astype(int)
        df_test[lc] = (df_test[lc] != 'normal').astype(int)

    # Clean numeric columns
    exclude_cols = {lc, mc}
    feature_cols = [c for c in df_train.columns
                    if c not in exclude_cols and df_train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    for df in [df_train, df_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split: 20% init, 80% stream, then test
    split = int(len(df_train) * 0.2)
    df_init = df_train.iloc[:split]
    df_stream = df_train.iloc[split:]

    X_init = df_init[feature_cols].values.astype(np.float32)
    y_init = df_init[lc].values.astype(np.int64)

    # Combine stream + test
    df_combined = pd.concat([df_stream, df_test], ignore_index=True)

    stream = _create_windows(df_combined, feature_cols, lc, mc,
                             cfg['label_positive'], window_size, slide_size)
    return X_init, y_init, stream, feature_cols


def load_unsw_nb15(window_size: int = WINDOW_SIZE, slide_size: int = SLIDE_SIZE
                   ) -> Tuple[np.ndarray, np.ndarray, Iterator[Window], list]:
    """Load UNSW-NB15 dataset with multi-class labels from original files."""
    cfg = DATASET_CONFIGS['UNSW-NB15']
    df_train = pd.read_csv(cfg['train_path'])
    df_test = pd.read_csv(cfg['test_path'])

    # Add multi-class labels from original files
    mc = cfg['multiclass_col']
    if mc not in df_train.columns:
        orig_train = pd.read_csv(cfg['original_train_path'])
        orig_test = pd.read_csv(cfg['original_test_path'])
        # Align by index (preprocessed files maintain row order)
        if len(orig_train) == len(df_train):
            df_train[mc] = orig_train[mc].values
        else:
            # Fallback: binary to Normal/Attack
            df_train[mc] = np.where(df_train[cfg['label_col']] == 0, 'Normal', 'Attack')
        if len(orig_test) == len(df_test):
            df_test[mc] = orig_test[mc].values
        else:
            df_test[mc] = np.where(df_test[cfg['label_col']] == 0, 'Normal', 'Attack')

    lc = cfg['label_col']
    exclude_cols = {lc, mc}
    feature_cols = [c for c in df_train.columns
                    if c not in exclude_cols and df_train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    for df in [df_train, df_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    split = int(len(df_train) * 0.2)
    df_init = df_train.iloc[:split]
    df_stream = df_train.iloc[split:]

    X_init = df_init[feature_cols].values.astype(np.float32)
    y_init = df_init[lc].values.astype(np.int64)

    df_combined = pd.concat([df_stream, df_test], ignore_index=True)
    stream = _create_windows(df_combined, feature_cols, lc, mc,
                             cfg['label_positive'], window_size, slide_size)
    return X_init, y_init, stream, feature_cols


def load_cic_ids_2017(data_dir: str, window_size: int = WINDOW_SIZE,
                      slide_size: int = SLIDE_SIZE
                      ) -> Tuple[np.ndarray, np.ndarray, Iterator[Window], list]:
    """Load CIC-IDS-2017 dataset.

    Monday = initial training (benign only, subsampled to 50K)
    Tuesday-Friday = streaming evaluation

    Handles multi-file days (Thursday=2 files, Friday=3 files).
    Adds synthetic protocol/service columns for ECBA entity grouping.
    """
    # Day-to-files mapping (Thursday and Friday have multiple files)
    day_files = {
        'Monday': ['Monday-WorkingHours.pcap_ISCX.csv'],
        'Tuesday': ['Tuesday-WorkingHours.pcap_ISCX.csv'],
        'Wednesday': ['Wednesday-workingHours.pcap_ISCX.csv'],
        'Thursday': [
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        ],
        'Friday': [
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        ],
    }

    def _add_entity_columns(df):
        """Add synthetic protocol/service one-hot columns from Destination Port."""
        port_col = 'Destination Port'
        if port_col not in df.columns:
            # Fallback: try to find it
            for c in df.columns:
                if 'destination' in c.lower() and 'port' in c.lower():
                    port_col = c
                    break
        if port_col in df.columns:
            port = pd.to_numeric(df[port_col], errors='coerce').fillna(0).astype(int)
            df['cic_service_web'] = port.isin([80, 443, 8080, 8443]).astype(int)
            df['cic_service_ssh'] = port.isin([22]).astype(int)
            df['cic_service_ftp'] = port.isin([20, 21]).astype(int)
            df['cic_service_dns'] = port.isin([53]).astype(int)
            df['cic_service_mail'] = port.isin([25, 110, 143, 993, 995]).astype(int)
            df['cic_service_other'] = (~port.isin([80,443,8080,8443,22,20,21,53,25,110,143,993,995])).astype(int)
            df['cic_proto_system'] = (port < 1024).astype(int)
            df['cic_proto_ephemeral'] = (port >= 1024).astype(int)
        return df

    def _load_day(day_name):
        """Load and concatenate all CSV files for a given day."""
        files = day_files[day_name]
        dfs = []
        for fname in files:
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                # Try alternate naming
                for f in os.listdir(data_dir):
                    if day_name.lower() in f.lower() and fname.split('.')[0].lower().replace('-', '') in f.lower().replace('-', '') and f.endswith('.csv'):
                        path = os.path.join(data_dir, f)
                        break
            if os.path.exists(path):
                df = pd.read_csv(path, encoding='utf-8', low_memory=False)
                df.columns = df.columns.str.strip()
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No CSV files found for {day_name} in {data_dir}")
        df = pd.concat(dfs, ignore_index=True)

        # Label column
        label_col = 'Label'
        if label_col not in df.columns:
            for c in df.columns:
                if 'label' in c.lower():
                    label_col = c
                    break
        # Binary label
        df['label_binary'] = (df[label_col].str.strip() != 'BENIGN').astype(int)
        df['attack_class'] = df[label_col].str.strip()

        # Add entity grouping columns
        df = _add_entity_columns(df)

        # Numeric features only
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['label_binary']]
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df, num_cols, label_col

    # Monday: initial training (subsampled to 50K)
    df_mon, feature_cols, _ = _load_day('Monday')
    if len(df_mon) > 50000:
        df_mon = df_mon.sample(n=50000, random_state=42).reset_index(drop=True)
    X_init = df_mon[feature_cols].values.astype(np.float32)
    y_init = df_mon['label_binary'].values.astype(np.int64)

    # Tuesday-Friday: streaming
    def _stream_generator():
        wid = 0
        for day in ['Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            df_day, fcols, _ = _load_day(day)
            # Use same feature set as Monday (intersection)
            common_cols = [c for c in feature_cols if c in df_day.columns]
            n = len(df_day)
            pos = 0
            while pos + window_size <= n:
                chunk = df_day.iloc[pos:pos + window_size]
                X = chunk[common_cols].values.astype(np.float32)
                # Pad if feature mismatch
                if X.shape[1] < len(feature_cols):
                    pad = np.zeros((X.shape[0], len(feature_cols) - X.shape[1]), dtype=np.float32)
                    X = np.hstack([X, pad])
                y_bin = chunk['label_binary'].values.astype(np.int64)
                y_mc = chunk['attack_class'].values
                meta = {
                    'day': day,
                    'anomaly_ratio': float(y_bin.mean()),
                    'size': len(chunk),
                }
                yield Window(index=wid, X=X, y_binary=y_bin, y_multiclass=y_mc, metadata=meta)
                pos += slide_size
                wid += 1

    return X_init, y_init, _stream_generator(), feature_cols


def load_dataset(dataset_name: str, window_size: int = WINDOW_SIZE,
                 slide_size: int = SLIDE_SIZE, cic_data_dir: str = None
                 ) -> Tuple[np.ndarray, np.ndarray, Iterator[Window], list]:
    """Unified dataset loader.

    Returns: (X_init, y_init, stream_iterator, feature_columns)
    """
    if dataset_name == 'NSL-KDD':
        return load_nsl_kdd(window_size, slide_size)
    elif dataset_name == 'UNSW-NB15':
        return load_unsw_nb15(window_size, slide_size)
    elif dataset_name == 'CIC-IDS-2017':
        if cic_data_dir is None:
            cic_data_dir = DATASET_CONFIGS.get('CIC-IDS-2017', {}).get('data_dir')
        if cic_data_dir is None:
            raise ValueError("cic_data_dir must be provided for CIC-IDS-2017")
        return load_cic_ids_2017(cic_data_dir, window_size, slide_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

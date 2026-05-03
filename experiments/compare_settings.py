"""Compare all models under two evaluation settings:

Setting A (Traditional / Standalone-like):
  - Phase 0: Initialize on 20% train
  - Phase 1: Sequential update on remaining 80% train (train-only, no testing)
  - Phase 2: Evaluate on held-out test set
  → Simulates the standalone pipeline's train/test protocol

Setting B (Prequential / Phase 1):
  - Initialize on 20% train
  - Test-then-Train on remaining 80% train + test combined stream
  → Streaming evaluation with fading-weighted metrics

Usage:
    python -m experiments.compare_settings --datasets NSL-KDD UNSW-NB15
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import time
from typing import List, Dict, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import (
    SEED, WINDOW_SIZE, SLIDE_SIZE, DATASET_CONFIGS, FADING_FACTOR
)
from experiments.streaming_interface import StreamingModel, Window
from experiments.evaluator import prequential_evaluate, aggregate_metrics, compute_fading_average
from experiments.ida_spade_wrapper import IDASpadeStreaming
from experiments.baselines.ewc_baseline import EWCBaseline
from experiments.baselines.lwf_baseline import LwFBaseline
from experiments.baselines.ssf_baseline import SSFBaseline
from experiments.baselines.aoc_ids import AOCIDSBaseline
from experiments.baselines.cids_baseline import CIDSBaseline
from experiments.baselines.card_baseline import CARDBaseline
from experiments.baselines.feco_baseline import FeCoBaseline
from experiments.baselines.unflows_baseline import UnFlowsBaseline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_windows(df, feature_cols, label_col, label_positive,
                  multiclass_col, window_size=WINDOW_SIZE):
    """Create Window objects from a DataFrame."""
    windows = []
    n = len(df)
    wid = 0
    pos = 0
    while pos + window_size <= n:
        chunk = df.iloc[pos:pos + window_size]
        X = chunk[feature_cols].values.astype(np.float32)
        y_bin = (chunk[label_col] == label_positive).values.astype(np.int64)
        y_mc = chunk[multiclass_col].values if multiclass_col in chunk.columns else np.full(len(chunk), 'unknown')
        meta = {'anomaly_ratio': float(y_bin.mean()), 'size': len(chunk)}
        windows.append(Window(index=wid, X=X, y_binary=y_bin, y_multiclass=y_mc, metadata=meta))
        pos += window_size
        wid += 1
    return windows


def load_split_data(ds_name):
    """Load dataset with SEPARATE train and test window lists.

    Returns: X_init, y_init, train_windows, test_windows, feature_cols
    """
    cfg = DATASET_CONFIGS[ds_name]
    df_train = pd.read_csv(cfg['train_path'])
    df_test = pd.read_csv(cfg['test_path'])

    lc = cfg['label_col']
    mc = cfg.get('multiclass_col', lc)

    if ds_name == 'NSL-KDD' and df_train[lc].dtype == object:
        df_train[lc] = (df_train[lc] != 'normal').astype(int)
        df_test[lc] = (df_test[lc] != 'normal').astype(int)

    # UNSW: add multiclass col if missing
    if ds_name == 'UNSW-NB15' and mc not in df_train.columns:
        orig_train_path = cfg.get('original_train_path')
        orig_test_path = cfg.get('original_test_path')
        if orig_train_path and os.path.exists(orig_train_path):
            orig_train = pd.read_csv(orig_train_path)
            orig_test = pd.read_csv(orig_test_path)
            if len(orig_train) == len(df_train):
                df_train[mc] = orig_train[mc].values
            else:
                df_train[mc] = np.where(df_train[lc] == 0, 'Normal', 'Attack')
            if len(orig_test) == len(df_test):
                df_test[mc] = orig_test[mc].values
            else:
                df_test[mc] = np.where(df_test[lc] == 0, 'Normal', 'Attack')
        else:
            df_train[mc] = np.where(df_train[lc] == 0, 'Normal', 'Attack')
            df_test[mc] = np.where(df_test[lc] == 0, 'Normal', 'Attack')

    exclude_cols = {lc, mc}
    feature_cols = [c for c in df_train.columns
                    if c not in exclude_cols
                    and df_train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    for df in [df_train, df_test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split train: 20% init + 80% train windows
    split = int(len(df_train) * 0.2)
    df_init = df_train.iloc[:split]
    df_train_stream = df_train.iloc[split:]

    X_init = df_init[feature_cols].values.astype(np.float32)
    y_init = df_init[lc].values.astype(np.int64)

    init_windows = _make_windows(df_init, feature_cols, lc,
                                 cfg['label_positive'], mc)
    train_windows = _make_windows(df_train_stream, feature_cols, lc,
                                  cfg['label_positive'], mc)
    test_windows = _make_windows(df_test, feature_cols, lc,
                                 cfg['label_positive'], mc)

    return X_init, y_init, init_windows, train_windows, test_windows, feature_cols


def traditional_evaluate(model: StreamingModel, init_windows: List[Window],
                         train_windows: List[Window],
                         test_windows: List[Window],
                         X_init: np.ndarray, y_init: np.ndarray) -> Dict:
    """Setting A: Traditional train/test evaluation.

    Faithfully reproduces the standalone pipeline:
    1. Phase 0: Window-by-window init training (same as standalone)
    2. Phase 1: Window-by-window online training (same as standalone)
    3. Phase 2: Evaluate on held-out test (predict only, no update)

    All phases process data window-by-window, matching the standalone's
    process_stream() which aggregates per window.
    """
    # Phase 0: Window-by-window initialization
    # First window used for model init, all windows trained with init epochs
    first = True
    for w in init_windows:
        if first:
            model.initialize(w.X, w.y_binary)
            first = True  # initialize already trains
        else:
            model.detect_drift(w.X)
            model.update(w.X, w.y_binary)
    if not init_windows:
        model.initialize(X_init, y_init)

    # Phase 1: Online training on remaining train windows
    for w in train_windows:
        model.detect_drift(w.X)
        model.update(w.X, w.y_binary)

    # Phase 2: Evaluate on test windows (no training)
    all_preds = []
    all_labels = []
    for w in test_windows:
        preds, y_true = model.predict_evaluate(w.X, w.y_binary)
        if len(preds) > 0:
            all_preds.append(preds)
            all_labels.append(y_true)

    if not all_preds:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'n_test_samples': 0}

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'n_test_samples': len(all_labels),
    }


def create_models(feature_cols=None, dataset_name='NSL-KDD'):
    return {
        'IDA-SPADE': IDASpadeStreaming(
            name='IDA-SPADE', feature_cols=feature_cols, dataset_name=dataset_name),
        'SSF': SSFBaseline(),
        'AOC-IDS': AOCIDSBaseline(),
        'EWC': EWCBaseline(),
        'LwF': LwFBaseline(),
        'CIDS': CIDSBaseline(),
        'CARD': CARDBaseline(),
        'FeCo': FeCoBaseline(),
        'unFlowS': UnFlowsBaseline(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['NSL-KDD', 'UNSW-NB15'])
    args = parser.parse_args()

    for ds_name in args.datasets:
        print(f"\n{'='*90}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*90}")

        # Load data with separate init/train/test splits
        X_init, y_init, init_windows, train_windows, test_windows, feature_cols = load_split_data(ds_name)
        print(f"  Init: {X_init.shape} ({len(init_windows)} windows), "
              f"Train: {len(train_windows)} windows, Test: {len(test_windows)} windows")
        print(f"  Init attack ratio: {y_init.mean()*100:.1f}%")

        # Also load combined stream for prequential
        from experiments.data_loader import load_dataset
        _, _, stream_combined, _ = load_dataset(ds_name)
        combined_windows = list(stream_combined)

        print(f"\n  {'Model':<22} {'Setting A (Traditional)':<32} {'Setting B (Prequential)':<32}")
        print(f"  {'':22} {'F1':>7} {'Pre':>7} {'Rec':>7} {'Acc':>7}   {'F1':>7} {'Pre':>7} {'Rec':>7} {'Acc':>7}")
        print(f"  {'-'*86}")

        models_a = create_models(feature_cols=feature_cols, dataset_name=ds_name)
        models_b = create_models(feature_cols=feature_cols, dataset_name=ds_name)

        for model_name in models_a:
            # Setting A: Traditional
            set_seed()
            t0 = time.time()
            result_a = traditional_evaluate(
                models_a[model_name], init_windows, train_windows,
                test_windows, X_init, y_init)
            time_a = time.time() - t0

            # Setting B: Prequential
            set_seed()
            t0 = time.time()
            results_b = prequential_evaluate(
                models_b[model_name], iter(combined_windows), X_init, y_init,
                verbose=False)
            time_b = time.time() - t0
            agg_b = aggregate_metrics(results_b)

            print(f"  {model_name:<22} "
                  f"{result_a['f1']*100:6.2f}% {result_a['precision']*100:6.2f}% "
                  f"{result_a['recall']*100:6.2f}% {result_a['accuracy']*100:6.2f}%   "
                  f"{agg_b['f1']*100:6.2f}% {agg_b['precision']*100:6.2f}% "
                  f"{agg_b['recall']*100:6.2f}% {agg_b['accuracy']*100:6.2f}%")

        print()


if __name__ == '__main__':
    main()

"""CIC-IDS-2017 Full Evaluation: Per-Day + Aggregate + Ablation.

Usage:
    /c/Users/Litsay/anaconda3/envs/CL/python.exe -u -m experiments.run_cic_full_eval
"""
import sys
import os
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from experiments.config import SEED, WINDOW_SIZE, SLIDE_SIZE, FADING_FACTOR
from experiments.streaming_interface import Window
from experiments.evaluator import prequential_evaluate, aggregate_metrics
from experiments.run_all import set_seed, create_models, create_ablation_models
from experiments.data_loader import load_dataset
from experiments.run_cic_perday import load_day, create_windows, DAY_ATTACKS

EXCLUDE_BASELINES = ['SSF', 'AOC-IDS', 'EWC', 'LwF', 'CIDS', 'CARD', 'FeCo', 'unFlowS']


def fmt(agg, elapsed=None):
    s = (f"F1={agg['f1']*100:6.2f}% Acc={agg['accuracy']*100:6.2f}% "
         f"Pre={agg['precision']*100:6.2f}% Rec={agg['recall']*100:6.2f}% "
         f"AUC={agg['auc']*100:6.2f}% F1std={agg['f1_std']:.4f} "
         f"Drifts={agg['n_drifts']}/{agg['n_windows']}")
    if elapsed is not None:
        s += f" ({elapsed:.0f}s)"
    return s


def main():
    set_seed(SEED)
    results_all = {}

    # ================================================================
    # PART 1: Per-Day Phase 1
    # ================================================================
    print("=" * 80)
    print("PART 1: Per-Day Phase 1 — IDA-SPADE (S1+S3)")
    print("=" * 80)
    sys.stdout.flush()

    df_mon, feature_cols = load_day('Monday')
    if len(df_mon) > 50000:
        df_mon = df_mon.sample(n=50000, random_state=42).reset_index(drop=True)
    X_init = df_mon[feature_cols].values.astype(np.float32)
    y_init = df_mon['label_binary'].values.astype(np.int64)
    print(f"  Init: {X_init.shape}, features: {len(feature_cols)}")
    sys.stdout.flush()

    test_days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday']
    perday_results = {}

    for day in test_days:
        df_day, _ = load_day(day)
        windows = create_windows(df_day, feature_cols)
        models = create_models(
            feature_cols=feature_cols, dataset_name='CIC-IDS-2017',
            exclude=EXCLUDE_BASELINES)
        for mn, m in models.items():
            set_seed(SEED)
            t0 = time.time()
            r = prequential_evaluate(
                m, iter(windows), X_init, y_init,
                alpha=FADING_FACTOR, verbose=False)
            elapsed = time.time() - t0
            agg = aggregate_metrics(r)
            agg['time'] = elapsed
            print(f"  {day:<12s} {fmt(agg, elapsed)}")
            sys.stdout.flush()
            perday_results[day] = agg

    results_all['perday'] = perday_results

    # ================================================================
    # PART 2: Aggregate Phase 1 (sequential all days)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PART 2: Aggregate Phase 1 — Sequential streaming")
    print("=" * 80)
    sys.stdout.flush()

    X_init_seq, y_init_seq, stream, feat_cols_seq = load_dataset('CIC-IDS-2017')
    windows_seq = list(stream)
    print(f"  Init: {X_init_seq.shape}, Stream: {len(windows_seq)} windows")
    sys.stdout.flush()

    set_seed(SEED)
    model = create_models(
        feature_cols=feat_cols_seq, dataset_name='CIC-IDS-2017',
        exclude=EXCLUDE_BASELINES)['IDA-SPADE']
    t0 = time.time()
    r = prequential_evaluate(
        model, iter(windows_seq), X_init_seq, y_init_seq,
        alpha=FADING_FACTOR, verbose=False)
    elapsed = time.time() - t0
    agg = aggregate_metrics(r)
    agg['time'] = elapsed
    print(f"  IDA-SPADE    {fmt(agg, elapsed)}")
    sys.stdout.flush()
    results_all['aggregate'] = agg

    # ================================================================
    # PART 3: Phase 4 Ablation
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PART 3: Phase 4 Ablation — 4 variants")
    print("=" * 80)
    sys.stdout.flush()

    ablation_models = create_ablation_models(
        feature_cols=feat_cols_seq, dataset_name='CIC-IDS-2017')
    ablation_results = {}
    for name, abl_model in ablation_models.items():
        set_seed(SEED)
        t0 = time.time()
        r = prequential_evaluate(
            abl_model, iter(windows_seq), X_init_seq, y_init_seq,
            alpha=FADING_FACTOR, verbose=False)
        elapsed = time.time() - t0
        agg = aggregate_metrics(r)
        agg['time'] = elapsed
        print(f"  {name:<25s} {fmt(agg, elapsed)}")
        sys.stdout.flush()
        ablation_results[name] = agg

    results_all['ablation'] = ablation_results

    # Save
    os.makedirs('experiment_results', exist_ok=True)
    out_path = os.path.join('experiment_results', 'cic_s1s3_full_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()

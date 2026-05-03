"""Main experiment runner: executes all four phases and exports results.

Paper experiment design:
- Phase 1: Baseline NID performance (Tab. 1) — 8 models x 2 datasets
- Phase 2: Concept drift perception (Tab. 2 + Fig. 3)
- Phase 3: Ablation study (Tab. 3)
- Phase 4: Computation efficiency (Tab. 4)

Usage:
    python -m experiments.run_all --datasets NSL-KDD UNSW-NB15
    python -m experiments.run_all --phase 1 --datasets NSL-KDD
"""
import sys
import os
import argparse
import numpy as np
import torch
import time
from collections import defaultdict
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import (
    SEED, RESULTS_DIR, WINDOW_SIZE, SLIDE_SIZE,
    DATASET_CONFIGS, V_SCALING_VALUES, T_SCALING_VALUES,
    FADING_FACTOR
)
from experiments.data_loader import load_dataset
from experiments.streaming_interface import Window
from experiments.evaluator import prequential_evaluate, aggregate_metrics
from experiments.ida_spade_wrapper import IDASpadeStreaming
from experiments.baselines.ewc_baseline import EWCBaseline
from experiments.baselines.lwf_baseline import LwFBaseline
from experiments.baselines.ssf_baseline import SSFBaseline
from experiments.baselines.aoc_ids import AOCIDSBaseline
from experiments.baselines.unflows_adwin import UnFlowsADWINBaseline
from experiments.baselines.feco_adwin import FeCoADWINBaseline
from experiments.baselines.cids_baseline import CIDSBaseline
from experiments.baselines.card_baseline import CARDBaseline
from experiments.baselines.feco_baseline import FeCoBaseline
from experiments.baselines.unflows_baseline import UnFlowsBaseline
from experiments.ablation import (
    IDASpadeReactive, IDASpadeGlobal, IDASpadeStatistical,
    IDASpadeNoCL, IDASpadeNoProto, profile_components
)
from experiments.drift_injection import (
    identify_drift_points, extract_time_anchors, DriftEvent,
    compute_phase2_summary, compute_detection_quality,
    find_natural_drift_points
)
from experiments.knowledge_retention import (
    identify_stationary_phases, compute_per_class_recall, compute_retention_summary
)
from experiments import export


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_models(feature_cols=None, dataset_name='NSL-KDD', exclude=None):
    """Create streaming models (IDA-SPADE + baselines). Use exclude to skip slow models."""
    all_models = {
        'IDA-SPADE': IDASpadeStreaming(
            name='IDA-SPADE', feature_cols=feature_cols, dataset_name=dataset_name),
        # CL-based methods
        'SSF': SSFBaseline(),
        'AOC-IDS': AOCIDSBaseline(),
        'EWC': EWCBaseline(),
        'LwF': LwFBaseline(),
        # Static methods (paper requires 4 static baselines)
        'CIDS': CIDSBaseline(),
        'CARD': CARDBaseline(),
        'FeCo': FeCoBaseline(),
        'unFlowS': UnFlowsBaseline(),
    }
    if exclude:
        for name in exclude:
            all_models.pop(name, None)
    return all_models


def create_ablation_models(feature_cols=None, dataset_name='NSL-KDD'):
    """Create ablation variant models (paper Table 3)."""
    return {
        'IDA-SPADE': IDASpadeStreaming(
            name='IDA-SPADE', feature_cols=feature_cols, dataset_name=dataset_name),
        'IDA-SPADE-Reactive': IDASpadeReactive(
            feature_cols=feature_cols, dataset_name=dataset_name),
        'IDA-SPADE-Global': IDASpadeGlobal(
            feature_cols=feature_cols, dataset_name=dataset_name),
        'IDA-SPADE-Statistical': IDASpadeStatistical(
            feature_cols=feature_cols, dataset_name=dataset_name),
        'IDA-SPADE-NoCL': IDASpadeNoCL(
            feature_cols=feature_cols, dataset_name=dataset_name),
        'IDA-SPADE-NoProto': IDASpadeNoProto(
            feature_cols=feature_cols, dataset_name=dataset_name),
    }


def _collect_stream(stream):
    return list(stream)


# =====================================================================
# Phase 1: Prequential Baseline (Paper Tab. 1)
# =====================================================================

def run_phase1(datasets, output_dir):
    """Phase 1: Prequential evaluation for all models on all datasets."""
    print("\n" + "=" * 80)
    print("PHASE 1: Prequential Evaluation Baseline (Paper Tab. 1)")
    print("=" * 80)

    all_results = {}
    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        X_init, y_init, stream, feature_cols = load_dataset(ds_name)
        windows = _collect_stream(stream)
        print(f"Init: {X_init.shape}, Stream windows: {len(windows)}")

        models = create_models(feature_cols=feature_cols, dataset_name=ds_name)
        ds_results = {}

        for model_name, model in models.items():
            set_seed()
            print(f"\n  Model: {model_name}")
            t0 = time.time()
            results = prequential_evaluate(
                model, iter(windows), X_init, y_init,
                alpha=FADING_FACTOR, verbose=False
            )
            elapsed = time.time() - t0
            agg = aggregate_metrics(results)
            print(f"    F1: {agg.get('f1', 0)*100:.2f}% | "
                  f"Acc: {agg.get('accuracy', 0)*100:.2f}% | "
                  f"Pre: {agg.get('precision', 0)*100:.2f}% | "
                  f"Rec: {agg.get('recall', 0)*100:.2f}% | "
                  f"Windows: {len(results)} | Time: {elapsed:.1f}s")
            ds_results[model_name] = results

        all_results[ds_name] = ds_results

    export.export_phase1_per_window(all_results, output_dir)
    export.export_phase1_aggregate(all_results, output_dir)
    export.save_json(
        {ds: {m: aggregate_metrics(r) for m, r in models.items()}
         for ds, models in all_results.items()},
        os.path.join(output_dir, 'phase1_summary.json')
    )
    print(f"\nPhase 1 results saved to {output_dir}")
    return all_results


def run_multi_seed_phase1(datasets, output_dir, seeds):
    """Run Phase 1 across multiple seeds and compute statistics."""
    print("\n" + "=" * 80)
    print(f"MULTI-SEED Phase 1: {len(seeds)} seeds = {seeds}")
    print("=" * 80)

    # Collect per-seed aggregated metrics
    # Structure: {dataset: {model: {metric: [values_per_seed]}}}
    all_seed_results = {ds: {} for ds in datasets}
    per_seed_raw = {}  # Store full results for one representative seed

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")
        set_seed(seed)

        for ds_name in datasets:
            X_init, y_init, stream, feature_cols = load_dataset(ds_name)
            windows = _collect_stream(stream)

            models = create_models(feature_cols=feature_cols, dataset_name=ds_name)

            for model_name, model in models.items():
                set_seed(seed)
                t0 = time.time()
                results = prequential_evaluate(
                    model, iter(windows), X_init, y_init,
                    alpha=FADING_FACTOR, verbose=False)
                elapsed = time.time() - t0
                agg = aggregate_metrics(results)

                if model_name not in all_seed_results[ds_name]:
                    all_seed_results[ds_name][model_name] = {
                        'f1': [], 'accuracy': [], 'precision': [], 'recall': [],
                        'f1_std': [], 'f1_mean': [], 'n_windows': [], 'n_drifts': [],
                    }
                for k in ['f1', 'accuracy', 'precision', 'recall', 'f1_std', 'f1_mean', 'n_windows', 'n_drifts']:
                    all_seed_results[ds_name][model_name][k].append(agg.get(k, 0))

                print(f"    [{ds_name}] {model_name}: F1={agg.get('f1',0)*100:.2f}% ({elapsed:.1f}s)")

                # Store first seed's full results for per-window export
                if seed_idx == 0:
                    if ds_name not in per_seed_raw:
                        per_seed_raw[ds_name] = {}
                    per_seed_raw[ds_name][model_name] = results

    # Compute mean +/- std
    summary = {}
    for ds_name in datasets:
        summary[ds_name] = {}
        for model_name, metrics_dict in all_seed_results[ds_name].items():
            model_summary = {}
            for metric, values in metrics_dict.items():
                model_summary[f'{metric}_mean'] = float(np.mean(values))
                model_summary[f'{metric}_std'] = float(np.std(values))
                model_summary[f'{metric}_values'] = values
            summary[ds_name][model_name] = model_summary

    # Statistical tests: IDA-SPADE vs best baseline
    stat_tests = {}
    for ds_name in datasets:
        ds_tests = {}
        ida_f1s = all_seed_results[ds_name].get('IDA-SPADE', {}).get('f1', [])
        if len(ida_f1s) >= 3:
            for model_name, metrics_dict in all_seed_results[ds_name].items():
                if model_name == 'IDA-SPADE':
                    continue
                other_f1s = metrics_dict.get('f1', [])
                if len(other_f1s) == len(ida_f1s) and len(ida_f1s) >= 3:
                    try:
                        # Wilcoxon signed-rank test
                        w_stat, w_p = scipy_stats.wilcoxon(ida_f1s, other_f1s)
                        # Paired t-test
                        t_stat, t_p = scipy_stats.ttest_rel(ida_f1s, other_f1s)
                        ds_tests[model_name] = {
                            'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_p),
                            'ttest_stat': float(t_stat), 'ttest_p': float(t_p),
                            'ida_mean': float(np.mean(ida_f1s)),
                            'ida_std': float(np.std(ida_f1s)),
                            'other_mean': float(np.mean(other_f1s)),
                            'other_std': float(np.std(other_f1s)),
                        }
                    except Exception as e:
                        ds_tests[model_name] = {'error': str(e)}
        stat_tests[ds_name] = ds_tests

    # Export
    export.save_json(summary, os.path.join(output_dir, 'phase1_multi_seed_summary.json'))
    export.save_json(stat_tests, os.path.join(output_dir, 'phase1_statistical_tests.json'))
    if per_seed_raw:
        export.export_phase1_per_window(per_seed_raw, output_dir)
        export.export_phase1_aggregate(per_seed_raw, output_dir)

    # Export multi-seed markdown
    export.export_multi_seed_table(summary, stat_tests, output_dir, 'phase1')

    print(f"\nMulti-seed Phase 1 results saved to {output_dir}")
    return summary, stat_tests, per_seed_raw


def compute_f1_loss_area(f1_proactive, f1_reactive, drift_points, lookahead=15):
    """Compute cumulative F1 loss area around drift points.

    For each drift point, compute the area between the pre-drift F1 baseline
    and the actual F1 trajectory over the next `lookahead` windows.
    Lower area = faster recovery = better.
    """
    results = []
    for dp in drift_points:
        for label, f1_series in [('PC', f1_proactive), ('KS', f1_reactive)]:
            # Pre-drift baseline: avg F1 over 5 windows before drift
            pre_start = max(0, dp - 5)
            baseline_f1 = np.mean(f1_series[pre_start:dp]) if dp > 0 else 1.0

            # F1 loss area: sum of (baseline - actual) for windows after drift
            end = min(dp + lookahead, len(f1_series))
            if dp < len(f1_series):
                losses = [max(0, baseline_f1 - f1_series[i]) for i in range(dp, end)]
                area = float(np.sum(losses))
            else:
                area = 0.0
            results.append({
                'drift_point': dp,
                'method': label,
                'baseline_f1': float(baseline_f1),
                'loss_area': area,
                'n_windows': end - dp,
            })
    return results


# =====================================================================
# Phase 2: Concept Drift Perception (Paper Tab. 2 + Fig. 3)
# =====================================================================

def run_phase2(datasets, output_dir, phase1_results=None):
    """Phase 2: Concept Drift Perception — IDA-SPADE (PC) vs Reactive (KS-test).

    Uses anomaly ratio distribution changes as ground truth drift points.
    Measures: opportunity window ΔT, detection quality, F1 recovery speed.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Concept Drift Perception (Paper Tab. 2)")
    print("=" * 80)

    all_phase2 = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        X_init, y_init, stream, feature_cols = load_dataset(ds_name)
        windows = _collect_stream(stream)

        # Ground truth drift points from anomaly ratio distribution changes
        drift_points = identify_drift_points(windows, threshold=0.10, min_gap=8)
        print(f"  Ground truth drift points: {len(drift_points)}")
        if drift_points:
            print(f"    Windows: {drift_points}")

        # Run IDA-SPADE (proactive PC-DriftForecasting)
        set_seed()
        ida_spade = IDASpadeStreaming(
            name='IDA-SPADE', feature_cols=feature_cols, dataset_name=ds_name)
        results_proactive = prequential_evaluate(
            ida_spade, iter(windows), X_init, y_init, verbose=False)

        # Run IDA-SPADE-Reactive (KS-test)
        set_seed()
        reactive = IDASpadeReactive(
            feature_cols=feature_cols, dataset_name=ds_name)
        results_reactive = prequential_evaluate(
            reactive, iter(windows), X_init, y_init, verbose=False)

        # Per-window F1 series
        f1_proactive = [r.f1 for r in results_proactive]
        f1_reactive = [r.f1 for r in results_reactive]

        # Perceived drift counts
        pro_detected = sum(1 for r in results_proactive if r.drift_detected)
        rea_detected = sum(1 for r in results_reactive if r.drift_detected)
        print(f"  IDA-SPADE (PC) detected: {pro_detected}")
        print(f"  Reactive (KS) detected: {rea_detected}")

        # Drift detection quality (Precision / Recall / F1)
        qual_proactive = compute_detection_quality(
            results_proactive, drift_points, tolerance=3)
        qual_reactive = compute_detection_quality(
            results_reactive, drift_points, tolerance=3)
        print(f"  PC detection quality:  P={qual_proactive['precision']:.3f} "
              f"R={qual_proactive['recall']:.3f} F1={qual_proactive['f1']:.3f}")
        print(f"  KS detection quality:  P={qual_reactive['precision']:.3f} "
              f"R={qual_reactive['recall']:.3f} F1={qual_reactive['f1']:.3f}")

        # Extract time anchors for each drift event
        events = []
        for dp in drift_points:
            event = extract_time_anchors(
                results_proactive, results_reactive, dp, windows)
            event.description = f"{ds_name}_drift@W{dp}"
            events.append(event)

            t_a = event.T_alert if event.T_alert >= 0 else 'N/D'
            t_r = event.T_react if event.T_react >= 0 else 'N/D'
            print(f"    Drift @W{dp}: T_alert={t_a}, T_react={t_r}, "
                  f"ΔT={event.opportunity_window}, "
                  f"F1_drop(PC)={event.max_f1_drop_proactive:.4f}, "
                  f"F1_drop(KS)={event.max_f1_drop_reactive:.4f}, "
                  f"Recovery(PC)={event.recovery_windows_proactive}, "
                  f"Recovery(KS)={event.recovery_windows_reactive}")

        # Compute cumulative F1 loss area around drift points
        f1_loss_area = compute_f1_loss_area(
            f1_proactive, f1_reactive, drift_points, lookahead=15)

        # Summary
        summary = compute_phase2_summary(events, 'IDA-SPADE')
        summary['n_true_drifts'] = len(drift_points)

        all_phase2[ds_name] = {
            'events': events,
            'detection_quality': {
                'IDA-SPADE (PC)': qual_proactive,
                'Reactive (KS)': qual_reactive,
            },
            'proactive_f1': f1_proactive,
            'reactive_f1': f1_reactive,
            'true_drift_windows': drift_points,
            'summary': summary,
            'f1_loss_area': f1_loss_area,
        }

    export.export_phase2(all_phase2, output_dir)
    export.export_f1_loss_area(all_phase2, output_dir)
    print(f"\nPhase 2 results saved to {output_dir}")
    return all_phase2


# =====================================================================
# Phase 3: Ablation Study (Paper Tab. 3)
# =====================================================================

def run_phase3(datasets, output_dir, phase2_events=None):
    """Phase 3: Ablation study — IDA-SPADE variants."""
    print("\n" + "=" * 80)
    print("PHASE 3: Ablation Study (Paper Tab. 3)")
    print("=" * 80)

    all_ablation = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        X_init, y_init, stream, feature_cols = load_dataset(ds_name)
        windows = _collect_stream(stream)

        models = create_ablation_models(
            feature_cols=feature_cols, dataset_name=ds_name)
        ds_ablation = {}

        for variant_name, model in models.items():
            set_seed()
            print(f"  {variant_name}...")
            results = prequential_evaluate(
                model, iter(windows), X_init, y_init, verbose=False)
            agg = aggregate_metrics(results)
            print(f"    F1: {agg.get('f1', 0)*100:.2f}% | "
                  f"Acc: {agg.get('accuracy', 0)*100:.2f}%")
            ds_ablation[variant_name] = results

        all_ablation[ds_name] = ds_ablation

    export.export_phase4_ablation(all_ablation, output_dir)
    print(f"\nPhase 3 results saved to {output_dir}")
    return all_ablation


def run_multi_seed_phase3(datasets, output_dir, seeds):
    """Run Phase 3 (ablation) across multiple seeds and compute statistics."""
    print("\n" + "=" * 80)
    print(f"MULTI-SEED Phase 3: {len(seeds)} seeds = {seeds}")
    print("=" * 80)

    all_seed_results = {ds: {} for ds in datasets}
    per_seed_raw = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

        for ds_name in datasets:
            X_init, y_init, stream, feature_cols = load_dataset(ds_name)
            windows = _collect_stream(stream)
            models = create_ablation_models(feature_cols=feature_cols, dataset_name=ds_name)

            for variant_name, model in models.items():
                set_seed(seed)
                results = prequential_evaluate(
                    model, iter(windows), X_init, y_init, verbose=False)
                agg = aggregate_metrics(results)

                if variant_name not in all_seed_results[ds_name]:
                    all_seed_results[ds_name][variant_name] = {
                        'f1': [], 'accuracy': [], 'precision': [], 'recall': [],
                    }
                for k in ['f1', 'accuracy', 'precision', 'recall']:
                    all_seed_results[ds_name][variant_name][k].append(agg.get(k, 0))

                print(f"    [{ds_name}] {variant_name}: F1={agg.get('f1',0)*100:.2f}%")

                if seed_idx == 0:
                    if ds_name not in per_seed_raw:
                        per_seed_raw[ds_name] = {}
                    per_seed_raw[ds_name][variant_name] = results

    # Compute statistics
    summary = {}
    for ds_name in datasets:
        summary[ds_name] = {}
        for variant_name, metrics_dict in all_seed_results[ds_name].items():
            variant_summary = {}
            for metric, values in metrics_dict.items():
                variant_summary[f'{metric}_mean'] = float(np.mean(values))
                variant_summary[f'{metric}_std'] = float(np.std(values))
                variant_summary[f'{metric}_values'] = values
            summary[ds_name][variant_name] = variant_summary

    # Paired t-test: Full vs each ablation variant
    stat_tests = {}
    for ds_name in datasets:
        ds_tests = {}
        full_f1s = all_seed_results[ds_name].get('IDA-SPADE', {}).get('f1', [])
        if len(full_f1s) >= 3:
            for variant_name in ['IDA-SPADE-Reactive', 'IDA-SPADE-Global', 'IDA-SPADE-Statistical']:
                other_f1s = all_seed_results[ds_name].get(variant_name, {}).get('f1', [])
                if len(other_f1s) == len(full_f1s):
                    try:
                        t_stat, t_p = scipy_stats.ttest_rel(full_f1s, other_f1s)
                        ds_tests[variant_name] = {
                            'ttest_stat': float(t_stat), 'ttest_p': float(t_p),
                            'full_mean': float(np.mean(full_f1s)),
                            'full_std': float(np.std(full_f1s)),
                            'variant_mean': float(np.mean(other_f1s)),
                            'variant_std': float(np.std(other_f1s)),
                            'diff_mean': float(np.mean(np.array(full_f1s) - np.array(other_f1s))),
                        }
                    except Exception as e:
                        ds_tests[variant_name] = {'error': str(e)}
        stat_tests[ds_name] = ds_tests

    export.save_json(summary, os.path.join(output_dir, 'phase3_multi_seed_summary.json'))
    export.save_json(stat_tests, os.path.join(output_dir, 'phase3_statistical_tests.json'))
    if per_seed_raw:
        export.export_phase4_ablation(per_seed_raw, output_dir)
    export.export_multi_seed_table(summary, stat_tests, output_dir, 'phase3')

    print(f"\nMulti-seed Phase 3 results saved to {output_dir}")
    return summary, stat_tests


# =====================================================================
# Phase 4: Computation Efficiency (Paper Tab. 4)
# =====================================================================

def run_phase4(datasets, output_dir):
    """Phase 4: Component timing breakdown."""
    print("\n" + "=" * 80)
    print("PHASE 4: Computation Efficiency (Paper Tab. 4)")
    print("=" * 80)

    all_timing = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        X_init, y_init, stream, feature_cols = load_dataset(ds_name)
        windows = _collect_stream(stream)

        ds_timing = {}
        for variant_name in ['IDA-SPADE', 'IDA-SPADE-Reactive']:
            set_seed()
            model = (IDASpadeStreaming(
                        name=variant_name, feature_cols=feature_cols,
                        dataset_name=ds_name)
                     if variant_name == 'IDA-SPADE'
                     else IDASpadeReactive(
                        feature_cols=feature_cols, dataset_name=ds_name))
            model.initialize(X_init, y_init)
            if windows:
                timing = profile_components(
                    model, windows[0].X, windows[0].y_binary, n_rounds=10)
                ds_timing[variant_name] = timing
                print(f"  {variant_name}: {timing}")

        all_timing[ds_name] = ds_timing

    export.export_phase4_timing(all_timing, output_dir)

    # T-scaling
    print("\n  Running T-scaling...")
    t_scaling_results = []
    for T in T_SCALING_VALUES:
        set_seed()
        ds_name = datasets[0]
        X_init, y_init, stream, feature_cols = load_dataset(
            ds_name, window_size=T, slide_size=T)
        windows_t = _collect_stream(stream)
        model = IDASpadeStreaming(
            name='IDA-SPADE', feature_cols=feature_cols, dataset_name=ds_name)
        model.initialize(X_init, y_init)

        t0 = time.perf_counter()
        if windows_t:
            for w in windows_t[:20]:
                model.predict(w.X)
                model.detect_drift(w.X)
                model.update(w.X, w.y_binary)
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / min(20, len(windows_t)) if windows_t else 0
        throughput = T / (avg_ms / 1000) if avg_ms > 0 else 0

        t_scaling_results.append({
            'T': T, 'total_time_ms': avg_ms, 'throughput': throughput
        })
        print(f"    T={T}: {avg_ms:.1f}ms/window, {throughput:.0f} conn/s")

    export.export_phase4_scaling([], t_scaling_results, output_dir)
    print(f"\nPhase 4 results saved to {output_dir}")
    return all_timing


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='IDA-SPADE Experiment Runner')
    parser.add_argument('--datasets', nargs='+', default=['NSL-KDD', 'UNSW-NB15'],
                        choices=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017'])
    parser.add_argument('--phase', type=int, default=0,
                        help='Run specific phase (1-4), or 0 for all')
    parser.add_argument('--output', type=str, default=RESULTS_DIR,
                        help='Output directory for results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='Random seeds for multi-seed runs (e.g., --seeds 42 123 456 789 1024)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    set_seed()

    print(f"IDA-SPADE Experiment Framework (Paper-Aligned)")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output}")
    print(f"Phase: {'All' if args.phase == 0 else args.phase}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    t_start = time.time()

    seeds = args.seeds
    multi_seed = len(seeds) > 1

    if args.phase == 0 or args.phase == 1:
        if multi_seed:
            phase1_summary, phase1_tests, phase1_results = run_multi_seed_phase1(
                args.datasets, args.output, seeds)
        else:
            phase1_results = run_phase1(args.datasets, args.output)

    if args.phase == 0 or args.phase == 2:
        run_phase2(args.datasets, args.output)

    if args.phase == 0 or args.phase == 3:
        if multi_seed:
            run_multi_seed_phase3(args.datasets, args.output, seeds)
        else:
            run_phase3(args.datasets, args.output)

    if args.phase == 0 or args.phase == 4:
        run_phase4(args.datasets, args.output)

    total_time = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"All experiments completed in {total_time:.1f}s")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

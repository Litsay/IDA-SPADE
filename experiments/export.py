"""Export experiment results to markdown tables matching the data templates."""
import os
import json
import numpy as np
from typing import Dict, List

from .evaluator import PrequentialMetrics, aggregate_metrics
from .drift_injection import DriftEvent
from .config import RESULTS_DIR, FADING_FACTOR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# =====================================================================
# Phase 1: Prequential Baseline
# =====================================================================

def export_phase1_per_window(all_results: Dict[str, Dict[str, List[PrequentialMetrics]]],
                             output_dir: str):
    """Export per-window F1 tables (Tables 1.1-1.3).

    Args:
        all_results: {dataset_name: {model_name: [PrequentialMetrics]}}
    """
    ensure_dir(output_dir)
    model_order = ['IDA-SPADE', 'SSF', 'AOC-IDS', 'EWC', 'LwF', 'CIDS', 'CARD', 'FeCo', 'unFlowS']

    for ds_name, models_data in all_results.items():
        lines = [f"# Phase 1: Per-window Prequential F1 — {ds_name}\n"]

        # Find max window count
        max_windows = max(len(v) for v in models_data.values()) if models_data else 0

        # Header
        header = "| Window | " + " | ".join(model_order) + " |"
        sep = "|---|" + "|".join(["---"] * len(model_order)) + "|"
        lines.extend([header, sep])

        for k in range(max_windows):
            row = [str(k)]
            for m in model_order:
                if m in models_data and k < len(models_data[m]):
                    row.append(f"{models_data[m][k].f1:.4f}")
                else:
                    row.append("")
            lines.append("| " + " | ".join(row) + " |")

        filepath = os.path.join(output_dir, f"phase1_per_window_{ds_name.replace('-', '_')}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    # Also save as JSON for programmatic access
    json_data = {}
    for ds_name, models_data in all_results.items():
        json_data[ds_name] = {}
        for model_name, results in models_data.items():
            json_data[ds_name][model_name] = [
                {'window': r.window_index, 'f1': r.f1, 'acc': r.accuracy,
                 'prec': r.precision, 'rec': r.recall, 'fading_f1': r.fading_f1}
                for r in results
            ]
    save_json(json_data, os.path.join(output_dir, 'phase1_per_window.json'))


def export_phase1_aggregate(all_results: Dict[str, Dict[str, List[PrequentialMetrics]]],
                            output_dir: str):
    """Export aggregated metrics table (Table 1.4)."""
    ensure_dir(output_dir)
    model_order = ['EWC', 'LwF', 'CIDS', 'CARD', 'FeCo', 'unFlowS', 'AOC-IDS', 'SSF', 'IDA-SPADE']

    lines = ["# Phase 1: Prequential Aggregated Performance\n"]

    for ds_name, models_data in all_results.items():
        lines.append(f"\n### {ds_name}\n")
        lines.append("| NID Method | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | F1 Std. |")
        lines.append("|---|---|---|---|---|---|")

        for m in model_order:
            if m in models_data:
                agg = aggregate_metrics(models_data[m])
                lines.append(
                    f"| {m} | {agg['accuracy']*100:.2f} | {agg['precision']*100:.2f} | "
                    f"{agg['recall']*100:.2f} | {agg['f1']*100:.2f} | {agg['f1_std']:.4f} |"
                )
            else:
                lines.append(f"| {m} |  |  |  |  |  |")

    filepath = os.path.join(output_dir, 'phase1_aggregate.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =====================================================================
# Phase 2: Adversarial Opportunity Window
# =====================================================================

def export_phase2(phase2_data: Dict, output_dir: str):
    """Export Phase 2 results: drift events, detection quality, F1 time series."""
    ensure_dir(output_dir)
    lines = ["# Phase 2: Concept Drift Perception\n"]

    for ds_name, ds_data in phase2_data.items():
        lines.append(f"\n## {ds_name}\n")

        # Drift events table
        events = ds_data.get('events', [])
        lines.append("### Drift Events (Opportunity Window)\n")
        lines.append("| # | T_true | AR_before | AR_after | T_alert(PC) | T_react(KS) | ΔT | F1_drop(PC) | F1_drop(KS) | Recovery(PC) | Recovery(KS) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, e in enumerate(events):
            t_alert = str(e.T_alert) if e.T_alert >= 0 else 'N/D'
            t_react = str(e.T_react) if e.T_react >= 0 else 'N/D'
            rec_p = str(e.recovery_windows_proactive) if e.recovery_windows_proactive >= 0 else 'N/R'
            rec_r = str(e.recovery_windows_reactive) if e.recovery_windows_reactive >= 0 else 'N/R'
            lines.append(
                f"| {i+1} | {e.T_true} | {e.anomaly_ratio_before:.2f} | {e.anomaly_ratio_after:.2f} | "
                f"{t_alert} | {t_react} | {e.opportunity_window} | "
                f"{e.max_f1_drop_proactive:.4f} | {e.max_f1_drop_reactive:.4f} | "
                f"{rec_p} | {rec_r} |")

        # Detection quality table
        lines.append("\n### Drift Detection Quality\n")
        lines.append("| Method | Detected | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for method, qual in ds_data.get('detection_quality', {}).items():
            lines.append(
                f"| {method} | {qual['n_detected']} | {qual['tp']} | {qual['fp']} | {qual['fn']} | "
                f"{qual['precision']:.3f} | {qual['recall']:.3f} | {qual['f1']:.3f} |")

        # Summary
        lines.append(f"\n### Summary")
        summ = ds_data.get('summary', {})
        if summ:
            lines.append(f"- Ground truth drift points: {summ.get('n_true_drifts', 0)}")
            lines.append(f"- Mean ΔT (opportunity window): {summ.get('avg_opportunity_window', 0):.1f} windows")
            lines.append(f"- Mean F1 drop — PC: {summ.get('avg_f1_drop_proactive', 0):.4f}, KS: {summ.get('avg_f1_drop_reactive', 0):.4f}")
            lines.append(f"- Mean recovery — PC: {summ.get('avg_recovery_proactive', 'N/A')}, KS: {summ.get('avg_recovery_reactive', 'N/A')}")

    filepath = os.path.join(output_dir, 'phase2_drift_perception.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Also save per-window F1 as JSON for plotting
    f1_data = {}
    for ds_name, ds_data in phase2_data.items():
        f1_data[ds_name] = {
            'proactive_f1': ds_data.get('proactive_f1', []),
            'reactive_f1': ds_data.get('reactive_f1', []),
            'true_drift_windows': ds_data.get('true_drift_windows', []),
        }
    save_json(f1_data, os.path.join(output_dir, 'phase2_f1_timeseries.json'))


# =====================================================================
# Phase 3: Knowledge Retention
# =====================================================================

def export_phase3(retention_data: Dict[str, Dict], output_dir: str):
    """Export Phase 3 per-class recall tables."""
    ensure_dir(output_dir)
    lines = ["# Phase 3: Knowledge Retention\n"]

    for ds_name, data in retention_data.items():
        lines.append(f"\n## {ds_name}\n")
        classes = list(data.get('summary', {}).keys())
        if not classes:
            continue

        lines.append("| Attack Class | Mean Recall | Std Recall | Min Recall | N Windows |")
        lines.append("|---|---|---|---|---|")
        for cls in classes:
            s = data['summary'][cls]
            lines.append(
                f"| {cls} | {s['mean_recall']:.4f} | {s['std_recall']:.4f} | "
                f"{s['min_recall']:.4f} | {s['n_windows']} |"
            )

    filepath = os.path.join(output_dir, 'phase3_knowledge_retention.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =====================================================================
# Phase 4: Ablation & Compute-Benefit
# =====================================================================

def export_phase4_ablation(ablation_results: Dict[str, Dict[str, List[PrequentialMetrics]]],
                           output_dir: str):
    """Export Phase 4 ablation tables."""
    ensure_dir(output_dir)
    variant_order = ['IDA-SPADE', 'IDA-SPADE-Reactive', 'IDA-SPADE-Global', 'IDA-SPADE-Statistical']

    lines = ["# Phase 4: Ablation Study\n"]

    for ds_name, variants_data in ablation_results.items():
        lines.append(f"\n### {ds_name}\n")
        lines.append("| Ablation Variant | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | F1 Std. |")
        lines.append("|---|---|---|---|---|---|")

        for v in variant_order:
            if v in variants_data:
                agg = aggregate_metrics(variants_data[v])
                lines.append(
                    f"| {v} | {agg['accuracy']*100:.2f} | {agg['precision']*100:.2f} | "
                    f"{agg['recall']*100:.2f} | {agg['f1']*100:.2f} | {agg['f1_std']:.4f} |"
                )

    filepath = os.path.join(output_dir, 'phase4_ablation.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_phase4_timing(timing_data: Dict[str, Dict], output_dir: str):
    """Export Phase 4 timing breakdown tables."""
    ensure_dir(output_dir)
    lines = ["# Phase 4: Component Timing Breakdown\n"]

    for ds_name, models_timing in timing_data.items():
        lines.append(f"\n## {ds_name}\n")
        for model_name, components in models_timing.items():
            lines.append(f"\n### {model_name}\n")
            lines.append("| Component | Mean (ms) | Std (ms) |")
            lines.append("|---|---|---|")
            for comp, vals in components.items():
                lines.append(f"| {comp} | {vals['mean_ms']:.2f} | {vals['std_ms']:.2f} |")

    filepath = os.path.join(output_dir, 'phase4_timing.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_phase4_scaling(v_scaling: Dict, t_scaling: Dict, output_dir: str):
    """Export V-scaling and T-scaling experiment results."""
    ensure_dir(output_dir)
    lines = ["# Phase 4: Scaling Experiments\n"]

    if v_scaling:
        lines.append("\n## V-Scaling (Entity Count)\n")
        lines.append("| |V| | PC Time (ms) | Total Time (ms) | O(V²) Ratio |")
        lines.append("|---|---|---|---|")
        for entry in v_scaling:
            lines.append(
                f"| {entry['V']} | {entry['pc_time_ms']:.2f} | "
                f"{entry['total_time_ms']:.2f} | {entry['v2_ratio']:.2f} |"
            )

    if t_scaling:
        lines.append("\n## T-Scaling (Window Size)\n")
        lines.append("| T | Total Time (ms) | Throughput (conn/s) |")
        lines.append("|---|---|---|")
        for entry in t_scaling:
            lines.append(
                f"| {entry['T']} | {entry['total_time_ms']:.2f} | {entry['throughput']:.1f} |"
            )

    filepath = os.path.join(output_dir, 'phase4_scaling.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_multi_seed_table(summary: Dict, stat_tests: Dict, output_dir: str, phase: str):
    """Export multi-seed results as markdown table with mean±std and p-values."""
    ensure_dir(output_dir)
    lines = [f"# {phase.upper()}: Multi-Seed Statistical Summary\n"]

    for ds_name, models_data in summary.items():
        lines.append(f"\n## {ds_name}\n")

        if phase == 'phase1':
            model_order = ['EWC', 'LwF', 'CIDS', 'CARD', 'FeCo', 'unFlowS',
                          'AOC-IDS', 'SSF', 'IDA-SPADE']
            lines.append("| NID Method | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | p-value (vs IDA-SPADE) |")
            lines.append("|---|---|---|---|---|---|")
        elif phase == 'phase3':
            model_order = ['IDA-SPADE', 'IDA-SPADE-Reactive', 'IDA-SPADE-Global', 'IDA-SPADE-Statistical']
            lines.append("| Ablation Variant | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | p-value (vs Full) |")
            lines.append("|---|---|---|---|---|---|")
        else:
            model_order = list(models_data.keys())
            lines.append("| Method | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | p-value |")
            lines.append("|---|---|---|---|---|---|")

        for m in model_order:
            if m not in models_data:
                continue
            d = models_data[m]
            acc = f"{d.get('accuracy_mean', 0)*100:.2f}±{d.get('accuracy_std', 0)*100:.2f}"
            pre = f"{d.get('precision_mean', 0)*100:.2f}±{d.get('precision_std', 0)*100:.2f}"
            rec = f"{d.get('recall_mean', 0)*100:.2f}±{d.get('recall_std', 0)*100:.2f}"
            f1 = f"{d.get('f1_mean', 0)*100:.2f}±{d.get('f1_std', 0)*100:.2f}"

            # Get p-value
            ds_tests = stat_tests.get(ds_name, {})
            if phase == 'phase1' and m != 'IDA-SPADE':
                test = ds_tests.get(m, {})
                p = test.get('wilcoxon_p', test.get('ttest_p', None))
                p_str = f"{p:.4f}" if p is not None else "—"
            elif phase == 'phase3' and m != 'IDA-SPADE':
                test = ds_tests.get(m, {})
                p = test.get('ttest_p', None)
                p_str = f"{p:.4f}" if p is not None else "—"
            else:
                p_str = "—"

            lines.append(f"| {m} | {acc} | {pre} | {rec} | {f1} | {p_str} |")

    filepath = os.path.join(output_dir, f'{phase}_multi_seed_table.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_f1_loss_area(phase2_data: Dict, output_dir: str):
    """Export F1 loss area comparison for PC vs Reactive methods."""
    ensure_dir(output_dir)
    lines = ["# F1 Loss Area Comparison (PC-DriftForecasting vs Reactive)\n"]
    lines.append("Lower loss area = faster recovery from drift = better.\n")

    for ds_name, ds_data in phase2_data.items():
        lines.append(f"\n## {ds_name}\n")
        f1_loss = ds_data.get('f1_loss_area', [])
        if not f1_loss:
            lines.append("No F1 loss area data available.\n")
            continue

        lines.append("| Drift Point | Method | Baseline F1 | Loss Area | Windows |")
        lines.append("|---|---|---|---|---|")

        total_pc = 0.0
        total_ks = 0.0
        for entry in f1_loss:
            lines.append(
                f"| W{entry['drift_point']} | {entry['method']} | "
                f"{entry['baseline_f1']:.4f} | {entry['loss_area']:.4f} | "
                f"{entry['n_windows']} |")
            if entry['method'] == 'PC':
                total_pc += entry['loss_area']
            else:
                total_ks += entry['loss_area']

        lines.append(f"\n**Total F1 Loss Area**: PC = {total_pc:.4f}, KS = {total_ks:.4f}")
        if total_ks > 0:
            reduction = (1 - total_pc / total_ks) * 100
            lines.append(f"  → PC reduces cumulative F1 loss by {reduction:.1f}%")

    filepath = os.path.join(output_dir, 'phase2_f1_loss_area.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Also save as JSON
    json_data = {}
    for ds_name, ds_data in phase2_data.items():
        json_data[ds_name] = ds_data.get('f1_loss_area', [])
    save_json(json_data, os.path.join(output_dir, 'phase2_f1_loss_area.json'))

"""T2.2: Random Alert baseline for Tab.2.

Uniformly samples N alert windows (N=64 for UNSW-NB15, 175 for CIC-IDS-2017)
from the available window range, with ground-truth tolerance ±3, and reports
mean ± std of recall / precision / detection-F1 over 1000 repeats.

Output: experiment_results/t22_random_alert.json
"""
import json
import os
import numpy as np


def evaluate_alerts(alert_windows, gt, tolerance):
    """Two metric flavors for alert quality vs ground-truth events.

    Tab.2 paper convention: precision = caught_gt / n_alerts (event-level)
    Standard alert-level: precision = tp_alerts / n_alerts
    Recall is identical under both: caught_gt / n_gt.
    """
    gt_set = set(gt)
    alert_set = set(alert_windows)

    # Alert is TP if it falls within ±tolerance of any GT
    tp_alerts = sum(1 for a in alert_set
                    if any(abs(a - g) <= tolerance for g in gt_set))
    # GT is "caught" if any alert falls within ±tolerance
    caught_gt = sum(1 for g in gt_set
                    if any(abs(a - g) <= tolerance for a in alert_set))

    recall = caught_gt / max(len(gt_set), 1)
    n_alerts = max(len(alert_set), 1)

    # Paper convention (matches Tab.2): caught_gt / n_alerts
    precision_paper = caught_gt / n_alerts
    f1_paper = 2 * recall * precision_paper / max(recall + precision_paper, 1e-9)

    # Standard alert-level
    precision_std = tp_alerts / n_alerts
    f1_std = 2 * recall * precision_std / max(recall + precision_std, 1e-9)

    return {
        'recall': recall,
        'precision_paper': precision_paper,
        'f1_paper': f1_paper,
        'precision_std': precision_std,
        'f1_std': f1_std,
        'caught_gt': caught_gt,
        'tp_alerts': tp_alerts,
    }


def main():
    with open('experiment_results/tab2_unified_detection.json') as f:
        tab2 = json.load(f)

    out = {}
    rng = np.random.default_rng(42)

    for ds_name in ['UNSW-NB15', 'CIC-IDS-2017']:
        block = tab2[ds_name]
        gt = block['gt_drift_points']
        tolerance = block['tolerance']
        pc_alerts_idx = block['PC']['alert_windows']
        n_alerts_pc = len(pc_alerts_idx)
        # use max_alert + 1 as conservative total window estimate
        # to keep distributions comparable to actual sampling space
        total_windows = max(pc_alerts_idx) + 1
        # Allow alerts in any window from 0..total_windows-1
        valid_range = np.arange(total_windows)

        n_repeats = 1000
        recalls = []
        prec_paper, f1_paper = [], []
        prec_std,   f1_std   = [], []
        for _ in range(n_repeats):
            sampled = rng.choice(valid_range, size=n_alerts_pc, replace=False)
            m = evaluate_alerts(sampled.tolist(), gt, tolerance)
            recalls.append(m['recall'])
            prec_paper.append(m['precision_paper'])
            f1_paper.append(m['f1_paper'])
            prec_std.append(m['precision_std'])
            f1_std.append(m['f1_std'])

        # PC and KS event-level metrics from tab2 directly
        out[ds_name] = {
            'n_gt': len(gt),
            'tolerance': tolerance,
            'n_alerts_matched_to_PC': n_alerts_pc,
            'total_windows_search_space': int(total_windows),
            'n_repeats': n_repeats,
            'random_recall_mean': float(np.mean(recalls)),
            'random_recall_std':  float(np.std(recalls)),
            'random_precision_paper_mean': float(np.mean(prec_paper)),
            'random_precision_paper_std':  float(np.std(prec_paper)),
            'random_f1_paper_mean': float(np.mean(f1_paper)),
            'random_f1_paper_std':  float(np.std(f1_paper)),
            'random_precision_std_mean': float(np.mean(prec_std)),
            'random_precision_std_std':  float(np.std(prec_std)),
            'random_f1_std_mean': float(np.mean(f1_std)),
            'random_f1_std_std':  float(np.std(f1_std)),
            'PC_recall':       block['PC']['recall'],
            'PC_precision':    block['PC']['precision'],
            'PC_f1':           block['PC'].get('f1', None),
            'KS_recall':       block['KS']['recall'],
            'KS_precision':    block['KS']['precision'],
            'KS_f1':           block['KS'].get('f1', None),
        }
        print(f'\n=== {ds_name} ===')
        print(f'  GT events: {len(gt)}, tolerance: ±{tolerance}, total windows: {total_windows}')
        print(f'  N alerts (matched to PC): {n_alerts_pc}')
        print(f'  Random baseline (n_repeats={n_repeats}):')
        print(f'    recall:                  {np.mean(recalls):.3f} ± {np.std(recalls):.3f}')
        print(f'    precision_paper:         {np.mean(prec_paper):.3f} ± {np.std(prec_paper):.3f}')
        print(f'    F1_paper:                {np.mean(f1_paper):.3f} ± {np.std(f1_paper):.3f}')
        print(f'    precision_std:           {np.mean(prec_std):.3f} ± {np.std(prec_std):.3f}')
        print(f'    F1_std:                  {np.mean(f1_std):.3f} ± {np.std(f1_std):.3f}')
        # compute F1 from r,p
        def _f1(r, p): return 2*r*p / max(r+p, 1e-9)
        pc_f1 = _f1(block["PC"]["recall"], block["PC"]["precision"])
        ks_f1 = _f1(block["KS"]["recall"], block["KS"]["precision"])
        out[ds_name]['PC_f1'] = pc_f1
        out[ds_name]['KS_f1'] = ks_f1
        print(f'  PC (Tab.2): recall={block["PC"]["recall"]:.3f}, prec={block["PC"]["precision"]:.3f}, F1={pc_f1:.3f}')
        print(f'  KS (Tab.2): recall={block["KS"]["recall"]:.3f}, prec={block["KS"]["precision"]:.3f}, F1={ks_f1:.3f}')

    with open('experiment_results/t22_random_alert.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to experiment_results/t22_random_alert.json')


if __name__ == '__main__':
    main()

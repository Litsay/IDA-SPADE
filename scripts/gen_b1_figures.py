"""Regenerate Fig.3 (F1 trajectory + drift events) and Fig.4 (drift-period bars)
using B1 results.

Outputs (overwriting):
    figures/drift_period_comparison.pdf   (Fig.4)
    figures/Concept Drift Percept.pdf     (Fig.3)
    figures/Concept Drift Percept.png     (Fig.3 PNG copy for paper)

Run from the project root:
    /c/Users/Litsay/anaconda3/envs/CL/python.exe gen_b1_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

B1_TAB3_JSON = os.path.join(PROJECT_ROOT, 'IDA-SPADE', 'experiment_results',
                            'b1_tab3_drift_period.json')
LEGACY_TAB3_JSON = os.path.join(PROJECT_ROOT, 'IDA-SPADE', 'experiment_results',
                                'tab3_unified_drift_analysis.json')


# ---------------------------------------------------------------------------
# Fig.4 -- drift-period comparison (8 methods x 2 datasets)
# ---------------------------------------------------------------------------
def make_fig4(out_path):
    """Bar chart: Drift-Period F1 + Mean Recovery across 8 methods."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

    methods = ['IDA-SPADE', 'CIDS', 'CARD', 'SSF', 'EWC', 'LwF', 'FeCo', 'unFlowS']

    # B1-updated UNSW-NB15 values
    unsw_drift_f1 = [86.37, 72.96, 77.57, 71.47, 73.15, 76.27, 44.96, 26.44]
    unsw_recovery = [1.71,   3.00,  3.00,  4.57,  3.71,  4.86,  6.86,  6.43]

    # B1-updated CIC-IDS-2017 values
    cic_drift_f1  = [75.78, 66.31, 79.00, 72.19, 64.54, 63.96, 55.06, 11.22]
    cic_recovery  = [1.76,   2.93,  2.70,  2.67,  3.30,  2.54,  3.43,  2.46]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    x = np.arange(len(methods))
    width = 0.35
    c_unsw = '#4472C4'
    c_cic = '#ED7D31'

    # (a) Drift-Period F1
    bars1 = ax1.bar(x - width/2, unsw_drift_f1, width, label='UNSW-NB15',
                    color=c_unsw, edgecolor='black', linewidth=0.4)
    bars2 = ax1.bar(x + width/2, cic_drift_f1, width, label='CIC-IDS-2017',
                    color=c_cic, edgecolor='black', linewidth=0.4)
    ax1.set_ylabel('Drift-Period F1 (%)')
    ax1.set_title('(a) Drift-Period F1', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=40, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=unsw_drift_f1[0], color=c_unsw, linestyle='--',
                linewidth=0.6, alpha=0.5)
    ax1.axhline(y=cic_drift_f1[0], color=c_cic, linestyle='--',
                linewidth=0.6, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.4)
    for bar in bars1[:1]:
        bar.set_hatch('//')
    for bar in bars2[:1]:
        bar.set_hatch('//')

    # (b) Mean Recovery
    bars3 = ax2.bar(x - width/2, unsw_recovery, width, label='UNSW-NB15',
                    color=c_unsw, edgecolor='black', linewidth=0.4)
    bars4 = ax2.bar(x + width/2, cic_recovery, width, label='CIC-IDS-2017',
                    color=c_cic, edgecolor='black', linewidth=0.4)
    ax2.set_ylabel('Mean Recovery (windows)')
    ax2.set_title('(b) Mean Recovery Time', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=40, ha='right')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.4)
    for bar in bars3[:1]:
        bar.set_hatch('//')
    for bar in bars4[:1]:
        bar.set_hatch('//')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(out_path)
    plt.close()
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Fig.3 -- per-window F1 trajectory + drift events on UNSW-NB15
# ---------------------------------------------------------------------------
def make_fig3(out_path_pdf, out_path_png):
    """Two-row figure:
        (a) per-window F1 of B1 + reactive baseline + drift events
        (b) cumulative drift detections by PC and by KS-test
    """
    with open(B1_TAB3_JSON, encoding='utf-8') as f:
        b1_data = json.load(f)
    with open(LEGACY_TAB3_JSON, encoding='utf-8') as f:
        legacy = json.load(f)

    ds = 'UNSW-NB15'
    drift_points = b1_data[ds]['drift_points']
    f1_b1 = np.asarray(b1_data[ds]['f1_traces']['B1']) * 100
    f1_ssf = np.asarray(legacy[ds]['f1_traces'].get('SSF', [])) * 100
    n = len(f1_b1)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.16, 3.6),
                                     sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    c_b1 = '#1F4E79'   # IDA-SPADE color
    c_ssf = '#9DA9B5'  # baseline grey
    c_drift = '#C00000'  # drift event red

    windows = np.arange(n)
    ax_a.plot(windows, f1_b1, color=c_b1, linewidth=1.0,
              label='IDA-SPADE (per-window F1)')
    if len(f1_ssf):
        n_ssf = min(len(f1_ssf), n)
        ax_a.plot(np.arange(n_ssf), f1_ssf[:n_ssf], color=c_ssf,
                  linewidth=0.8, alpha=0.85, label='SSF (per-window F1)')

    # Drift events as vertical lines + markers on top axis
    for dp in drift_points:
        ax_a.axvline(x=dp, color=c_drift, linestyle=':', linewidth=0.5, alpha=0.6)
    ymin, ymax = ax_a.get_ylim()
    marker_y = ymax + 0.5
    for dp in drift_points:
        ax_a.plot(dp, marker_y, marker='v', color=c_drift, markersize=4,
                  clip_on=False)

    ax_a.set_ylabel('Per-window F1 (%)')
    ax_a.set_title(f'(a) Per-window detection F1 on UNSW-NB15  '
                   f'(red triangles = ground-truth drift events, '
                   f'n={len(drift_points)})', fontweight='bold')
    ax_a.set_ylim(0, 102)
    ax_a.grid(axis='y', alpha=0.3, linewidth=0.4)
    ax_a.legend(loc='lower right', framealpha=0.9)

    # Panel (b): drift events as a step function (cumulative count)
    cum = np.zeros(n)
    for dp in drift_points:
        if 0 <= dp < n:
            cum[dp:] += 1
    ax_b.fill_between(windows, 0, cum, color=c_drift, alpha=0.25, step='pre')
    ax_b.step(windows, cum, color=c_drift, linewidth=0.8, where='pre',
              label='Cumulative drift events')
    ax_b.set_xlabel('Window index')
    ax_b.set_ylabel('# drifts')
    ax_b.set_title('(b) Cumulative ground-truth drift events',
                   fontweight='bold')
    ax_b.set_ylim(0, max(cum.max() + 1, 2))
    ax_b.grid(axis='y', alpha=0.3, linewidth=0.4)
    ax_b.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout(h_pad=0.8)
    plt.savefig(out_path_pdf)
    plt.savefig(out_path_png)
    plt.close()
    print(f'  Saved: {out_path_pdf}')
    print(f'  Saved: {out_path_png}')


def main():
    print('=' * 60)
    print('Regenerating B1 figures')
    print('=' * 60)
    print('[Fig.4] drift-period comparison')
    make_fig4(os.path.join(FIGURES_DIR, 'drift_period_comparison.pdf'))
    print('[Fig.3] F1 trajectory + drift events')
    make_fig3(
        os.path.join(FIGURES_DIR, 'Concept Drift Percept.pdf'),
        os.path.join(FIGURES_DIR, 'Concept Drift Percept.png'),
    )
    print('Done. Run gen_b1_tsne.py for Fig.5 (separate; needs GPU).')


if __name__ == '__main__':
    main()

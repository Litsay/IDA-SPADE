"""Regenerate Fig.3 (Concept Drift Percept) with dual-dataset + PC vs KS
alert overlay.

Layout: 2 rows (UNSW-NB15, CIC-IDS-2017), each showing
    - B1 per-window F1 trajectory (line)
    - Ground-truth drift events (red triangle markers + dotted vertical lines)
    - PC alert events (B1's Drift_Alert) as orange dots
    - KS alert events (B1-Reactive's KS) as grey x-marks

Reads:
    IDA-SPADE/experiment_results/fig3_alert_data.json

Outputs (overwriting):
    figures/Concept Drift Percept.pdf
    figures/Concept Drift Percept.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_JSON = os.path.join(PROJECT_ROOT, 'IDA-SPADE', 'experiment_results',
                         'fig3_alert_data.json')


def rolling_mean(arr, k):
    """Centered rolling mean with reflection at edges."""
    arr = np.asarray(arr, dtype=float)
    if k <= 1 or len(arr) <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode='valid')


def panel(ax, label, ds_data, title, smooth_k=5):
    f1_raw = np.asarray(ds_data['B1']['f1_trace']) * 100
    f1 = rolling_mean(f1_raw, smooth_k)
    if len(f1) > len(f1_raw):
        f1 = f1[:len(f1_raw)]
    n = len(f1_raw)
    pc_alerts = ds_data['B1'].get('alerts', [])
    ks_alerts = ds_data['B1-Reactive'].get('alerts', [])
    drift_pts = ds_data.get('drift_points', [])

    c_b1 = '#1F4E79'
    c_drift = '#C00000'
    c_pc = '#ED7D31'
    c_ks = '#7F7F7F'

    windows = np.arange(n)
    # Light grey raw trace as background
    ax.plot(windows, f1_raw, color='#BFBFBF', linewidth=0.4, alpha=0.55,
            label='Raw per-window F1')
    # Bold smoothed line
    ax.plot(windows, f1, color=c_b1, linewidth=1.2,
            label=f'IDA-SPADE F1 ({smooth_k}-window smoothed)')

    # Ground-truth drift events: dotted vertical lines + top-edge red triangles
    for dp in drift_pts:
        if 0 <= dp < n:
            ax.axvline(x=dp, color=c_drift, linestyle=':',
                       linewidth=0.45, alpha=0.55)
    if drift_pts:
        ax.scatter(drift_pts, [104] * len(drift_pts),
                   marker='v', color=c_drift, s=20,
                   clip_on=False, zorder=5,
                   label=f'Real Drift (n={len(drift_pts)})')

    # PC alerts: orange dots in their own y-strip
    if pc_alerts:
        ax.scatter(pc_alerts, [70] * len(pc_alerts), marker='o',
                   color=c_pc, s=8, alpha=0.85, edgecolors='none',
                   label=f'PC Alert (n={len(pc_alerts)})')

    # KS alerts: grey x-marks slightly below PC
    if ks_alerts:
        ax.scatter(ks_alerts, [65] * len(ks_alerts), marker='x',
                   color=c_ks, s=12, linewidths=0.8, alpha=0.85,
                   label=f'KS Alert (n={len(ks_alerts)})')

    ax.set_ylim(60, 105)
    ax.set_ylabel('F1 Score (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.4)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2,
              fontsize=6, columnspacing=1.0, handletextpad=0.4,
              borderpad=0.3)


def main():
    with open(DATA_JSON, encoding='utf-8') as f:
        data = json.load(f)

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.0), sharey=True)

    if 'UNSW-NB15' in data:
        panel(ax1, 'UNSW-NB15', data['UNSW-NB15'],
              f'(a) UNSW-NB15 (per-window F1, '
              f'{len(data["UNSW-NB15"].get("drift_points", []))} ground-truth drift events)')
    if 'CIC-IDS-2017' in data:
        panel(ax2, 'CIC-IDS-2017', data['CIC-IDS-2017'],
              f'(b) CIC-IDS-2017 (per-window F1, '
              f'{len(data["CIC-IDS-2017"].get("drift_points", []))} ground-truth drift events)')

    ax2.set_xlabel('Window index')
    plt.tight_layout(h_pad=0.8)

    out_pdf = os.path.join(FIGURES_DIR, 'Concept Drift Percept.pdf')
    out_png = os.path.join(FIGURES_DIR, 'Concept Drift Percept.png')
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()

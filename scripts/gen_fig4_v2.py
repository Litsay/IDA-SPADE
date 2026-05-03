"""Regenerate Fig.4 (drift_period_comparison.pdf) with 3 panels:
    (a) Drift-Period F1
    (b) Stable-Period F1   <-- NEW: exposes the trick that CARD's high Drift F1
                              on CIC comes from a weak stable baseline.
    (c) Mean Recovery (windows)

Uses B1 numbers for IDA-SPADE; baselines unchanged.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')


def main():
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

    # B1 numbers (replacing IDA-SPADE row); baselines unchanged.
    unsw_drift  = [86.37, 72.96, 77.57, 71.47, 73.15, 76.27, 44.96, 26.44]
    unsw_stable = [97.29, 97.44, 84.19, 92.24, 89.07, 87.63, 51.96, 38.04]
    unsw_recov  = [1.71,  3.00,  3.00,  4.57,  3.71,  4.86,  6.86,  6.43]

    cic_drift   = [75.78, 66.31, 79.00, 72.19, 64.54, 63.96, 55.06, 11.22]
    cic_stable  = [87.90, 63.04, 72.82, 71.61, 53.10, 68.22, 27.84, 10.88]
    cic_recov   = [1.76,  2.93,  2.70,  2.67,  3.30,  2.54,  3.43,  2.46]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    ax_drift, ax_stable, ax_recov = axes

    x = np.arange(len(methods))
    width = 0.35
    c_unsw = '#4472C4'
    c_cic = '#ED7D31'

    def grouped_bars(ax, vals_unsw, vals_cic, ylabel, title, ylim=None,
                     legend_loc='upper right', hatch_first=True):
        b1 = ax.bar(x - width/2, vals_unsw, width, label='UNSW-NB15',
                    color=c_unsw, edgecolor='black', linewidth=0.4)
        b2 = ax.bar(x + width/2, vals_cic, width, label='CIC-IDS-2017',
                    color=c_cic, edgecolor='black', linewidth=0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha='right')
        ax.legend(loc=legend_loc, framealpha=0.9)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3, linewidth=0.4)
        if hatch_first:
            for bb in (b1[:1], b2[:1]):
                for bar in bb:
                    bar.set_hatch('//')
        # Reference dashed lines at IDA-SPADE level
        ax.axhline(y=vals_unsw[0], color=c_unsw, linestyle='--',
                   linewidth=0.6, alpha=0.5)
        ax.axhline(y=vals_cic[0], color=c_cic, linestyle='--',
                   linewidth=0.6, alpha=0.5)
        return b1, b2

    # (a) Drift F1
    grouped_bars(ax_drift, unsw_drift, cic_drift,
                 'Drift-Period F1 (%)', '(a) Drift-Period F1',
                 ylim=(0, 100), legend_loc='upper right')

    # (b) Stable F1 -- NEW
    grouped_bars(ax_stable, unsw_stable, cic_stable,
                 'Stable-Period F1 (%)', '(b) Stable-Period F1',
                 ylim=(0, 100), legend_loc='upper right')

    # (c) Mean Recovery
    grouped_bars(ax_recov, unsw_recov, cic_recov,
                 'Mean Recovery (windows)', '(c) Mean Recovery Time',
                 legend_loc='upper left')

    plt.tight_layout(w_pad=1.2)
    out = os.path.join(FIGURES_DIR, 'drift_period_comparison.pdf')
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()

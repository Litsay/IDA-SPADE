import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

unsw_drift_f1 = [86.24, 72.96, 77.57, 71.47, 73.15, 76.27, 44.96, 26.44]
unsw_recovery = [1.71, 3.00, 3.00, 4.57, 3.71, 4.86, 6.86, 6.43]

cic_drift_f1 = [77.35, 66.31, 79.00, 72.19, 64.54, 63.96, 55.06, 11.22]
cic_recovery = [1.93, 2.93, 2.70, 2.67, 3.30, 2.54, 3.43, 2.46]

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
ax1.axhline(y=unsw_drift_f1[0], color=c_unsw, linestyle='--', linewidth=0.6, alpha=0.5)
ax1.axhline(y=cic_drift_f1[0], color=c_cic, linestyle='--', linewidth=0.6, alpha=0.5)
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
outpath = 'figures/drift_period_comparison.pdf'
plt.savefig(outpath)
plt.close()
print(f'Saved: {outpath}')

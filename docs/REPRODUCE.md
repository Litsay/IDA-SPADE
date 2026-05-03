# Reproducing the IDA-SPADE Paper Results

This document maps every numerical claim in the IDA-SPADE paper to a
specific runner script in `scripts/`, and lists the seed lists, expected
runtimes, and output JSON / PDF locations.

All commands assume:

```bash
# from repository root
export PYTHONPATH=.
```

and a Python 3.10+ environment with the dependencies in `requirements.txt`
installed.

## Seed lists

| Use case | Seeds |
| --- | --- |
| Five-seed base set (all datasets, all variants) | `42, 123, 456, 789, 1024` |
| Ten-seed superset (NSL-KDD and UNSW-NB15 only) | base set ∪ `2048, 3333, 4096, 5555, 7777` |
| Tab.~V hyperparameter sensitivity | `42, 123, 456` |
| Tab.~III drift-period analysis | `42` (single-seed protocol) |
| Tab.~VI computational profiling | `42` (single-seed protocol, 20 windows) |

## Per-table / per-figure reproduction

### Table I — Streaming effectiveness (8 baselines + IDA-SPADE × 3 datasets)

```bash
python scripts/run_b1_tab1.py
```

Runs eight baselines plus IDA-SPADE under the prequential test-then-train
loop, with the canonical `1%` per-window expert-label budget and `20%`
bootstrap pool. Output: `experiment_results/b1_tab1_main.json`. Expected
walltime on a single workstation GPU: about 6–8 hours total. Resume is
supported (skips `(dataset, method, seed)` triples already in the JSON).

### Table II — Drift alert quality (PC vs reactive KS-test)

```bash
python scripts/run_b1_tab3.py    # produces tab2_unified_detection.json
```

Runs PC-DriftForecasting and the KS-test alert head on UNSW-NB15 and
CIC-IDS-2017 against the canonical anomaly-ratio change-point set
(threshold 0.10, minimum gap 8). Output:
`experiment_results/tab2_unified_detection.json` and
`experiment_results/tab3_unified_drift_analysis.json`.

### Table III — Drift-period analysis (8 methods × 2 datasets, single seed)

Same script as Table II:

```bash
python scripts/run_b1_tab3.py
```

Re-uses the canonical drift set from Table II to score per-method drift /
stable F1 and recovery time.

### Table IV — Seven-variant ablation (10 / 10 / 5 seeds × 3 datasets)

```bash
python scripts/run_b1_tab4.py
```

Variants: `B1` (Full), `B1-NoPCAlert`, `B1-Reactive`, `B1-Statistical`,
`B1-Global`, `B1-NoCL`, `B1-NoProto`, and the combined
`B1-Reactive-NoProto`. Output: `experiment_results/b1_tab4_ablation.json`.
Expected walltime: about 3–4 hours.

### Table V — Hyperparameter sensitivity (5 parameters, 3 seeds)

```bash
python scripts/run_b1_tab5.py        # alpha, sigma, K_hot
python scripts/run_b1_tab5_ext.py    # theta_LID, beta_hot, lambda_p
```

Output: `experiment_results/b1_tab5_sensitivity.json` and
`experiment_results/b1_tab5_ext.json`.

### Table VI — Computational profiling

```bash
python scripts/run_b1_tab6.py
```

Profiles ECBA, PC-DriftForecasting, NID inference, and incremental update
on each dataset over 20 windows at seed 42. Output:
`experiment_results/b1_tab6_runtime.json` and
`experiment_results/baseline_runtime_for_tab6.json`.

### Figure 3 — Per-window F1 and PC vs KS alerts on UNSW + CIC

```bash
python scripts/collect_fig3_data.py    # collect per-window F1 trace
python scripts/gen_fig3_v2.py          # render PDF/PNG
```

### Figure 4 — Drift-period comparison (Drift F1 / Recovery / Stable F1)

```bash
python scripts/gen_fig4_v2.py
```

Reads `experiment_results/tab3_unified_drift_analysis.json` and produces
`figures/drift_period_comparison.pdf`.

### Figure 5 — t-SNE visualization (UNSW-NB15)

```bash
python scripts/gen_b1_tsne.py
```

### FNN / mutual-information diagnostics for E and tau

```bash
python scripts/compute_fnn_mi.py
```

Reproduces the False-Nearest-Neighbor and auto-mutual-information curves
that justify the conservative `E=3, tau=1` defaults discussed in
Sec.~III-C.

### Tier 2 supplementary experiments

The Tier 2 experiments were used to defend the contrastive-head and
2×2 ablation arguments and to characterize the hot-mode duty cycle:

```bash
python scripts/run_t21_hot_mode_stats.py    # T2.1 hot-mode + reversal trace
python scripts/run_t22_random_alert.py       # T2.2 random-alert baseline (1000 reps)
python scripts/run_t24_repr_quality.py       # T2.4 silhouette / cluster purity
```

## Convenience: reproduce everything

```bash
python scripts/run_b1_all.py
```

Runs Tab.I, Tab.II/III, Tab.IV, Tab.V, Tab.VI back-to-back with resume
support. About 14–18 hours total on a single workstation GPU.

## Output layout

```
experiment_results/
├── b1_tab1_main.json
├── tab2_unified_detection.json
├── tab3_unified_drift_analysis.json
├── b1_tab4_ablation.json
├── b1_tab5_sensitivity.json
├── b1_tab5_ext.json
├── b1_tab6_runtime.json
├── baseline_runtime_for_tab6.json
├── t21_hot_mode_stats.json
├── t22_random_alert.json
└── t24_repr_quality.json
```

## Hardware note

All experiments in the paper were performed on a single workstation with one
NVIDIA GPU (CUDA 12.6, PyTorch 2.8, NumPy 1.26, pandas 2.3, River 0.21).
The Tab.~VI portion breakdown (per-component %) is architecture-invariant
across GPU models because the dominant cost is the per-window nearest
neighbor search inside the shadow-attractor reconstruction, which has the
same algorithmic profile on any modern accelerator.

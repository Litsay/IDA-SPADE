# IDA-SPADE

**Intrusion Detection via Anticipatory Spatio-Temporal Potential Causality
Analysis on Drift Adaptation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c)](https://pytorch.org/)

This repository is the official implementation of the paper

> *From Observation to Analysis: Network Intrusion Detection Based on
> Spatio-Temporal Potential Causality*, IEEE Transactions on Dependable and
> Secure Computing (TDSC), 2026.

IDA-SPADE replaces the dominant *reactive* (observe-respond) drift-adaptation
paradigm in continual-learning network intrusion detection (CL-NID) with a
*proactive* (anticipate-prepare) paradigm built on three components:

1. **ECBA** — Entity-Centric Behavioral Aggregation. Reshapes streaming
   connection records into per-entity temporal behavioral sequences using
   transient-preserving aggregators that retain anomalous-forcing signatures.
2. **PC-DriftForecasting** — Reconstructs shadow attractors from each
   entity's behavioral series via delay-coordinate embedding and forecasts
   concept drift from the instantaneous breakdown of historical
   cross-manifold mappings.
3. **Causally-coupled continual learning** — An alert-gated EMA prototype
   with reversal-aware fusion converts each drift forecast into a
   representation-side anticipatory update, complemented by a manifold-guided
   supervised contrastive regularizer that aligns representations with the
   PC-DriftForecasting coupling structure.

Headline results (prequential evaluation, see Tab.~I of the paper):

| Dataset | IDA-SPADE F1 | Best baseline | Margin |
| --- | --- | --- | --- |
| NSL-KDD (near stationary) | **98.58 ± 0.13** | CIDS 98.25 | +0.33 (p < 0.001) |
| UNSW-NB15 (moderate drift) | **94.40 ± 0.78** | CIDS 91.88 | +2.52 (p < 0.001) |
| CIC-IDS-2017 (high drift) | **83.69 ± 0.99** | CARD 75.10 | +8.59 (p < 0.005) |

## Quick start

```bash
# 1) clone and create the environment
git clone https://github.com/<TBD-USER>/IDA-SPADE.git
cd IDA-SPADE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) prepare the three benchmarks (see docs/DATASETS.md)
#    expected layout:
#    NSL_pre_data/{PKDDTrain+,PKDDTest+}.csv
#    UNSW_pre_data/{UNSWTrain,UNSWTest}.csv
#    CIC_pre_data/CIC-IDS-2017_full.csv

# 3) reproduce Table I (streaming effectiveness)
python scripts/run_b1_tab1.py

# 4) reproduce Table IV (seven-variant ablation including
#    the combined Reactive&NoProto cell)
python scripts/run_b1_tab4.py
```

The `scripts/` directory contains one runner per paper artifact; see
[docs/REPRODUCE.md](docs/REPRODUCE.md) for the complete map.

## Repository layout

```
IDA-SPADE-release/
├── src/
│   └── ida_spade_singlefile.py       # standalone single-file pipeline (CLI)
├── experiments/                       # streaming evaluation framework
│   ├── ida_spade_wrapper.py           # IDA-SPADE as StreamingModel
│   ├── ida_spade_b1.py                # ablation variants used in Tab.IV
│   ├── contrastive_modules.py         # manifold-guided SupCon loss
│   ├── streaming_interface.py         # StreamingModel ABC
│   ├── evaluator.py                   # prequential test-then-train loop
│   ├── data_loader.py                 # NSL/UNSW/CIC loaders
│   ├── drift_injection.py             # synthetic drift utilities
│   ├── knowledge_retention.py         # forgetting metrics
│   ├── ablation.py                    # ablation harness
│   ├── config.py                      # hyperparameters and dataset paths
│   ├── compare_settings.py            # cross-protocol settings checks
│   ├── export.py                      # JSON / CSV export helpers
│   ├── run_all.py                     # phase-by-phase orchestrator
│   ├── run_cic_full_eval.py           # CIC-only end-to-end runner
│   ├── run_cic_perday.py              # CIC per-day analysis
│   └── baselines/                     # streaming wrappers for baselines
│       ├── ssf_baseline.py            # Zhang INFOCOM 2025
│       ├── aoc_ids.py                 # Zhang INFOCOM 2024
│       ├── card_baseline.py           # Huang TDSC 2025
│       ├── cids_baseline.py           # Yue TNSM 2022
│       ├── feco_baseline.py           # Wang INFOCOM 2022
│       ├── unflows_baseline.py        # Yang TIFS 2025
│       ├── ewc_baseline.py            # Kirkpatrick PNAS 2017
│       ├── lwf_baseline.py            # Li TPAMI 2018
│       ├── feco_adwin.py              # FeCo + ADWIN drift handler
│       └── unflows_adwin.py           # unFlowS + ADWIN drift handler
├── scripts/                           # per-table / per-figure runners
│   ├── run_b1_tab1.py                 # Tab.I main metrics
│   ├── run_b1_tab3.py                 # Tab.II + Tab.III drift analysis
│   ├── run_b1_tab4.py                 # Tab.IV seven-variant ablation
│   ├── run_b1_tab5.py                 # Tab.V (alpha, sigma, K_hot)
│   ├── run_b1_tab5_ext.py             # Tab.V extension (theta_LID, beta_hot, lambda_p)
│   ├── run_b1_tab6.py                 # Tab.VI runtime profiling
│   ├── run_b1_all.py                  # one-shot reproduction
│   ├── compute_fnn_mi.py              # FNN / mutual-information diagnostics
│   ├── collect_fig3_data.py           # Fig.3 per-window F1 trace
│   ├── gen_fig3_v2.py                 # Fig.3 renderer
│   ├── gen_fig4_v2.py                 # Fig.4 drift-period comparison
│   ├── gen_b1_tsne.py                 # Fig.5 t-SNE
│   ├── gen_b1_figures.py              # auxiliary figure generators
│   ├── gen_drift_period_fig.py        # alternative drift-period plot
│   ├── run_t21_hot_mode_stats.py      # supplementary hot-mode duty cycle
│   ├── run_t22_random_alert.py        # supplementary random-alert baseline
│   └── run_t24_repr_quality.py        # supplementary silhouette / purity
├── configs/
│   └── canonical.json                 # the canonical hyperparameter file
├── docs/
│   ├── DATASETS.md                    # dataset download + preprocessing
│   └── REPRODUCE.md                   # per-table / per-figure replication
├── requirements.txt
├── LICENSE                            # MIT
└── README.md
```

## Configuration

The single canonical hyperparameter set
([`configs/canonical.json`](configs/canonical.json)) is fixed across
NSL-KDD, UNSW-NB15, and CIC-IDS-2017. UNSW-NB15 is the development stream
where Tab.~V was tuned; the resulting values transfer unchanged to NSL-KDD
and CIC-IDS-2017.

Key knobs (paper notation):

| Group | Symbol | Value |
| --- | --- | --- |
| ECBA | `T` (window) | 1000 connections, non-overlapping |
| PC-DriftForecasting | `E`, `tau` | 3, 1 |
| PC-DriftForecasting | `theta_LID`, `K_LID` | 15.0, 10 |
| PC-DriftForecasting | `sigma` | 1.0 |
| Backbone | MLP | [128, 64, 32], Adam @ 1e-3, dropout 0.2, batch 32 |
| EWC | `lambda_ewc` | 0.1 |
| SupCon | `alpha`, `tau_sc` | 0.15, 0.1 |
| Prototype | `beta` quiet/hot | 0.99 / 0.90 |
| Prototype | `K_hot` | 3 |
| Prototype | `lambda_p` quiet/reversal | 0.2 / 0.6 |
| Prototype | `theta_rev` | 0.3 |
| Backbone freeze | `N_freeze` | 15 consecutive quiet windows |

## Citing

If you use this code or build on the proactive drift-adaptation paradigm,
please cite:

```bibtex
@article{Li:TDSC-2026,
  title   = {From Observation to Analysis: Network Intrusion Detection Based on Spatio-Temporal Potential Causality},
  author  = {Siyu Li and Jin Yang},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026},
  note    = {To appear},
}
```

## License

This project is released under the [MIT License](LICENSE). Datasets are
distributed under their original licenses; please review and accept those
before downloading (see [docs/DATASETS.md](docs/DATASETS.md)).

## Acknowledgments

This work is supported by the National Natural Science Foundation of China
(Grants No.~61872254 and No.~62162057), the Key Lab of Information Network
Security of Ministry of Public Security (Grant No.~C20606), and the Sichuan
Science and Technology Program (Grant No.~2021JDRC0004).

The PC-DriftForecasting module builds on the Pattern Causality framework of
Stavroglou et al. (PNAS, 2020) and the convergent cross-mapping machinery of
Sugihara et al. (Science, 2012); see Sec.~II-B of the paper for the
theoretical background.

## Contact

Questions, issues, and reproduction reports are welcome via GitHub Issues, or
by email to `sy_lee_real@icloud.com` (corresponding author: Jin Yang,
`yj@scu.edu.cn`).

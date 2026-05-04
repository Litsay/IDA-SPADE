# IDA-SPADE

**Intrusion Detection via Anticipatory Spatio-Temporal Potential Causality
Analysis on Drift Adaptation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c)](https://pytorch.org/)

This repository contains the implementation and reproduction scripts
accompanying the manuscript

> *From Observation to Analysis: Network Intrusion Detection Based on
> Spatio-Temporal Potential Causality*

which is currently under submission to IEEE Transactions on Dependable and
Secure Computing (TDSC). The repository will be updated as the manuscript
is reviewed and revised.

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

## Main results

Prequential evaluation under a `1%` per-window expert-label budget and a
`20%` initial bootstrap pool. Numbers are mean ± std across 5 seeds (10 seeds
on NSL-KDD and UNSW-NB15 for IDA-SPADE).

| Dataset | IDA-SPADE F1 | Best baseline | Margin |
| --- | --- | --- | --- |
| NSL-KDD (near stationary) | **98.58 ± 0.13** | CIDS 98.25 | +0.33 (Welch's *p* < 0.001) |
| UNSW-NB15 (moderate drift) | **94.40 ± 0.78** | CIDS 91.88 | +2.52 (*p* < 0.001) |
| CIC-IDS-2017 (high drift) | **83.69 ± 0.99** | CARD 75.10 | +8.59 (*p* < 0.005) |

The advantage over the runner-up grows with dataset drift intensity, which
is the regime where the proactive paradigm of IDA-SPADE is designed to
matter.

## Quick start

```bash
# 1) clone and create the environment
git clone https://github.com/Litsay/IDA-SPADE.git
cd IDA-SPADE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) prepare the three benchmarks (see docs/DATASETS.md)
#    expected layout:
#    NSL_pre_data/{PKDDTrain+,PKDDTest+}.csv
#    UNSW_pre_data/{UNSWTrain,UNSWTest}.csv
#    CIC_pre_data/CIC-IDS-2017_full.csv

# 3) reproduce the main result tables one by one
export PYTHONPATH=.
python scripts/run_b1_tab1.py     # Tab.I  streaming effectiveness (8 baselines)
python scripts/run_b1_tab3.py     # Tab.II + Tab.III drift alert + drift-period
python scripts/run_b1_tab4.py     # Tab.IV  seven-variant ablation
python scripts/run_b1_tab5.py     # Tab.V   sensitivity (alpha, sigma, K_hot)
python scripts/run_b1_tab5_ext.py # Tab.V   sensitivity (theta_LID, beta_hot, lambda_p)
python scripts/run_b1_tab6.py     # Tab.VI  computational profiling

# or, all of the above in one shot (~14-18 hours on a single GPU):
python scripts/run_b1_all.py
```

## Repository layout

```
IDA-SPADE/
├── src/
│   └── ida_spade_singlefile.py       # standalone single-file CLI pipeline
├── experiments/                       # streaming evaluation framework
│   ├── ida_spade_wrapper.py           # IDA-SPADE as StreamingModel
│   ├── ida_spade_b1.py                # ablation variants used in Tab.IV
│   ├── contrastive_modules.py         # manifold-guided SupCon loss
│   ├── streaming_interface.py         # StreamingModel ABC
│   ├── evaluator.py                   # prequential test-then-train loop
│   ├── data_loader.py                 # NSL/UNSW/CIC loaders
│   ├── config.py                      # hyperparameters and dataset paths
│   ├── ablation.py / drift_injection.py / knowledge_retention.py / ...
│   └── baselines/                     # streaming wrappers for 10 baselines
│       ├── ssf_baseline.py            # Zhang INFOCOM 2025
│       ├── aoc_ids.py                 # Zhang INFOCOM 2024
│       ├── card_baseline.py           # Huang TDSC 2025
│       ├── cids_baseline.py           # Yue TNSM 2022
│       ├── feco_baseline.py           # Wang INFOCOM 2022
│       ├── unflows_baseline.py        # Yang TIFS 2025
│       ├── ewc_baseline.py            # Kirkpatrick PNAS 2017
│       └── lwf_baseline.py            # Li TPAMI 2018
├── scripts/                           # paper-table reproduction runners
│   ├── run_b1_tab1.py                 # Tab.I  main metrics
│   ├── run_b1_tab3.py                 # Tab.II + Tab.III drift analysis
│   ├── run_b1_tab4.py                 # Tab.IV seven-variant ablation
│   ├── run_b1_tab5.py                 # Tab.V  alpha / sigma / K_hot
│   ├── run_b1_tab5_ext.py             # Tab.V  theta_LID / beta_hot / lambda_p
│   ├── run_b1_tab6.py                 # Tab.VI runtime profiling
│   └── run_b1_all.py                  # one-shot reproduction
├── configs/
│   └── canonical.json                 # the canonical hyperparameter file
├── docs/
│   └── DATASETS.md                    # dataset download + preprocessing
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

## Hardware

All experiments in the paper were performed on a single workstation with one
NVIDIA GPU (CUDA 12.6, PyTorch 2.8, NumPy 1.26, pandas 2.3, River 0.21).

## License

This project is released under the [MIT License](LICENSE). Datasets are
distributed under their original licenses; please review and accept those
before downloading (see [docs/DATASETS.md](docs/DATASETS.md)).

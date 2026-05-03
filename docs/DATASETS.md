# Datasets

IDA-SPADE is evaluated on three public network intrusion detection (NID)
benchmarks chosen to span a near-stationary → moderately drift-prone → highly
drift-intensive gradient. The benchmarks themselves are NOT redistributed in
this repository because of their licenses and size; users must download them
from the official sources below and place the prepared CSV files into the
expected directory structure.

| Dataset | Drift intensity | Behavioral dim *M* | Source |
| --- | --- | --- | --- |
| NSL-KDD | near stationary | 45 | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | moderate | 56 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS-2017 | high (5-day full capture) | 36 | https://www.unb.ca/cic/datasets/ids-2017.html |

## Expected directory layout

After downloading and preprocessing, the working directory should contain:

```
IDA-SPADE-release/
├── NSL_pre_data/
│   ├── PKDDTrain+.csv     # processed train (one-hot label column "labels2")
│   └── PKDDTest+.csv
├── UNSW_pre_data/
│   ├── UNSWTrain.csv      # processed train (label column "label")
│   └── UNSWTest.csv
└── CIC_pre_data/
    └── CIC-IDS-2017_full.csv  # five-day merged capture, label column "Label"
```

Paths are configured in `experiments/config.py` (variables `NSL_DATA_PATH`,
`UNSW_DATA_PATH`, `CIC_DATA_PATH`); adjust them if your layout differs.

## Preprocessing

Each dataset goes through three steps before reaching the streaming pipeline.
Step 1 is the only step that differs per dataset; steps 2 and 3 are identical
across all three benchmarks and are performed automatically inside the streaming
loop.

### Step 1: Original-format → CSV with binary label

| Dataset | Action |
| --- | --- |
| NSL-KDD | Download `KDDTrain+.txt` and `KDDTest+.txt`. Add header from `Field Names.csv`. Re-encode the categorical columns `protocol_type`, `service`, `flag` with one-hot. Map the multi-class label to binary by collapsing all attack classes to `1`; keep `normal` as `0`. Save as `PKDDTrain+.csv` / `PKDDTest+.csv`. The label column should be named `labels2`. |
| UNSW-NB15 | Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`. Drop the `id` and `attack_cat` columns. Keep the binary `label` column. Save as `UNSWTrain.csv` / `UNSWTest.csv`. |
| CIC-IDS-2017 | Download the eight per-day CSVs from `MachineLearningCSV.zip`. Concatenate them in chronological order. Strip whitespace from column names. Map `BENIGN` to `0` and any non-`BENIGN` label to `1`; keep the resulting column named `Label`. Save the concatenated file as `CIC-IDS-2017_full.csv`. |

### Step 2: Feature standardization (automatic)

The streaming loop fits a per-feature `StandardScaler` on the initial-pool
bootstrap (the first `20%` of the prepared train split) and freezes it for the
rest of the stream. This is implemented in
`experiments/ida_spade_wrapper.py::_aggregate_and_prepare`.

### Step 3: Entity-centric behavioral aggregation (automatic)

After standardization, ECBA aggregates raw connection records into per-entity
behavioral vectors of dimension `M` per non-overlapping window of `T = 1000`
connections. The aggregation rules are dataset-specific and are stored in
`experiments/config.py::DATASET_PRESETS`. ECBA exclusively uses
transient-preserving statistics (max, local variance, Shannon entropy)
rather than smoothing operators, so abrupt payload-injection signatures are
preserved in the reconstructed shadow attractors.

Inside each window, the per-entity binary label is the OR of its constituent
connection labels (any attack connection makes the entity an attack).

## Drift events used in Tab.~II / Tab.~III of the paper

Ground-truth drift events are derived from the per-window attack ratio with a
three-window rolling smoothing, a five-window rolling baseline, a deviation
threshold of `0.10`, and a minimum gap of `8` windows between consecutive
events. This yields:

| Dataset | # events | Tolerance |
| --- | --- | --- |
| NSL-KDD | 0 (excluded from drift analysis) | n/a |
| UNSW-NB15 | 7 | ±3 windows |
| CIC-IDS-2017 | 46 | ±3 windows |

The exact event indices are stored in
`experiment_results/tab2_unified_detection.json` and reproduced from
`scripts/run_b1_tab3.py`.

## License

The three benchmarks are distributed under their original licenses; please
review and accept them before downloading. This repository contains no
benchmark data and only references the public download URLs above.

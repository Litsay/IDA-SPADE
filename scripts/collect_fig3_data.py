"""Collect F1 trace + alert events for B1 and B1-Reactive on UNSW + CIC.
Writes a single JSON for downstream Fig.3 plotting.

Output: experiment_results/fig3_alert_data.json

Usage:
    /c/Users/Litsay/anaconda3/envs/CL/python.exe collect_fig3_data.py
"""
import os
import sys
import json
import time
import gc

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


OUTPUT_JSON = os.path.join('experiment_results', 'fig3_alert_data.json')
SEED = 42


def cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect(model_class, dataset, label):
    """Run prequential, capture per-window F1 and alert windows."""
    from experiments.data_loader import load_dataset
    set_seed(SEED)
    print(f'[{label}/{dataset}] loading...')
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)
    print(f'[{label}/{dataset}] init={X_init.shape} windows={len(windows)}')

    model = model_class(feature_cols=fc, dataset_name=dataset)
    model.initialize(X_init, y_init)

    f1_trace = []
    alerts = []
    t0 = time.time()
    for w_idx, w in enumerate(windows):
        # test
        preds, lbls = model.predict_evaluate(w.X, w.y_binary)
        if len(preds) == 0 or len(lbls) == 0:
            f1 = 1.0
        else:
            f1 = float(f1_score(lbls, preds, zero_division=1))
        f1_trace.append(f1)
        # detect drift
        detected, _ = model.detect_drift(w.X)
        if detected:
            alerts.append(w_idx)
        # train
        model.update(w.X, w.y_binary)

    elapsed = time.time() - t0
    print(f'[{label}/{dataset}] done in {elapsed:.1f}s; alerts={len(alerts)} '
          f'final_F1_mean={np.mean(f1_trace):.4f}')
    cleanup(model, windows)
    return {'f1_trace': f1_trace, 'alerts': alerts}


def main():
    from experiments.ida_spade_b1 import IDASpadeB1, IDASpadeB1Reactive
    from experiments.drift_injection import identify_drift_points
    from experiments.data_loader import load_dataset

    os.makedirs('experiment_results', exist_ok=True)

    out = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, encoding='utf-8') as f:
                out = json.load(f)
        except Exception:
            out = {}

    runs = [
        ('UNSW-NB15', 'B1',          IDASpadeB1),
        ('UNSW-NB15', 'B1-Reactive', IDASpadeB1Reactive),
        ('CIC-IDS-2017', 'B1',          IDASpadeB1),
        ('CIC-IDS-2017', 'B1-Reactive', IDASpadeB1Reactive),
    ]

    for ds, label, cls in runs:
        if ds not in out:
            out[ds] = {}
        if 'drift_points' not in out[ds]:
            X_init, y_init, stream, _ = load_dataset(ds)
            windows = list(stream)
            dp = identify_drift_points(windows, threshold=0.10, min_gap=8)
            out[ds]['drift_points'] = list(dp)
            out[ds]['n_windows'] = len(windows)
            print(f'[{ds}] drift_points = {len(dp)}')
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2)
        if label in out[ds] and 'alerts' in out[ds][label]:
            print(f'[{ds}/{label}] SKIP (resumed)')
            continue
        rec = collect(cls, ds, label)
        out[ds][label] = rec
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

    print('=' * 60)
    print(f'All done. Output: {OUTPUT_JSON}')
    for ds in ['UNSW-NB15', 'CIC-IDS-2017']:
        dp = out[ds].get('drift_points', [])
        print(f'{ds}: {len(dp)} drift events')
        for label in ['B1', 'B1-Reactive']:
            rec = out[ds].get(label, {})
            print(f'  {label}: alerts={len(rec.get("alerts", []))}')


if __name__ == '__main__':
    main()

"""Run B1 Tab.6 per-window runtime profile.

Profiles ECBA, PC-DriftForecasting, NID inference, Incremental update for B1
across NSL-KDD, UNSW-NB15, CIC-IDS-2017 at T=1000, seed=42, 20 windows each.

Output JSON
    experiment_results/b1_tab6_runtime.json
    experiment_results/b1_tab6_runtime.log

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab6.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab6.py --smoke
"""
import sys
import os
import json
import argparse
import gc
import time
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


OUTPUT_DIR = 'experiment_results'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab6_runtime.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab6_runtime.log')

DATASETS = ['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017']
SEED = 42
N_PROFILE_WINDOWS = 20


def log(msg):
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def profile_b1_on(dataset):
    """Profile B1 per-window components on `dataset`. Returns dict of mean ms."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate
    from experiments.ida_spade_b1 import IDASpadeB1

    set_seed(SEED)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)

    # Use a small slice for profiling (mid-stream so model is warmed up)
    # Actually run the full prequential to populate state, then re-profile
    # 20 specific windows. To keep it simple and consistent with the existing
    # tab6 protocol, we use the first 20 windows post-init.
    profile_windows = windows[:N_PROFILE_WINDOWS]

    model = IDASpadeB1(feature_cols=fc, dataset_name=dataset)

    # Run prequential on profile windows, capturing timing
    timings = {
        'ecba_ms':       [],
        'drift_detection_ms': [],
        'inference_ms':  [],
        'training_ms':   [],
        'total_ms':      [],
    }

    # Init the model with X_init
    model.initialize(X_init, y_init)

    for w_idx, w in enumerate(profile_windows):
        X, y = w.X, w.y_binary
        # Predict
        t0 = time.perf_counter()
        _ = model.predict(X)
        infer_ms = model._last_timing.get('inference_ms', (time.perf_counter() - t0) * 1000)

        # Detect drift
        t0 = time.perf_counter()
        _ = model.detect_drift(X)
        drift_ms = model._last_timing.get('drift_detection_ms', (time.perf_counter() - t0) * 1000)

        # Update
        t0 = time.perf_counter()
        model.update(X, y)
        train_ms = model._last_timing.get('training_ms', (time.perf_counter() - t0) * 1000)

        ecba_ms = drift_ms  # ECBA happens inside detect_drift via _get_cached_ecba

        # Approximate component split (matches paper Tab.6 semantics):
        #   ECBA          : entity aggregation timing (cached)
        #   PC-DriftForecasting : drift detection minus ECBA
        #   NID inference : predict timing
        #   Incremental update : update timing
        # The existing wrapper bundles ECBA into detect_drift_ms. We use
        # the published ratios as a guide: ECBA ~30-36%, PC ~39-41%,
        # Inference ~0.4-2%, Update ~24-25%. Actual timing per component
        # is exposed via model._last_timing if instrumented; otherwise we
        # report total drift_detection_ms separately.
        timings['ecba_ms'].append(0.0)  # placeholder; ECBA bundled in drift
        timings['drift_detection_ms'].append(drift_ms)
        timings['inference_ms'].append(infer_ms)
        timings['training_ms'].append(train_ms)
        timings['total_ms'].append(infer_ms + drift_ms + train_ms)

    cleanup(model, windows)

    out = {}
    for k, vals in timings.items():
        out[k.replace('_ms', '')] = {
            'mean_ms': float(np.mean(vals)),
            'std_ms':  float(np.std(vals)),
            'n_windows': len(vals),
        }
    return out


def main():
    global OUTPUT_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick run: NSL-KDD only, 5 profile windows.')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab6_runtime_smoke.json')
        global N_PROFILE_WINDOWS
        N_PROFILE_WINDOWS = 5

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.6 runtime profile: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  datasets    : {args.datasets}')
    log(f'  seed        : {SEED}')
    log(f'  windows     : {N_PROFILE_WINDOWS}')
    log('=' * 60)

    results = {}
    for ds in args.datasets:
        log(f'==== {ds} ====')
        t0 = time.perf_counter()
        try:
            timings = profile_b1_on(ds)
        except Exception as e:
            log(f'  ERROR [{ds}]: {type(e).__name__}: {e}')
            raise
        elapsed = time.perf_counter() - t0
        results[ds] = timings
        log(f'  DONE [{ds}]: total={timings["total"]["mean_ms"]:.2f}ms '
            f'(infer={timings["inference"]["mean_ms"]:.2f} | '
            f'drift={timings["drift_detection"]["mean_ms"]:.2f} | '
            f'train={timings["training"]["mean_ms"]:.2f}) '
            f'({elapsed:.1f}s wall)')
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    log('=' * 60)
    log('B1 Tab.6: ALL DONE')
    for ds in args.datasets:
        if ds in results:
            t = results[ds]
            log(f'  {ds}: total={t["total"]["mean_ms"]:.2f}ms '
                f'(infer={t["inference"]["mean_ms"]:.2f} | '
                f'drift={t["drift_detection"]["mean_ms"]:.2f} | '
                f'train={t["training"]["mean_ms"]:.2f})')
    log('=' * 60)


if __name__ == '__main__':
    main()

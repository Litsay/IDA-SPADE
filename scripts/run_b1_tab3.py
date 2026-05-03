"""Run B1 Tab.3 drift-period analysis.

Reuses the canonical Tab.2 drift set (identify_drift_points, threshold=0.10,
min_gap=8 ==> 7 events on UNSW-NB15, 46 on CIC-IDS-2017) and computes
Overall / Drift F1 / Stable F1 / Penalty / Recovery for B1.

Baseline rows do NOT need to be rerun (baselines are unchanged); their numbers
are already in `experiment_results/tab3_unified_drift_analysis.json`. This
script only computes B1's row.

Output JSON
    experiment_results/b1_tab3_drift_period.json
    experiment_results/b1_tab3_drift_period.log

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab3.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab3.py --smoke
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab3_drift_period.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab3_drift_period.log')

DATASETS = ['UNSW-NB15', 'CIC-IDS-2017']
SEED = 42
HALF_WINDOW = 3


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


def run_f1_trace(dataset):
    """Run B1 prequential eval on `dataset`, seed=42. Return per-window F1."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate
    from experiments.ida_spade_b1 import IDASpadeB1
    set_seed(SEED)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)
    model = IDASpadeB1(feature_cols=fc, dataset_name=dataset)
    results = prequential_evaluate(model, iter(windows), X_init, y_init, verbose=False)
    f1_trace = [float(r.f1) for r in results]
    cleanup(model, windows)
    return f1_trace


def compute_drift_points(dataset):
    """Canonical Tab.2 drift set: threshold=0.10, min_gap=8."""
    from experiments.data_loader import load_dataset
    from experiments.drift_injection import identify_drift_points
    X_init, y_init, stream, _ = load_dataset(dataset)
    windows = list(stream)
    dp = identify_drift_points(windows, threshold=0.10, min_gap=8)
    return list(dp), len(windows)


def drift_period_stats(f1_trace, drift_points, half_window=HALF_WINDOW):
    """Overall, Drift F1, Stable F1, Penalty, Recovery from per-window F1."""
    arr = np.asarray(f1_trace, dtype=float)
    n = len(arr)
    mask = np.zeros(n, dtype=bool)
    for dp in drift_points:
        lo = max(0, dp - half_window)
        hi = min(n, dp + half_window + 1)
        mask[lo:hi] = True
    drift_mask = mask
    stable_mask = ~mask
    overall = float(arr.mean()) if n > 0 else float('nan')
    drift_f1 = float(arr[drift_mask].mean()) if drift_mask.any() else float('nan')
    stable_f1 = float(arr[stable_mask].mean()) if stable_mask.any() else float('nan')
    penalty = stable_f1 - drift_f1

    recoveries = []
    lookback = 5
    lookahead = 15
    for dp in drift_points:
        if dp <= 0 or dp >= n - 1:
            continue
        pre_lo = max(0, dp - lookback)
        pre_level = arr[pre_lo:dp].mean() if dp - pre_lo > 0 else arr[dp]
        recovered = False
        for t in range(dp + 1, min(dp + 1 + lookahead, n)):
            if arr[t] >= pre_level:
                recoveries.append(t - dp)
                recovered = True
                break
        if not recovered:
            recoveries.append(lookahead)
    mean_recovery = float(np.mean(recoveries)) if recoveries else float('nan')

    return {
        'overall_f1':       overall,
        'drift_period_f1':  drift_f1,
        'stable_period_f1': stable_f1,
        'drift_penalty':    penalty,
        'mean_recovery':    mean_recovery,
        'n_drift_windows':  int(drift_mask.sum()),
        'n_stable_windows': int(stable_mask.sum()),
    }


def load_existing():
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log(f'WARN: cannot read {OUTPUT_JSON}: {e}')
    return {}


def save(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = OUTPUT_JSON + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    if os.path.exists(OUTPUT_JSON):
        os.replace(tmp, OUTPUT_JSON)
    else:
        os.rename(tmp, OUTPUT_JSON)


def main():
    global OUTPUT_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick run: UNSW-NB15 only.')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab3_drift_period_smoke.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.3 drift-period: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  datasets    : {args.datasets}')
    log(f'  seed        : {SEED}')
    log(f'  half_window : {HALF_WINDOW}')
    log('=' * 60)

    results = load_existing()

    for ds in args.datasets:
        if ds not in results:
            results[ds] = {}

        if 'drift_points' not in results[ds]:
            try:
                dp, n_windows = compute_drift_points(ds)
            except Exception as e:
                log(f'[{ds}] FAILED drift points: {type(e).__name__}: {e}')
                continue
            results[ds]['drift_points']    = dp
            results[ds]['n_drift_points']  = len(dp)
            results[ds]['n_windows']       = n_windows
            results[ds]['half_window']     = HALF_WINDOW
            log(f'[{ds}] drift points ({len(dp)}): '
                f'{dp if len(dp) < 30 else str(dp[:10]) + "...(truncated)"}')
            save(results)
        else:
            log(f'[{ds}] SKIP drift points (resumed): '
                f'n={results[ds]["n_drift_points"]}')

        if 'B1' in results[ds].get('analysis', {}):
            log(f'[{ds}/B1] SKIP (resumed)')
            continue

        t0 = time.perf_counter()
        try:
            f1_trace = run_f1_trace(ds)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err = f'{type(e).__name__}: {e}'
            log(f'[{ds}/B1] FAILED ({elapsed:.1f}s): {err}')
            if 'analysis' not in results[ds]:
                results[ds]['analysis'] = {}
            results[ds]['analysis']['B1'] = {'error': err}
            save(results)
            continue

        stats = drift_period_stats(f1_trace, results[ds]['drift_points'])
        if 'analysis' not in results[ds]:
            results[ds]['analysis'] = {}
        if 'f1_traces' not in results[ds]:
            results[ds]['f1_traces'] = {}
        results[ds]['analysis']['B1']  = stats
        results[ds]['f1_traces']['B1'] = f1_trace
        save(results)
        elapsed = time.perf_counter() - t0
        log(f'[{ds}/B1] DONE ({elapsed:.1f}s): '
            f'Overall={stats["overall_f1"]*100:.2f} '
            f'Drift={stats["drift_period_f1"]*100:.2f} '
            f'Stable={stats["stable_period_f1"]*100:.2f} '
            f'Pen={stats["drift_penalty"]*100:.2f} '
            f'Recov={stats["mean_recovery"]:.2f}')

    log('=' * 60)
    log('B1 Tab.3: ALL DONE')
    for ds in args.datasets:
        a = results.get(ds, {}).get('analysis', {}).get('B1', {})
        if 'error' in a:
            log(f'  {ds}: B1 ERROR: {a["error"]}')
        elif a:
            log(f'  {ds}: B1  Overall={a["overall_f1"]*100:6.2f} '
                f'Drift={a["drift_period_f1"]*100:6.2f} '
                f'Pen={a["drift_penalty"]*100:+6.2f} '
                f'Recov={a["mean_recovery"]:.2f}')


if __name__ == '__main__':
    main()

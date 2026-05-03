"""B1 Tab.5 sensitivity extension: theta_LID, beta_hot, lambda_p.

Sweeps three additional hyperparameters NOT covered by the original
run_b1_tab5.py (which sweeps alpha, sigma, K_hot):

    theta_lid : LID threshold for PC-DriftForecasting   in {10, 15*, 20, 25}
    beta_hot  : prototype EMA decay during hot mode     in {0.80, 0.90*, 0.95, 0.99}
                (PROTOTYPE_BETA_DRIFT in config.py;
                 quiet-mode beta=0.99 held fixed)
    lambda_p  : prototype fusion gate during quiet mode in {0.10, 0.20*, 0.40, 0.60}
                (PROTOTYPE_WEIGHT_BASE in config.py;
                 reversal-mode lambda_p=0.6 held fixed)

* = default (paper-canonical config). 3 seeds {42, 123, 456} per grid point.
UNSW-NB15 only, matching the existing Tab.5 design.

Output JSON
    experiment_results/b1_tab5_ext.json
    experiment_results/b1_tab5_ext.log

Resume semantics: skips (parameter, value, seed) tuples already in the JSON.

Implementation note. The wrapper module imports config constants by value
(`from .config import PROTOTYPE_BETA_DRIFT, ...`), so mutating cfg.X has no
effect on already-loaded modules. We instead patch the wrapper module's
own globals (`ida_spade_wrapper.PROTOTYPE_BETA_DRIFT = X`) before constructing
the model. This works because Python resolves free names inside methods
against the defining module's globals at call time.

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5_ext.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5_ext.py --smoke
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5_ext.py --params theta_lid
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab5_ext.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab5_ext.log')

DATASET = 'UNSW-NB15'
SEEDS = [42, 123, 456]

PARAM_GRID = {
    'theta_lid': [10.0, 15.0, 20.0, 25.0],
    'beta_hot':  [0.80, 0.90, 0.95, 0.99],
    'lambda_p':  [0.10, 0.20, 0.40, 0.60],
}
DEFAULTS = {
    'theta_lid': 15.0,
    'beta_hot':  0.90,
    'lambda_p':  0.20,
}


def log(msg):
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def load_existing_results():
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log(f'WARN: failed to load {OUTPUT_JSON}: {e}')
    return {}


def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = OUTPUT_JSON + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    if os.path.exists(OUTPUT_JSON):
        os.replace(tmp, OUTPUT_JSON)
    else:
        os.rename(tmp, OUTPUT_JSON)


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


def run_one(param, value, seed):
    """Run B1 with one parameter overridden. Returns metric dict (percent)."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate, aggregate_metrics
    from experiments.ida_spade_b1 import IDASpadeB1
    import experiments.ida_spade_wrapper as wrapper_mod

    # Save originals (wrapper module globals)
    orig_theta_lid = wrapper_mod.PC_THETA_LID
    orig_beta_drift = wrapper_mod.PROTOTYPE_BETA_DRIFT
    orig_lambda_base = wrapper_mod.PROTOTYPE_WEIGHT_BASE

    try:
        if param == 'theta_lid':
            wrapper_mod.PC_THETA_LID = float(value)
        elif param == 'beta_hot':
            wrapper_mod.PROTOTYPE_BETA_DRIFT = float(value)
        elif param == 'lambda_p':
            wrapper_mod.PROTOTYPE_WEIGHT_BASE = float(value)
        else:
            raise ValueError(f'Unknown param: {param!r}')

        set_seed(seed)
        X_init, y_init, stream, fc = load_dataset(DATASET)
        windows = list(stream)

        model = IDASpadeB1(feature_cols=fc, dataset_name=DATASET)

        # Belt-and-suspenders: also patch the constructed forecaster /
        # instance state directly, so a stale __init__ binding cannot
        # silently keep the default value.
        if param == 'theta_lid' and model.use_causal:
            try:
                model._pc_forecaster.theta_lid = float(value)
            except Exception:
                pass
        if param == 'lambda_p':
            model._prototype_weight = float(value)

        results = prequential_evaluate(
            model, iter(windows), X_init, y_init, verbose=False)
        agg = aggregate_metrics(results)
        out = {
            'f1':  float(agg['f1']) * 100,
            'acc': float(agg['accuracy']) * 100,
            'pre': float(agg['precision']) * 100,
            'rec': float(agg['recall']) * 100,
        }
        cleanup(model, windows)
        return out
    finally:
        # Restore wrapper module originals for the next iteration.
        wrapper_mod.PC_THETA_LID = orig_theta_lid
        wrapper_mod.PROTOTYPE_BETA_DRIFT = orig_beta_drift
        wrapper_mod.PROTOTYPE_WEIGHT_BASE = orig_lambda_base


def initialize_block(results, param, value):
    if param not in results:
        results[param] = {}
    key = str(value)
    if key not in results[param]:
        results[param][key] = {
            'f1_values':  [],
            'acc_values': [],
            'pre_values': [],
            'rec_values': [],
            'seeds_done': [],
        }


def already_done(results, param, value, seed):
    block = results.get(param, {}).get(str(value), {})
    return seed in block.get('seeds_done', [])


def append_run(results, param, value, seed, metrics):
    block = results[param][str(value)]
    block['f1_values'].append(metrics['f1'])
    block['acc_values'].append(metrics['acc'])
    block['pre_values'].append(metrics['pre'])
    block['rec_values'].append(metrics['rec'])
    block['seeds_done'].append(seed)


def finalize(results, param):
    for key, block in results[param].items():
        if key.startswith('_'):
            continue
        f1s = block.get('f1_values', [])
        if not f1s:
            continue
        block['f1_mean']  = float(np.mean(f1s))
        block['f1_std']   = float(np.std(f1s))
        block['acc_mean'] = float(np.mean(block.get('acc_values', [0])))
        block['acc_std']  = float(np.std(block.get('acc_values', [0])))


def run_param(results, param, values, seeds):
    log(f'==== sweep {param}: values={values} ({len(seeds)} seeds each) ====')
    for v in values:
        initialize_block(results, param, v)
    save_results(results)

    for v in values:
        for seed in seeds:
            if already_done(results, param, v, seed):
                log(f'  SKIP [{param}={v}/seed={seed}] (already done)')
                continue
            t0 = time.perf_counter()
            try:
                metrics = run_one(param, v, seed)
            except Exception as e:
                log(f'  ERROR [{param}={v}/seed={seed}]: {type(e).__name__}: {e}')
                raise
            elapsed = time.perf_counter() - t0
            append_run(results, param, v, seed, metrics)
            save_results(results)
            log(f'  DONE [{param}={v}/seed={seed}]: '
                f'F1={metrics["f1"]:.2f}% acc={metrics["acc"]:.2f}% '
                f'({elapsed:.1f}s)')

    finalize(results, param)
    save_results(results)
    log(f'  SUMMARY [{param}]:')
    for v in values:
        block = results[param].get(str(v), {})
        if 'f1_mean' in block:
            log(f'    {param}={v}: F1={block["f1_mean"]:.2f}% '
                f'+/- {block["f1_std"]:.2f}% '
                f'(n={len(block["f1_values"])})')


def main():
    global OUTPUT_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick run: theta_lid=10 only, seed=42 only.')
    parser.add_argument('--params', nargs='+',
                        default=list(PARAM_GRID.keys()),
                        choices=list(PARAM_GRID.keys()))
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab5_ext_smoke.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.5 EXT sensitivity: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  dataset     : {DATASET}')
    log(f'  seeds       : {SEEDS}')
    log(f'  params      : {args.params}')
    log('=' * 60)

    if args.smoke:
        results = {}
        run_param(results, 'theta_lid', [10.0], [42])
        log('SMOKE OK')
        return

    results = load_existing_results()
    if results:
        log(f'  RESUME: loaded prior partial state')

    for param in args.params:
        run_param(results, param, PARAM_GRID[param], SEEDS)

    log('=' * 60)
    log('B1 Tab.5 EXT: ALL DONE')
    log('=' * 60)


if __name__ == '__main__':
    main()

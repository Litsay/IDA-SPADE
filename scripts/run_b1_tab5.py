"""Run B1 Tab.5 hyperparameter sensitivity sweep on UNSW-NB15.

Sweeps three parameters one at a time (others held at default):
    alpha  : SupCon weight        in {0.05, 0.15*, 0.30, 0.50}
    sigma  : PC drift sensitivity in {0.5, 1.0*, 1.5, 2.0}
    K_hot  : alert cooldown        in {1, 3*, 5, 8}

* = default. 3 seeds {42, 123, 456} per grid point. UNSW-NB15 only (matches
the existing Tab.5 design in the paper, which sweeps only alpha and sigma).

Output JSON
    experiment_results/b1_tab5_sensitivity.json
    experiment_results/b1_tab5_sensitivity.log

Resume: skips (parameter, value, seed) tuples already in the JSON.

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5.py --smoke
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab5.py --params alpha
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab5_sensitivity.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab5_sensitivity.log')

DATASET = 'UNSW-NB15'
SEEDS = [42, 123, 456]

PARAM_GRID = {
    'alpha': [0.05, 0.15, 0.30, 0.50],
    'sigma': [0.5, 1.0, 1.5, 2.0],
    'K_hot': [1, 3, 5, 8],
}
DEFAULTS = {'alpha': 0.15, 'sigma': 1.0, 'K_hot': 3}


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
    # Patch config for alpha / sigma BEFORE constructing model
    import experiments.config as cfg

    # Save originals
    orig_alpha = cfg.CONTRASTIVE_ALPHA
    orig_sigma = cfg.PC_SIGMA

    try:
        if param == 'alpha':
            cfg.CONTRASTIVE_ALPHA = float(value)
        if param == 'sigma':
            cfg.PC_SIGMA = float(value)

        set_seed(seed)
        X_init, y_init, stream, fc = load_dataset(DATASET)
        windows = list(stream)

        K_hot = int(value) if param == 'K_hot' else DEFAULTS['K_hot']
        model = IDASpadeB1(feature_cols=fc, dataset_name=DATASET, K_hot=K_hot)

        # If param is alpha, also set it on the constructed model (the
        # wrapper copies CONTRASTIVE_ALPHA into self._contrastive_alpha at
        # __init__ time; it should already be picked up from cfg, but set
        # explicitly to be safe).
        if param == 'alpha':
            model._contrastive_alpha = float(value)
        if param == 'sigma' and model.use_causal:
            try:
                model._pc_forecaster.sigma = float(value)
            except Exception:
                pass

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
        # Restore config originals
        cfg.CONTRASTIVE_ALPHA = orig_alpha
        cfg.PC_SIGMA = orig_sigma


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
                        help='Quick run: alpha=0.05 only, seed=42 only.')
    parser.add_argument('--params', nargs='+',
                        default=list(PARAM_GRID.keys()),
                        choices=list(PARAM_GRID.keys()))
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab5_sensitivity_smoke.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.5 sensitivity: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  dataset     : {DATASET}')
    log(f'  seeds       : {SEEDS}')
    log(f'  params      : {args.params}')
    log('=' * 60)

    if args.smoke:
        results = {}
        run_param(results, 'alpha', [0.05], [42])
        log('SMOKE OK')
        return

    results = load_existing_results()
    if results:
        log(f'  RESUME: loaded prior partial state')

    for param in args.params:
        run_param(results, param, PARAM_GRID[param], SEEDS)

    log('=' * 60)
    log('B1 Tab.5: ALL DONE')
    log('=' * 60)


if __name__ == '__main__':
    main()

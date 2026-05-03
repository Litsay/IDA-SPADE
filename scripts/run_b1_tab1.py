"""Run B1 Tab.1 main results: 10/10/5-seed prequential evaluation.

Produces the new "Full = B1" row to replace IDA-SPADE in Tab.1 of the paper.
Other 8 baselines are NOT rerun here (their numbers are stable per the existing
phase1_summary.json / cic_tab1_metrics_5seed.json), so this script only runs
the B1 model. Welch's t-test against the runner-up baseline is computed by
loading the existing baseline JSONs.

Output JSON
    experiment_results/b1_tab1_main.json
    experiment_results/b1_tab1_main.log

Resume: skips (dataset, seed) tuples already in the JSON.

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab1.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab1.py --smoke
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab1.py --datasets UNSW-NB15
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab1_main.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab1_main.log')

NSL_UNSW_SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 4096, 5555, 7777]
CIC_SEEDS = [42, 123, 456, 789, 1024]


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


def run_one(dataset, seed):
    """Run B1 on one (dataset, seed). Returns metric dict (percent)."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate, aggregate_metrics
    from experiments.ida_spade_b1 import IDASpadeB1

    set_seed(seed)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)
    model = IDASpadeB1(feature_cols=fc, dataset_name=dataset)
    results = prequential_evaluate(model, iter(windows), X_init, y_init, verbose=False)
    agg = aggregate_metrics(results)
    out = {
        'f1':  float(agg['f1']) * 100,
        'acc': float(agg['accuracy']) * 100,
        'pre': float(agg['precision']) * 100,
        'rec': float(agg['recall']) * 100,
        'n_windows': len(results),
        'n_drifts':  int(agg.get('n_drifts', 0)),
    }
    cleanup(model, windows)
    return out


def initialize_dataset_block(results, dataset):
    if dataset not in results:
        results[dataset] = {
            'B1': {
                'f1_values':  [],
                'acc_values': [],
                'pre_values': [],
                'rec_values': [],
                'seeds_done': [],
            },
        }


def already_done(results, dataset, seed):
    return seed in results.get(dataset, {}).get('B1', {}).get('seeds_done', [])


def append_run(results, dataset, seed, metrics):
    block = results[dataset]['B1']
    block['f1_values'].append(metrics['f1'])
    block['acc_values'].append(metrics['acc'])
    block['pre_values'].append(metrics['pre'])
    block['rec_values'].append(metrics['rec'])
    block['seeds_done'].append(seed)


def finalize_dataset(results, dataset):
    block = results[dataset]['B1']
    f1s = block.get('f1_values', [])
    if not f1s:
        return
    block['f1_mean']  = float(np.mean(f1s))
    block['f1_std']   = float(np.std(f1s))
    block['acc_mean'] = float(np.mean(block['acc_values']))
    block['acc_std']  = float(np.std(block['acc_values']))
    block['pre_mean'] = float(np.mean(block['pre_values']))
    block['pre_std']  = float(np.std(block['pre_values']))
    block['rec_mean'] = float(np.mean(block['rec_values']))
    block['rec_std']  = float(np.std(block['rec_values']))


def run_dataset(results, dataset, seeds):
    log(f'==== {dataset}: B1 x {len(seeds)} seeds ====')
    initialize_dataset_block(results, dataset)
    save_results(results)

    for seed in seeds:
        if already_done(results, dataset, seed):
            log(f'  SKIP [{dataset}/B1/seed={seed}] (already done)')
            continue
        t0 = time.perf_counter()
        try:
            metrics = run_one(dataset, seed)
        except Exception as e:
            log(f'  ERROR [{dataset}/B1/seed={seed}]: {type(e).__name__}: {e}')
            raise
        elapsed = time.perf_counter() - t0
        append_run(results, dataset, seed, metrics)
        save_results(results)
        log(f'  DONE [{dataset}/B1/seed={seed}]: '
            f'F1={metrics["f1"]:.2f}% acc={metrics["acc"]:.2f}% '
            f'pre={metrics["pre"]:.2f}% rec={metrics["rec"]:.2f}% '
            f'(windows={metrics["n_windows"]}, {elapsed:.1f}s)')

    finalize_dataset(results, dataset)
    save_results(results)
    block = results[dataset]['B1']
    if 'f1_mean' in block:
        log(f'  SUMMARY [{dataset}]: B1 F1={block["f1_mean"]:.2f}% '
            f'+/- {block["f1_std"]:.2f}% (n={len(block["f1_values"])})')


def main():
    global OUTPUT_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick run: NSL-KDD seed=42 only.')
    parser.add_argument('--datasets', nargs='+',
                        default=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017'])
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab1_main_smoke.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.1 main results: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  datasets    : {args.datasets}')
    log('=' * 60)

    if args.smoke:
        results = {}
        run_dataset(results, 'NSL-KDD', [42])
        log('SMOKE OK')
        return

    results = load_existing_results()
    if results:
        log(f'  RESUME: loaded prior partial state')

    for ds in args.datasets:
        seeds = CIC_SEEDS if ds == 'CIC-IDS-2017' else NSL_UNSW_SEEDS
        run_dataset(results, ds, seeds)

    log('=' * 60)
    log('B1 Tab.1: ALL DONE')
    log('=' * 60)


if __name__ == '__main__':
    main()

"""Run B1 Tab.4 ablation study: 7 variants x 3 datasets x 5-10 seeds.

Variants
    B1 (Full)      : new baseline (replaces IDA-SPADE in Tab.4)
    B1-NoPCAlert   : PC channel removed from alert signal
    B1-Reactive    : KS-test alert (replaces PC)
    B1-Statistical : sliding z-score alert (replaces PC)
    B1-Global      : random entity grouping (replaces ECBA)
    B1-NoCL        : without supervised contrastive head
    B1-NoProto     : without prototype module

For each (dataset, variant, seed), runs prequential evaluation and stores
the per-seed F1 / Acc / Pre / Rec. After all seeds complete for a (dataset,
variant), computes paired t-test and Wilcoxon against B1 (Full).

Output JSON
    experiment_results/b1_tab4_ablation.json
    experiment_results/b1_tab4_ablation.log

Resume: skips (dataset, variant, seed) tuples already in the JSON.

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab4.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab4.py --smoke
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab4.py --datasets UNSW-NB15
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_tab4.py --variants B1-NoPCAlert
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab4_ablation.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'b1_tab4_ablation.log')

NSL_UNSW_SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 4096, 5555, 7777]
CIC_SEEDS = [42, 123, 456, 789, 1024]

ALL_VARIANTS = [
    'B1',
    'B1-NoPCAlert',
    'B1-Reactive',
    'B1-Statistical',
    'B1-Global',
    'B1-NoCL',
    'B1-NoProto',
    'B1-Reactive-NoProto',
]


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


def run_one(dataset, variant, seed):
    """Run one (dataset, variant, seed). Returns metric dict in percent."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate, aggregate_metrics
    from experiments.ida_spade_b1 import make_b1_variant

    set_seed(seed)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)
    model = make_b1_variant(variant, feature_cols=fc, dataset_name=dataset)
    results = prequential_evaluate(model, iter(windows), X_init, y_init, verbose=False)
    agg = aggregate_metrics(results)
    out = {
        'f1':  float(agg['f1']) * 100,
        'acc': float(agg['accuracy']) * 100,
        'pre': float(agg['precision']) * 100,
        'rec': float(agg['recall']) * 100,
    }
    cleanup(model, windows)
    return out


def initialize_dataset_block(results, dataset, variants):
    if dataset not in results:
        results[dataset] = {}
    block = results[dataset]
    for v in variants:
        if v not in block:
            block[v] = {
                'f1_values':  [],
                'acc_values': [],
                'pre_values': [],
                'rec_values': [],
                'seeds_done': [],
            }
    if '_statistical_tests' not in block:
        block['_statistical_tests'] = {}


def already_done(results, dataset, variant, seed):
    return seed in results.get(dataset, {}).get(variant, {}).get('seeds_done', [])


def append_run(results, dataset, variant, seed, metrics):
    block = results[dataset][variant]
    block['f1_values'].append(metrics['f1'])
    block['acc_values'].append(metrics['acc'])
    block['pre_values'].append(metrics['pre'])
    block['rec_values'].append(metrics['rec'])
    block['seeds_done'].append(seed)


def finalize_dataset(results, dataset, variants):
    from scipy import stats as scipy_stats
    block = results[dataset]
    full_block = block.get('B1', {})
    full_f1s = full_block.get('f1_values', [])
    full_seeds = full_block.get('seeds_done', [])
    full_per_seed = {str(s): float(v) for s, v in zip(full_seeds, full_f1s)}

    for v in variants:
        vb = block.get(v, {})
        f1s = vb.get('f1_values', [])
        if not f1s:
            continue
        vb['f1_mean']  = float(np.mean(f1s))
        vb['f1_std']   = float(np.std(f1s))
        vb['acc_mean'] = float(np.mean(vb.get('acc_values', [0])))
        vb['acc_std']  = float(np.std(vb.get('acc_values', [0])))
        vb['pre_mean'] = float(np.mean(vb.get('pre_values', [0])))
        vb['pre_std']  = float(np.std(vb.get('pre_values', [0])))
        vb['rec_mean'] = float(np.mean(vb.get('rec_values', [0])))
        vb['rec_std']  = float(np.std(vb.get('rec_values', [0])))

        if v == 'B1' or not full_per_seed:
            continue
        done_seeds = vb.get('seeds_done', [])
        paired_full, paired_var = [], []
        for seed, val in zip(done_seeds, f1s):
            key = str(seed)
            if key in full_per_seed:
                paired_full.append(full_per_seed[key])
                paired_var.append(val)
        if len(paired_full) >= 2:
            try:
                t_stat, t_p = scipy_stats.ttest_rel(paired_full, paired_var)
                w_stat, w_p = scipy_stats.wilcoxon(paired_full, paired_var)
                block['_statistical_tests'][v] = {
                    'ttest_stat':    float(t_stat),
                    'ttest_p':       float(t_p),
                    'wilcoxon_stat': float(w_stat),
                    'wilcoxon_p':    float(w_p),
                    'n_pairs':       len(paired_full),
                }
            except Exception as e:
                block['_statistical_tests'][v] = {'error': str(e)}


def run_dataset(results, dataset, seeds, variants):
    log(f'==== {dataset}: {len(variants)} variants x {len(seeds)} seeds ====')
    initialize_dataset_block(results, dataset, variants)
    save_results(results)

    for variant in variants:
        for seed in seeds:
            if already_done(results, dataset, variant, seed):
                log(f'  SKIP [{dataset}/{variant}/seed={seed}] (already done)')
                continue
            t0 = time.perf_counter()
            try:
                metrics = run_one(dataset, variant, seed)
            except Exception as e:
                log(f'  ERROR [{dataset}/{variant}/seed={seed}]: '
                    f'{type(e).__name__}: {e}')
                raise
            elapsed = time.perf_counter() - t0
            append_run(results, dataset, variant, seed, metrics)
            save_results(results)
            log(f'  DONE [{dataset}/{variant}/seed={seed}]: '
                f'F1={metrics["f1"]:.2f}% acc={metrics["acc"]:.2f}% '
                f'pre={metrics["pre"]:.2f}% rec={metrics["rec"]:.2f}% '
                f'({elapsed:.1f}s)')

    finalize_dataset(results, dataset, variants)
    save_results(results)
    block = results[dataset]
    log(f'  SUMMARY [{dataset}]:')
    for v in variants:
        vb = block.get(v, {})
        if 'f1_mean' in vb:
            line = (f'    {v:18s}: F1={vb["f1_mean"]:.2f}% '
                    f'+/- {vb["f1_std"]:.2f}% '
                    f'(n={len(vb.get("f1_values", []))})')
            tests = block.get('_statistical_tests', {}).get(v)
            if tests and 'ttest_p' in tests:
                line += (f'   vs B1: t_p={tests["ttest_p"]:.4f} '
                         f'wilcoxon_p={tests["wilcoxon_p"]:.4f}')
            log(line)


def main():
    global OUTPUT_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick run: NSL-KDD seed=42, B1 + B1-NoPCAlert only.')
    parser.add_argument('--datasets', nargs='+',
                        default=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017'])
    parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS,
                        choices=ALL_VARIANTS)
    args = parser.parse_args()

    if args.smoke:
        OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'b1_tab4_ablation_smoke.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log('=' * 60)
    log('B1 Tab.4 ablation: starting' + (' [SMOKE]' if args.smoke else ''))
    log(f'  output JSON : {OUTPUT_JSON}')
    log(f'  log file    : {LOG_PATH}')
    log(f'  datasets    : {args.datasets}')
    log(f'  variants    : {args.variants}')
    log('=' * 60)

    if args.smoke:
        results = {}
        run_dataset(results, 'NSL-KDD', [42], ['B1', 'B1-NoPCAlert'])
        log('SMOKE OK')
        return

    results = load_existing_results()
    if results:
        log(f'  RESUME: loaded prior partial state')

    for ds in args.datasets:
        seeds = CIC_SEEDS if ds == 'CIC-IDS-2017' else NSL_UNSW_SEEDS
        run_dataset(results, ds, seeds, args.variants)

    log('=' * 60)
    log('B1 Tab.4: ALL DONE')
    log('=' * 60)


if __name__ == '__main__':
    main()

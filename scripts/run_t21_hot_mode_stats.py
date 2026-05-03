"""T2.1: Measure hot-mode ratio and reversal-trigger frequency for IDA-SPADE B1.

Runs B1 (Full) on UNSW-NB15 and CIC-IDS-2017 at seed=42 (matches Tab.3 protocol)
and tallies per-window:
  - alert flag (Drift_Alert(k))
  - hot vs quiet mode (alert_cooldown > 0)
  - reversal trigger (|attack_ratio - 5-window mean| > 0.3)

Output: experiment_results/t21_hot_mode_stats.json
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(dataset, seed=42):
    """Run B1 on dataset and return per-window state trace."""
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate
    from experiments.ida_spade_b1 import make_b1_variant

    set_seed(seed)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)

    model = make_b1_variant('B1', feature_cols=fc, dataset_name=dataset)

    # Instrumentation: monkey-patch detect_drift and update to track state
    state_trace = {
        'alert_fired': [],          # bool per window: did alert fire?
        'hot_mode': [],             # bool per window: alert_cooldown > 0
        'reversal_triggered': [],   # bool per window
        'attack_ratio': [],         # per-window attack ratio (entity-level)
    }

    orig_detect = model.detect_drift
    orig_update = model.update

    last_alert = {'fired': False}

    def wrapped_detect(X):
        detected, conf = orig_detect(X)
        last_alert['fired'] = bool(detected)
        return detected, conf

    def wrapped_update(X, y):
        # capture state BEFORE update mutates it
        is_hot = bool(model._alert_cooldown > 0 if hasattr(model, '_alert_cooldown') else False)

        # entity-level attack ratio (after ECBA)
        feat_tensor, entity_order, entity_ids = model._get_cached_ecba(X)
        if feat_tensor is None:
            current_ratio = 0.0
            reversal = False
        else:
            entity_labels = model._extract_entity_labels(y, entity_ids, entity_order)
            current_ratio = float(entity_labels.float().mean())

            # check reversal: |current - 5-window mean| > 0.3
            history = list(model._attack_ratio_history)
            if len(history) >= 5:
                # The wrapper uses [-6:-1] which is 5 windows ending right before current
                recent = history[-5:] if len(history) >= 5 else history
                recent_avg = float(np.mean(recent))
                reversal = abs(current_ratio - recent_avg) > 0.3
            else:
                reversal = False

        state_trace['alert_fired'].append(last_alert['fired'])
        state_trace['hot_mode'].append(is_hot)
        state_trace['reversal_triggered'].append(reversal)
        state_trace['attack_ratio'].append(current_ratio)

        return orig_update(X, y)

    model.detect_drift = wrapped_detect
    model.update = wrapped_update

    t0 = time.perf_counter()
    results = prequential_evaluate(model, iter(windows), X_init, y_init, verbose=False)
    elapsed = time.perf_counter() - t0

    n_total = len(state_trace['alert_fired'])
    n_alert = sum(state_trace['alert_fired'])
    n_hot = sum(state_trace['hot_mode'])
    n_reversal = sum(state_trace['reversal_triggered'])

    summary = {
        'dataset': dataset,
        'seed': seed,
        'total_windows': n_total,
        'alert_count': n_alert,
        'alert_ratio': n_alert / max(n_total, 1),
        'hot_mode_count': n_hot,
        'hot_mode_ratio': n_hot / max(n_total, 1),
        'reversal_count': n_reversal,
        'reversal_ratio': n_reversal / max(n_total, 1),
        'elapsed_sec': elapsed,
    }
    print(f'\n=== {dataset} (seed={seed}) ===')
    print(f'  total windows:     {n_total}')
    print(f'  alerts fired:      {n_alert} ({summary["alert_ratio"]*100:.2f}%)')
    print(f'  hot-mode windows:  {n_hot} ({summary["hot_mode_ratio"]*100:.2f}%)')
    print(f'  reversals:         {n_reversal} ({summary["reversal_ratio"]*100:.2f}%)')
    print(f'  elapsed:           {elapsed:.1f}s')

    return summary, state_trace


def main():
    out_dir = 'experiment_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 't21_hot_mode_stats.json')

    all_summaries = {}
    for dataset in ['UNSW-NB15', 'CIC-IDS-2017']:
        try:
            summary, trace = run_one(dataset, seed=42)
            all_summaries[dataset] = {
                'summary': summary,
                # store trace for later analysis but not expanded in summary
                'trace': trace,
            }
        except Exception as e:
            print(f'ERROR on {dataset}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            all_summaries[dataset] = {'error': f'{type(e).__name__}: {e}'}

    with open(out_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()

"""T2.4: Representation-quality non-F1 evidence for SupCon (NoCL) ablation.

Runs B1 (Full) and B1-NoCL on UNSW-NB15 at seed=42. After every window's
training step, extracts the backbone embedding z_{e,k} = f_theta(nu_{e,k})
of each entity in the current window and computes:

  - silhouette score (binary attack/benign labels) per window
  - cluster purity per window
  - per-class centroid drift rate (L2 norm between consecutive class centroids)

Aggregates across all windows and reports mean+/-std over the full stream
plus a drift-window restricted view (within +/-3 of GT events from tab2).

Output: experiment_results/t24_repr_quality.json
"""
import os, sys, json, time
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def silhouette_binary(features, labels):
    """Binary-label silhouette score. Returns NaN if one-class window."""
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float('nan')
    try:
        return float(silhouette_score(features, labels, metric='euclidean'))
    except Exception:
        return float('nan')


def cluster_purity(features, labels, n_clusters=2):
    """KMeans into n_clusters, then majority-label purity."""
    from sklearn.cluster import KMeans
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < n_clusters:
        return float('nan')
    try:
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(features)
        cluster_ids = km.labels_
        n_correct = 0
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            if mask.sum() == 0:
                continue
            majority = np.bincount(labels[mask].astype(int)).max()
            n_correct += majority
        return float(n_correct / len(labels))
    except Exception:
        return float('nan')


def run_one(variant, dataset, seed=42, drift_windows=None):
    from experiments.data_loader import load_dataset
    from experiments.evaluator import prequential_evaluate
    from experiments.ida_spade_b1 import make_b1_variant

    set_seed(seed)
    X_init, y_init, stream, fc = load_dataset(dataset)
    windows = list(stream)
    model = make_b1_variant(variant, feature_cols=fc, dataset_name=dataset)

    per_window = {
        'window_idx': [],
        'silhouette': [],
        'purity': [],
        'centroid_attack': [],   # mean attack-class embedding (vector)
        'centroid_benign': [],
        'n_attack': [],
        'n_benign': [],
    }

    orig_update = model.update
    win_idx = {'k': 0}

    def wrapped_update(X, y):
        result = orig_update(X, y)

        # extract entity-level features and labels post-update
        feat_tensor, entity_order, entity_ids = model._get_cached_ecba(X)
        if feat_tensor is None:
            win_idx['k'] += 1
            return result
        entity_labels = model._extract_entity_labels(y, entity_ids, entity_order)
        feat_dev = feat_tensor.to(model.device)

        model.model.eval()
        with torch.no_grad():
            try:
                logits, feat = model.model(feat_dev)
            except Exception:
                # try return_all signature
                out = model.model(feat_dev, return_all=True)
                feat = out[1]
        model.model.train()

        feats_np = feat.detach().cpu().numpy()
        labels_np = entity_labels.detach().cpu().numpy().astype(int)

        sil = silhouette_binary(feats_np, labels_np)
        pur = cluster_purity(feats_np, labels_np)

        attack_mask = labels_np == 1
        benign_mask = labels_np == 0
        attack_centroid = (feats_np[attack_mask].mean(axis=0).tolist()
                           if attack_mask.sum() > 0 else None)
        benign_centroid = (feats_np[benign_mask].mean(axis=0).tolist()
                           if benign_mask.sum() > 0 else None)

        per_window['window_idx'].append(win_idx['k'])
        per_window['silhouette'].append(sil)
        per_window['purity'].append(pur)
        per_window['centroid_attack'].append(attack_centroid)
        per_window['centroid_benign'].append(benign_centroid)
        per_window['n_attack'].append(int(attack_mask.sum()))
        per_window['n_benign'].append(int(benign_mask.sum()))

        win_idx['k'] += 1
        return result

    model.update = wrapped_update

    t0 = time.perf_counter()
    _ = prequential_evaluate(model, iter(windows), X_init, y_init, verbose=False)
    elapsed = time.perf_counter() - t0

    # Compute centroid drift per consecutive window pair
    centroid_drift_attack, centroid_drift_benign = [], []
    last_a, last_b = None, None
    for ca, cb in zip(per_window['centroid_attack'], per_window['centroid_benign']):
        if ca is not None:
            ca = np.asarray(ca)
            if last_a is not None:
                centroid_drift_attack.append(float(np.linalg.norm(ca - last_a)))
            last_a = ca
        if cb is not None:
            cb = np.asarray(cb)
            if last_b is not None:
                centroid_drift_benign.append(float(np.linalg.norm(cb - last_b)))
            last_b = cb

    sil_arr = np.array([v for v in per_window['silhouette'] if not np.isnan(v)])
    pur_arr = np.array([v for v in per_window['purity'] if not np.isnan(v)])

    summary = {
        'variant': variant,
        'dataset': dataset,
        'seed': seed,
        'elapsed_sec': elapsed,
        'n_windows': len(per_window['window_idx']),
        'silhouette_mean': float(np.mean(sil_arr)) if len(sil_arr) > 0 else None,
        'silhouette_std':  float(np.std(sil_arr))  if len(sil_arr) > 0 else None,
        'silhouette_n_valid': int(len(sil_arr)),
        'purity_mean': float(np.mean(pur_arr)) if len(pur_arr) > 0 else None,
        'purity_std':  float(np.std(pur_arr))  if len(pur_arr) > 0 else None,
        'centroid_drift_attack_mean': float(np.mean(centroid_drift_attack)) if centroid_drift_attack else None,
        'centroid_drift_attack_std':  float(np.std(centroid_drift_attack))  if centroid_drift_attack else None,
        'centroid_drift_benign_mean': float(np.mean(centroid_drift_benign)) if centroid_drift_benign else None,
        'centroid_drift_benign_std':  float(np.std(centroid_drift_benign))  if centroid_drift_benign else None,
    }

    # Drift-window restricted view: only keep windows within +/-3 of any GT event
    if drift_windows is not None:
        drift_set = set(drift_windows)
        is_drift_window = []
        for k in per_window['window_idx']:
            is_drift_window.append(any(abs(k - g) <= 3 for g in drift_set))
        sil_drift = [v for v, d in zip(per_window['silhouette'], is_drift_window)
                     if d and not np.isnan(v)]
        pur_drift = [v for v, d in zip(per_window['purity'], is_drift_window)
                     if d and not np.isnan(v)]
        summary['silhouette_mean_DRIFT'] = float(np.mean(sil_drift)) if sil_drift else None
        summary['purity_mean_DRIFT']     = float(np.mean(pur_drift)) if pur_drift else None
        summary['n_drift_windows']        = int(sum(is_drift_window))

    print(f'\n=== {variant} on {dataset} (seed={seed}) ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    return summary


def main():
    out_dir = 'experiment_results'
    os.makedirs(out_dir, exist_ok=True)

    # Load drift windows from existing tab2 detection set
    with open('experiment_results/tab2_unified_detection.json') as f:
        tab2 = json.load(f)
    unsw_drifts = tab2['UNSW-NB15']['gt_drift_points']

    out = {}
    for variant in ['B1', 'B1-NoCL']:
        try:
            s = run_one(variant, 'UNSW-NB15', seed=42, drift_windows=unsw_drifts)
            out[variant] = s
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[variant] = {'error': f'{type(e).__name__}: {e}'}

    with open(os.path.join(out_dir, 't24_repr_quality.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_dir}/t24_repr_quality.json')


if __name__ == '__main__':
    main()

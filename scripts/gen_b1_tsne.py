"""Regenerate Fig.5 (t-SNE of IDA-SPADE B1 vs SSF on UNSW-NB15).

Pipeline:
  1. Run B1 prequential evaluation on UNSW-NB15 (seed=42) to convergence
  2. Capture entity-level embeddings from the LAST few windows
  3. Run SSF prequential evaluation on UNSW-NB15 (seed=42)
  4. Capture connection-level embeddings from the SSF encoder on the same windows
  5. Apply t-SNE (perplexity=30, random_state=42) to each
  6. Render 1x2 figure: (a) IDA-SPADE  (b) SSF, color-coded by binary label

Outputs (overwriting):
    figures/t-SNE.pdf
    figures/t-SNE.png

Usage:
    /c/Users/Litsay/anaconda3/envs/CL/python.exe gen_b1_tsne.py
"""
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PROJECT_ROOT = os.path.dirname(THIS_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

DATASET = 'UNSW-NB15'
SEED = 42
N_TAIL_WINDOWS_B1 = 60   # B1 emits ~6 entity-level points per window
N_TAIL_WINDOWS_SSF = 5   # SSF emits ~1000 connection-level points per window


def prequential_collect_b1():
    """Run B1 prequential on UNSW; collect entity-level embeddings from the
    LAST `N_TAIL_WINDOWS` of the stream after the model is fully trained."""
    from experiments.data_loader import load_dataset
    from experiments.ida_spade_b1 import IDASpadeB1

    set_seed(SEED)
    print(f'[B1] loading {DATASET}...')
    X_init, y_init, stream, fc = load_dataset(DATASET)
    windows = list(stream)
    print(f'[B1] init={X_init.shape} windows={len(windows)}')

    print(f'[B1] constructing model...')
    model = IDASpadeB1(feature_cols=fc, dataset_name=DATASET)
    model.initialize(X_init, y_init)

    feats_buf = []
    labels_buf = []
    n_total = len(windows)

    print(f'[B1] running prequential ({n_total} windows)...')
    t0 = time.time()
    for w_idx, w in enumerate(windows):
        X, y = w.X, w.y_binary
        # Test-then-train
        model.predict_evaluate(X, y)
        model.detect_drift(X)
        model.update(X, y)

        # Harvest entity-level embeddings from the last N windows
        if w_idx >= n_total - N_TAIL_WINDOWS_B1:
            feat_tensor, entity_order, entity_ids = model._get_cached_ecba(X)
            if feat_tensor is None or model.model is None:
                continue
            model.model.eval()
            with torch.no_grad():
                cur = feat_tensor.to(model.device)
                _, feat, _ = model.model(cur, return_all=True)
                feats_buf.append(feat.cpu().numpy())
            entity_labels = model._extract_entity_labels(y, entity_ids, entity_order)
            labels_buf.append(entity_labels.numpy())

    feats = np.concatenate(feats_buf, axis=0)
    labels = np.concatenate(labels_buf, axis=0)
    elapsed = time.time() - t0
    print(f'[B1] done in {elapsed:.1f}s; embeddings: {feats.shape}, labels: {labels.shape}')
    return feats, labels


def prequential_collect_ssf():
    """Run SSF prequential on UNSW; collect connection-level encoder
    embeddings from the LAST `N_TAIL_WINDOWS` of the stream."""
    from experiments.data_loader import load_dataset
    from experiments.baselines.ssf_baseline import SSFBaseline

    set_seed(SEED)
    print(f'[SSF] loading {DATASET}...')
    X_init, y_init, stream, fc = load_dataset(DATASET)
    windows = list(stream)
    print(f'[SSF] init={X_init.shape} windows={len(windows)}')

    print(f'[SSF] constructing model...')
    model = SSFBaseline()
    model.initialize(X_init, y_init)

    feats_buf = []
    labels_buf = []
    n_total = len(windows)

    print(f'[SSF] running prequential ({n_total} windows)...')
    t0 = time.time()
    for w_idx, w in enumerate(windows):
        X, y = w.X, w.y_binary
        model.predict_evaluate(X, y)
        model.detect_drift(X)
        model.update(X, y)

        if w_idx >= n_total - N_TAIL_WINDOWS_SSF:
            # Forward through the encoder; SSF AE_classifier returns (decode, encode, logits)
            X_scaled = model._scale(X)
            x_t = model._to_tensor(X_scaled).to(model.device)
            model.model.eval()
            with torch.no_grad():
                out = model.model(x_t)
            # Extract the second tensor (encode) regardless of the exact tuple
            if isinstance(out, tuple) and len(out) >= 2:
                feat = out[1]
            else:
                feat = out
            feats_buf.append(feat.cpu().numpy())
            labels_buf.append(np.asarray(y))

    feats = np.concatenate(feats_buf, axis=0)
    labels = np.concatenate(labels_buf, axis=0)
    elapsed = time.time() - t0
    print(f'[SSF] done in {elapsed:.1f}s; embeddings: {feats.shape}, labels: {labels.shape}')
    return feats, labels


def tsne_2d(feats, labels, max_points=4000):
    """Subsample if too large (TSNE on 50k points takes ~10 min); apply t-SNE."""
    n = len(feats)
    if n > max_points:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n, size=max_points, replace=False)
        feats = feats[idx]
        labels = labels[idx]
        print(f'  subsampled to {max_points} points')
    perplexity = float(min(30, max(5, len(feats) - 1)))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                n_iter=750, init='pca', learning_rate='auto')
    print(f'  running t-SNE on {feats.shape} (perplexity={perplexity})...')
    t0 = time.time()
    z = tsne.fit_transform(feats)
    print(f'  t-SNE done in {time.time()-t0:.1f}s')
    return z, labels


def plot_tsne(z_b1, lab_b1, z_ssf, lab_ssf, out_pdf, out_png):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4))

    color_benign = '#4472C4'
    color_attack = '#C00000'

    def panel(ax, z, lab, title):
        is_attack = lab.astype(bool)
        ax.scatter(z[~is_attack, 0], z[~is_attack, 1],
                   s=4, color=color_benign, alpha=0.5, label='Benign',
                   edgecolors='none', rasterized=True)
        ax.scatter(z[is_attack, 0], z[is_attack, 1],
                   s=4, color=color_attack, alpha=0.7, label='Attack',
                   edgecolors='none', rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, markerscale=2.0)

    panel(axes[0], z_b1, lab_b1, '(a) IDA-SPADE')
    panel(axes[1], z_ssf, lab_ssf, '(b) SSF')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f'  Saved: {out_pdf}')
    print(f'  Saved: {out_png}')


def main():
    print('=' * 60)
    print('Regenerating Fig.5 (B1 vs SSF t-SNE on UNSW-NB15)')
    print('=' * 60)

    feats_b1, labels_b1 = prequential_collect_b1()
    feats_ssf, labels_ssf = prequential_collect_ssf()

    print('[t-SNE] B1 ...')
    z_b1, lab_b1 = tsne_2d(feats_b1, labels_b1)
    print('[t-SNE] SSF ...')
    z_ssf, lab_ssf = tsne_2d(feats_ssf, labels_ssf)

    print('[plot] rendering ...')
    plot_tsne(z_b1, lab_b1, z_ssf, lab_ssf,
              os.path.join(FIGURES_DIR, 't-SNE.pdf'),
              os.path.join(FIGURES_DIR, 't-SNE.png'))
    print('Done.')


if __name__ == '__main__':
    main()

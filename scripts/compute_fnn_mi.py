"""Compute FNN ratio at E in {1..6} and MI(X(t), X(t+tau)) for tau in {1..8}
on the per-entity ECBA-aggregated behavioral streams of NSL-KDD, UNSW-NB15, CIC-IDS-2017.

Strategy:
1. Load preprocessed CSV.
2. Assign entity = (proto, service) one-hot argmax (matches IDA-SPADE.py L283-300).
3. Pick the longest-lived entity per dataset.
4. Build T=1000-record windows per entity, compute transient-preserving aggregates
   (max, std, entropy) on a numeric feature column.
5. Compute FNN ratio at E=1..6 (Kennel et al. 1992, R_tol=10) and MI(X(t),X(t+tau)).
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd
from collections import defaultdict

T = 1000  # ECBA window size (matches paper Sec.IV-B)


def load_dataset(name):
    """Return (df, numeric_feature, proto_prefix, service_prefix)."""
    if name == 'NSL-KDD':
        path = r'C:/Users/Litsay/desktop/IDA-SPADE (TIFS)/IDA-SPADE/NSL_pre_data/PKDDTest+.csv'
        proto_prefix = 'protocol_type_'
        service_prefix = 'service_'
        feat = 'src_bytes' if True else 'duration'
    elif name == 'UNSW-NB15':
        path = r'C:/Users/Litsay/desktop/IDA-SPADE (TIFS)/IDA-SPADE/UNSW_pre_data/UNSWTest.csv'
        proto_prefix = 'proto_'
        service_prefix = 'service_'
        feat = 'sbytes'
    else:  # CIC-IDS-2017
        # CIC preprocessing varies; we use a typical column
        path = r'D:/CIC2017-IDS/MachineLearningCSV/Monday-WorkingHours.pcap_ISCX.csv'
        proto_prefix = ''
        service_prefix = ''
        feat = 'Total Length of Fwd Packets'
    return path, proto_prefix, service_prefix, feat


def assign_entities(df, proto_prefix, service_prefix):
    """Match IDA-SPADE.py _assign_entity_ids."""
    proto_cols = [c for c in df.columns if c.startswith(proto_prefix)] if proto_prefix else []
    serv_cols = [c for c in df.columns if c.startswith(service_prefix)] if service_prefix else []
    if proto_cols and serv_cols:
        proto_idx = df[proto_cols].values.argmax(axis=1)
        proto_names = np.array(proto_cols)[proto_idx]
        serv_idx = df[serv_cols].values.argmax(axis=1)
        serv_names = np.array(serv_cols)[serv_idx]
        return [f"{p}|{s}" for p, s in zip(proto_names, serv_names)]
    else:
        # CIC: no one-hot; group by 'Destination Port' if present
        if 'Destination Port' in df.columns:
            return df['Destination Port'].astype(str).tolist()
        return ['default'] * len(df)


def shannon_entropy(vals, n_bins=10):
    if len(vals) <= 1:
        return 0.0
    counts, _ = np.histogram(vals, bins=n_bins)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def ecba_streams(df, entities, feat, T):
    """Return dict {entity: per-window aggregated multidim vector list}.

    Each window's vector has 3 components: [max, std, entropy] of `feat`.
    """
    df = df.copy()
    df['_eid'] = entities
    df = df.reset_index(drop=True)
    df['_window'] = df.index // T
    out = defaultdict(list)
    for (eid, w), grp in df.groupby(['_eid', '_window']):
        if len(grp) < 2:
            continue
        x = grp[feat].dropna().values.astype(float)
        if len(x) < 2:
            continue
        out[eid].append([float(x.max()),
                          float(x.std()),
                          shannon_entropy(x)])
    return out


def fnn_ratio(scalar_series, E, tau=1, R_tol=10.0):
    """Kennel et al. 1992 False Nearest Neighbor ratio at embedding dim E."""
    s = np.asarray(scalar_series, dtype=float)
    s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
    N = len(s)
    if N < (E + 1) * tau + 5:
        return float('nan')
    # E-dim embedding
    M = N - E * tau
    XE = np.array([s[i + np.arange(E) * tau] for i in range(M)])
    XE1 = np.array([s[i + np.arange(E + 1) * tau] for i in range(M)])
    fn = 0
    valid = 0
    for i in range(M):
        # nearest-neighbor in E-dim
        d2 = np.sum((XE - XE[i]) ** 2, axis=1)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        d_E = math.sqrt(d2[j])
        if d_E < 1e-12:
            continue
        d_E1 = float(np.linalg.norm(XE1[i] - XE1[j]))
        # extra coordinate distance
        extra = math.sqrt(max(d_E1 ** 2 - d_E ** 2, 0.0))
        if (extra / d_E) > R_tol:
            fn += 1
        valid += 1
    return fn / max(valid, 1)


def mutual_information(x, tau, n_bins=16):
    """MI(X(t), X(t+tau)) using histogram estimator (in bits)."""
    s = np.asarray(x, dtype=float)
    s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
    a = s[:-tau]
    b = s[tau:]
    if len(a) < 10:
        return float('nan')
    H, _, _ = np.histogram2d(a, b, bins=n_bins, range=[[0, 1], [0, 1]])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if Pxy[i, j] > 0 and Px[i] > 0 and Py[j] > 0:
                mi += Pxy[i, j] * math.log2(Pxy[i, j] / (Px[i] * Py[j]))
    return float(mi)


def analyse(name):
    path, pp, sp, feat = load_dataset(name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: {path} not found")
        return None
    print(f"\n=== {name} ===")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if feat not in df.columns:
        # try a fallback numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat = num_cols[0] if num_cols else None
        if feat is None:
            print(f"  [SKIP] no numeric column in {name}")
            return None
        print(f"  fallback feat={feat}")
    print(f"  rows={len(df)}, feature='{feat}', proto_pref='{pp}', srv_pref='{sp}'")
    entities = assign_entities(df, pp, sp)
    streams = ecba_streams(df, entities, feat, T)
    if not streams:
        print(f"  [SKIP] no valid entity stream")
        return None
    # Pick top-N longest streams (more robust than a single entity)
    eids_sorted = sorted(streams.keys(), key=lambda e: -len(streams[e]))
    top = [(e, np.array(streams[e])) for e in eids_sorted if len(streams[e]) >= 25][:3]
    if not top:
        # fallback to longest, even if K<25
        top = [(eids_sorted[0], np.array(streams[eids_sorted[0]]))]
    print(f"  top entities (K windows): " +
          ", ".join(f"{e!r}={len(s)}" for e, s in top))

    res = {'dataset': name, 'top_entities': [(e, len(s)) for e, s in top],
           'feature': feat, 'fnn': {}, 'mi': {}}
    Es = list(range(1, 7))
    print("  FNN ratios (R_tol=10), median across 3 components x N entities:")
    for E in Es:
        rs = []
        for _, seq in top:
            for c in range(3):
                v = fnn_ratio(seq[:, c], E=E, tau=1)
                if not math.isnan(v):
                    rs.append(v)
        if rs:
            med = float(np.median(rs))
            mean = float(np.mean(rs))
            res['fnn'][E] = {'median': med, 'mean': mean, 'n': len(rs)}
            print(f"    E={E}: median={med*100:5.2f}%  mean={mean*100:5.2f}%  (n={len(rs)})")
    print("  MI(X(t), X(t+tau)) bits, median across components x entities:")
    for tau in range(1, 9):
        rs = []
        for _, seq in top:
            if len(seq) <= tau:
                continue
            for c in range(3):
                v = mutual_information(seq[:, c], tau=tau)
                if not math.isnan(v):
                    rs.append(v)
        if rs:
            med = float(np.median(rs))
            mean = float(np.mean(rs))
            res['mi'][tau] = {'median': med, 'mean': mean, 'n': len(rs)}
            print(f"    tau={tau}: median={med:5.3f}  mean={mean:5.3f} bits (n={len(rs)})")
    return res


def main():
    out = {}
    for name in ['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017']:
        try:
            r = analyse(name)
            if r:
                out[name] = r
        except Exception as e:
            print(f"  ERROR for {name}: {e}")
    save_path = r'C:/Users/Litsay/desktop/IDA-SPADE (TIFS)/IDA-SPADE/experiment_results/fnn_mi_diagnostics.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()

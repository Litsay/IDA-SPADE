"""CIC-IDS-2017 Per-Day Phase 1 Experiment.

Following unFlowS (TIFS 2025) protocol:
- Train on Monday (benign only)
- Test separately on Tuesday, Wednesday, Thursday, Friday
- Report per-day metrics for each model

Usage:
    /c/Users/Litsay/anaconda3/envs/CL/python.exe -m experiments.run_cic_perday
"""
import sys
import os
import time
import json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from experiments.config import (
    SEED, RESULTS_DIR, WINDOW_SIZE, SLIDE_SIZE, FADING_FACTOR
)
from experiments.streaming_interface import Window
from experiments.evaluator import prequential_evaluate, aggregate_metrics
from experiments.run_all import set_seed, create_models


CIC_DATA_DIR = r'D:\CIC2017-IDS\MachineLearningCSV'

DAY_FILES = {
    'Monday': ['Monday-WorkingHours.pcap_ISCX.csv'],
    'Tuesday': ['Tuesday-WorkingHours.pcap_ISCX.csv'],
    'Wednesday': ['Wednesday-workingHours.pcap_ISCX.csv'],
    'Thursday': [
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    ],
    'Friday': [
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    ],
}

DAY_ATTACKS = {
    'Tuesday': 'FTP-Patator, SSH-Patator',
    'Wednesday': 'DoS (Hulk, GoldenEye, slowloris, Slowhttptest), Heartbleed',
    'Thursday': 'Web Attacks (BF, XSS, SQLi), Infiltration',
    'Friday': 'Bot, PortScan, DDoS',
}


def _add_entity_columns(df):
    """Add synthetic protocol/service one-hot columns from Destination Port."""
    port_col = 'Destination Port'
    if port_col not in df.columns:
        for c in df.columns:
            if 'destination' in c.lower() and 'port' in c.lower():
                port_col = c
                break
    if port_col in df.columns:
        port = pd.to_numeric(df[port_col], errors='coerce').fillna(0).astype(int)
        df['cic_service_web'] = port.isin([80, 443, 8080, 8443]).astype(int)
        df['cic_service_ssh'] = port.isin([22]).astype(int)
        df['cic_service_ftp'] = port.isin([20, 21]).astype(int)
        df['cic_service_dns'] = port.isin([53]).astype(int)
        df['cic_service_mail'] = port.isin([25, 110, 143, 993, 995]).astype(int)
        df['cic_service_other'] = (~port.isin(
            [80, 443, 8080, 8443, 22, 20, 21, 53, 25, 110, 143, 993, 995])).astype(int)
        df['cic_proto_system'] = (port < 1024).astype(int)
        df['cic_proto_ephemeral'] = (port >= 1024).astype(int)
    return df


def load_day(day_name):
    """Load and concatenate all CSV files for a given day."""
    files = DAY_FILES[day_name]
    dfs = []
    for fname in files:
        path = os.path.join(CIC_DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV files found for {day_name}")
    df = pd.concat(dfs, ignore_index=True)

    # Label
    label_col = 'Label'
    if label_col not in df.columns:
        for c in df.columns:
            if 'label' in c.lower():
                label_col = c
                break
    df['label_binary'] = (df[label_col].str.strip() != 'BENIGN').astype(int)
    df['attack_class'] = df[label_col].str.strip()

    # Entity columns
    df = _add_entity_columns(df)

    # Numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label_binary']]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, num_cols


def create_windows(df, feature_cols, window_size=WINDOW_SIZE, slide_size=SLIDE_SIZE):
    """Create streaming windows from a DataFrame."""
    windows = []
    n = len(df)
    wid = 0
    pos = 0
    while pos + window_size <= n:
        chunk = df.iloc[pos:pos + window_size]
        common_cols = [c for c in feature_cols if c in chunk.columns]
        X = chunk[common_cols].values.astype(np.float32)
        if X.shape[1] < len(feature_cols):
            pad = np.zeros((X.shape[0], len(feature_cols) - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        y_bin = chunk['label_binary'].values.astype(np.int64)
        y_mc = chunk['attack_class'].values
        meta = {
            'anomaly_ratio': float(y_bin.mean()),
            'size': len(chunk),
        }
        windows.append(Window(index=wid, X=X, y_binary=y_bin, y_multiclass=y_mc, metadata=meta))
        pos += slide_size
        wid += 1
    return windows


def main():
    set_seed(SEED)
    print("=" * 80)
    print("CIC-IDS-2017 Per-Day Phase 1 (unFlowS protocol)")
    print("Train: Monday (benign) | Test: Tue/Wed/Thu/Fri separately")
    print("=" * 80)

    # Load Monday for init
    print("\nLoading Monday (init)...")
    df_mon, feature_cols = load_day('Monday')
    if len(df_mon) > 50000:
        df_mon = df_mon.sample(n=50000, random_state=42).reset_index(drop=True)
    X_init = df_mon[feature_cols].values.astype(np.float32)
    y_init = df_mon['label_binary'].values.astype(np.int64)
    print(f"  Init: {X_init.shape}, features: {len(feature_cols)}")

    # Test days
    test_days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday']
    all_results = {}

    # Exclude AOC-IDS (too slow)
    exclude_models = ['AOC-IDS']

    for day in test_days:
        print(f"\n{'='*60}")
        print(f"Testing on {day} — {DAY_ATTACKS[day]}")
        print(f"{'='*60}")

        df_day, _ = load_day(day)
        windows = create_windows(df_day, feature_cols)
        anomaly_ratio = df_day['label_binary'].mean()
        print(f"  Rows: {len(df_day)}, Windows: {len(windows)}, Anomaly ratio: {anomaly_ratio:.3f}")

        models = create_models(
            feature_cols=feature_cols, dataset_name='CIC-IDS-2017',
            exclude=exclude_models)

        day_results = {}
        for model_name, model in models.items():
            set_seed(SEED)
            t0 = time.time()
            results = prequential_evaluate(
                model, iter(windows), X_init, y_init,
                alpha=FADING_FACTOR, verbose=False)
            elapsed = time.time() - t0
            agg = aggregate_metrics(results)
            print(f"  {model_name:<20s} F1={agg.get('f1',0)*100:6.2f}% | "
                  f"Acc={agg.get('accuracy',0)*100:6.2f}% | "
                  f"Pre={agg.get('precision',0)*100:6.2f}% | "
                  f"Rec={agg.get('recall',0)*100:6.2f}% | "
                  f"AUC={agg.get('auc',0)*100:6.2f}% | "
                  f"Time={elapsed:.1f}s")
            day_results[model_name] = agg

        all_results[day] = day_results

    # Export results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'cic_perday_phase1.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Export markdown
    md_path = os.path.join(RESULTS_DIR, 'cic_perday_phase1.md')
    lines = ["# CIC-IDS-2017 Per-Day Phase 1 Results\n",
             "Protocol: Train on Monday (benign), test per-day separately.\n"]

    model_order = ['IDA-SPADE', 'SSF', 'EWC', 'LwF', 'CIDS', 'CARD', 'FeCo', 'unFlowS']

    for day in test_days:
        lines.append(f"\n## {day} — {DAY_ATTACKS[day]}\n")
        lines.append("| Method | Acc.(%) | Pre.(%) | Rec.(%) | F1(%) | AUC(%) |")
        lines.append("|---|---|---|---|---|---|")
        for m in model_order:
            if m in all_results[day]:
                d = all_results[day][m]
                lines.append(
                    f"| {m} | {d['accuracy']*100:.2f} | {d['precision']*100:.2f} | "
                    f"{d['recall']*100:.2f} | {d['f1']*100:.2f} | {d.get('auc',0)*100:.2f} |")
        lines.append("")

    # Summary table: F1 per day per model
    lines.append("\n## Summary: F1(%) per Day\n")
    lines.append("| Method | Tuesday | Wednesday | Thursday | Friday |")
    lines.append("|---|---|---|---|---|")
    for m in model_order:
        row = [m]
        for day in test_days:
            if m in all_results[day]:
                row.append(f"{all_results[day][m]['f1']*100:.2f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Summary table: AUC per day per model
    lines.append("\n## Summary: AUC(%) per Day\n")
    lines.append("| Method | Tuesday | Wednesday | Thursday | Friday |")
    lines.append("|---|---|---|---|---|")
    for m in model_order:
        row = [m]
        for day in test_days:
            if m in all_results[day]:
                row.append(f"{all_results[day][m].get('auc',0)*100:.2f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nResults saved to {output_path}")
    print(f"Markdown saved to {md_path}")


if __name__ == '__main__':
    main()

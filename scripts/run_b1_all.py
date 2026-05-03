"""Master orchestrator for the full B1 experiment campaign.

Runs (in this order, sequentially):
    1. Tab.6 runtime profile      (~5 min)
    2. Tab.1 main results         (~90 min: 10/10/5 seeds x 3 datasets)
    3. Tab.5 sensitivity sweep    (~70 min: 12 grid x 3 seeds)
    4. Tab.4 ablation matrix      (~16 hours: 7 variants x 25 seeds)
    Tab.3 drift-period is already complete from the prior smoke run.

Each phase has its own resume support, so interrupting and re-running
this script picks up where it left off.

Output JSONs (in experiment_results/)
    b1_tab6_runtime.json          (phase 1)
    b1_tab1_main.json             (phase 2)
    b1_tab5_sensitivity.json      (phase 3)
    b1_tab4_ablation.json         (phase 4)

Usage
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_all.py
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_all.py --phases 1 2
    /c/Users/Litsay/anaconda3/envs/CL/python.exe run_b1_all.py --skip 4   # everything except Tab.4
"""
import sys
import os
import argparse
import subprocess
import datetime as dt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\Litsay\anaconda3\envs\CL\python.exe'

PHASES = [
    {
        'id': 1,
        'name': 'Tab.6 runtime',
        'script': 'run_b1_tab6.py',
        'eta_min': 5,
    },
    {
        'id': 2,
        'name': 'Tab.1 main',
        'script': 'run_b1_tab1.py',
        'eta_min': 90,
    },
    {
        'id': 3,
        'name': 'Tab.5 sensitivity',
        'script': 'run_b1_tab5.py',
        'eta_min': 70,
    },
    {
        'id': 4,
        'name': 'Tab.4 ablation',
        'script': 'run_b1_tab4.py',
        'eta_min': 960,  # ~16 hours
    },
]

LOG_PATH = os.path.join('experiment_results', 'b1_all.log')


def log(msg):
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [orchestrator] {msg}'
    print(line, flush=True)
    try:
        os.makedirs('experiment_results', exist_ok=True)
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def run_phase(phase):
    log(f'>>> START phase {phase["id"]}: {phase["name"]} (ETA ~{phase["eta_min"]} min)')
    cmd = [PY, os.path.join(THIS_DIR, phase['script'])]
    log(f'    CMD: {" ".join(cmd)}')
    rc = subprocess.call(cmd, cwd=THIS_DIR)
    if rc == 0:
        log(f'<<< DONE phase {phase["id"]}: {phase["name"]} (rc=0)')
    else:
        log(f'<<< FAIL phase {phase["id"]}: {phase["name"]} (rc={rc})')
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phases', nargs='+', type=int,
                        default=[p['id'] for p in PHASES],
                        choices=[p['id'] for p in PHASES])
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                        choices=[p['id'] for p in PHASES])
    args = parser.parse_args()

    selected = [p for p in PHASES if p['id'] in args.phases and p['id'] not in args.skip]

    log('=' * 60)
    log(f'B1 master orchestrator starting')
    log(f'  selected phases: {[p["id"] for p in selected]}')
    log(f'  total ETA      : ~{sum(p["eta_min"] for p in selected)} min')
    log('=' * 60)

    failures = []
    for p in selected:
        rc = run_phase(p)
        if rc != 0:
            failures.append(p['id'])
            log(f'    -> continuing despite phase {p["id"]} failure')

    log('=' * 60)
    if failures:
        log(f'B1 orchestrator DONE with failures in phases: {failures}')
        sys.exit(1)
    else:
        log(f'B1 orchestrator DONE successfully')


if __name__ == '__main__':
    main()

"""Merge evaluate.py runs that differ only in their seeds (e.g. run in batches with run.tag=_s012 / _s34)
into one results.csv, and print the same summary table as evaluate.py.

    python scripts/merge_runs.py logs/<run>_s012 logs/<run>_s34 --out logs/<run>
-> <out>/results.csv and <out>/eval_summary.log  (per_ood_table.py can then be run on <out>)
"""
import os
import csv
import argparse

import numpy as np

COLS = ['MHKSA', 'MHKSA_e1only', 'MHKSA_e2only', 'best_head_oracle', 'best_head_loo', 'CoRP_Ens_percentile', 'CoRP_Ens_zscore']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs', nargs='+', help='logs/<run> folders, each with results.csv')
    ap.add_argument('--out', required=True, help='folder for the merged results.csv')
    a = ap.parse_args()

    header, rows, seen = None, [], {}
    for run in a.runs:
        with open(os.path.join(run, 'results.csv')) as f:
            r = csv.reader(f)
            header = next(r)
            part = list(r)
        for s in {x[0] for x in part}:
            assert s not in seen, f'seed {s} is in both {seen[s]} and {run}'
            seen[s] = run
        rows += part
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    Hs = sorted({int(x[1]) for x in rows})
    L = [f'merged {a.runs} -> seeds {sorted(seen, key=int)}', '',
         '==== summary: AVG over OOD sets, mean±std over seeds (FPR95 / AUROC) ====',
         f"{'H':>3s} " + ' '.join(f'{m[:20]:>22s}' for m in COLS)]
    for H in Hs:
        cells = []
        for m in COLS:
            v = np.array([[float(x[6]), float(x[7])] for x in rows if int(x[1]) == H and x[2] == m and x[5] == 'AVG'])
            cells.append(f'{v[:, 0].mean():5.2f}±{v[:, 0].std():4.2f}/{v[:, 1].mean():5.2f}' if len(v) else '-')
        L.append(f'{H:>3d} ' + ' '.join(f'{c:>22s}' for c in cells))
    base = np.array([float(x[6]) for x in rows if int(x[1]) == 1 and x[2] == 'single_head' and x[5] == 'AVG'])
    if len(base):
        L.append(f'baseline (H=1 single head): FPR95 {base.mean():.2f}±{base.std():.2f}')
    L.append(f"\nCSV -> {os.path.join(a.out, 'results.csv')}")
    open(os.path.join(a.out, 'eval_summary.log'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

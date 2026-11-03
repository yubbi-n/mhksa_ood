"""KPCA vs Ours per OOD set, from one evaluate.py run (its results.csv) -> one Markdown table.

    python scripts/per_ood_table.py logs/ImageNet-R50-ce-evr0.5-mh0.71_0.71-M4096-m0.3_10log \
        --ref "KPCA, paper (γ=3)" 50.07 62.56 72.76 9.02 --ref-auroc 87.53

Rows: KPCA = single head at H = 1 of the run (its base gamma, e.g. median), --ref rows (fixed numbers, e.g. the paper);
Ours = MHKSA and ensemble z-score at --h; best head (oracle, test-picked) as a reference.
FPR95 per OOD set = mean over seeds; Avg = mean over OOD sets, ± std over seeds.
-> results/markdown/per_ood_<run folder>_H<h>.md
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np

ROWS = [('KPCA', 'single_head', 1, 'median γ (H = 1)'),
        ('Ours', 'MHKSA', None, 'MHKSA'),
        ('Ours', 'CoRP_Ens_zscore', None, 'Ensemble (z-score)'),
        ('ref', 'best_head_oracle', None, 'best head (oracle, test-picked)')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', help='logs/<run> folder with results.csv')
    ap.add_argument('--h', type=int, default=10, help='H of Ours')
    ap.add_argument('--ref', nargs='+', action='append', default=[], metavar=('LABEL', 'FPR'),
                    help='fixed KPCA row: label then FPR95 per OOD set (in the run\'s OOD order); repeatable')
    ap.add_argument('--ref-auroc', nargs='*', type=float, default=[], help='Avg AUROC of each --ref row (same order)')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()

    v = defaultdict(list)                      # (method, H, dataset) -> [(seed, FPR, AUROC)]
    oods = []
    with open(os.path.join(a.run, 'results.csv')) as f:
        for r in csv.DictReader(f):
            v[(r['method'], int(r['H']), r['dataset'])].append((float(r['FPR95']), float(r['AUROC'])))
            if r['dataset'] != 'AVG' and r['dataset'] not in oods:
                oods.append(r['dataset'])

    L = [f'# KPCA vs Ours per OOD set — {os.path.basename(os.path.normpath(a.run))}', '',
         f'FPR95 (%) ↓ per OOD set (mean over seeds); Avg = mean over OOD sets ± std over seeds; AUROC (%) ↑. Ours at H = {a.h}.', '',
         '| | method | ' + ' | '.join(oods) + ' | Avg FPR95 ↓ | Avg AUROC ↑ |', '| --- | --- |' + ' --- |' * (len(oods) + 2)]

    def line(group, label, per, avg, au):
        return f'| {group} | {label} | ' + ' | '.join(f'{x:.2f}' for x in per) + f' | {avg} | {au} |'

    out = [line('KPCA', ROWS[0][3], *stats(v, *ROWS[0][1:3], oods))]
    for i, ref in enumerate(a.ref):
        fpr = [float(x) for x in ref[1:]]
        assert len(fpr) == len(oods), f'--ref {ref[0]!r}: {len(fpr)} values for {len(oods)} OOD sets {oods}'
        au = f'{a.ref_auroc[i]:.2f}' if i < len(a.ref_auroc) else '-'
        out.append(line('', ref[0], fpr, f'{np.mean(fpr):.2f}', au))
    prev = 'KPCA'
    for g, m, _, lab in ROWS[1:]:
        out.append(line(g if g != prev else '', f'{lab} (H = {a.h})', *stats(v, m, a.h, oods)))
        prev = g
    L += out
    md = os.path.join(a.out, 'markdown', f'per_ood_{os.path.basename(os.path.normpath(a.run))}_H{a.h}.md')
    os.makedirs(os.path.dirname(md), exist_ok=True)
    open(md, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', md)


def stats(v, method, H, oods):
    per = [np.mean([x[0] for x in v[(method, H, o)]]) for o in oods]
    avg = np.array(v[(method, H, 'AVG')])
    return per, f'{avg[:, 0].mean():.2f} ± {avg[:, 0].std():.2f}', f'{avg[:, 1].mean():.2f}'


if __name__ == '__main__':
    main()

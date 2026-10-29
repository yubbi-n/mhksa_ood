"""Head range x PCA ratio sweep at H = 10, without checkpoints (nothing is saved but a CSV and a table).

For every (seed, range) the H = 10 heads are rebuilt exactly as train.py does (same median gamma, same RFF draws),
each head's PCA is fitted once with its full spectrum kept, and every ratio is read off that spectrum:
    r1  stage-1 / per-head PCA ratio: the CoRP heads of the ensemble and of the single-head KPCA, and MHKSA stage 1
    r2  MHKSA stage-2 ratio (1.0 = no stage 2); stage 2 is refitted on the ID-train coords for every r1
Methods: MHKSA [2] (r1, r2), ensemble z-score [3] (r1), single-head KPCA at the median gamma (r1, H = 1).
r1 = mhksa_evr1, r2 = mhksa_evr2 is the checkpoint MHKSA, r1 = exp_var_ratio is the checkpoint ensemble / baseline.

    python scripts/sweep.py --mode ce --cache <feature cache> [--config configs/c100_r18_ce.yaml] [--mult-max 2 3 5 7 10] [--r1 0.6 0.7 0.8 0.9 0.95]
                            [--r2 0.9 0.95 0.99 1.0] [--seeds 0 1 2 3 4] [--gamma-base 2]
-> logs/sweep/sweep_<mode>_<median|gbX>_<tag>.csv (per seed, per OOD set) and results/markdown/sweep_<mode>_<base>_<tag>.md
   (<tag> = the ranges and ratios swept, e.g. m2-3-5-7-10_r1-0.6-0.7-0.8-0.9-0.95_r2-0.9-0.95-0.99-1, or --tag)

The best cell of this grid is picked on the test OOD sets: it is an analysis (where the good region is), not a
result of the method. Use it as a fixed setting on a dataset it was not picked on (e.g. ImageNet).
"""
import os
import sys
import csv
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, for src/
import numpy as np
from src.config import load_config
from src.data import load_eval, load_id_train, split_calib
from src.kernels import gamma_schedule, median_heuristic
from src.model import MultiHead, PCA
from src.metrics import evaluate_all
from src.utils import set_seed


def heads_for(cfg, gammas, H, seed, dim):
    """the RFF maps train.py draws for this (seed, H): same RNG, same order."""
    m = cfg.model
    return MultiHead(gammas, cfg.kernel.M, m.exp_var_ratio, m.mhksa_evr1, dim, np.random.RandomState([seed, H])).heads


def fit_heads(heads, x_fit, sets, r_max):
    """per head: PCA kept with spectrum; coords at the largest ratio and squared norms, for train and every set."""
    out = []
    for h in heads:
        f = h.rff(x_fit)
        p = PCA(r_max).fit(f)
        U = p.U                                               # (M, q_max)
        xc = f - p.mu
        tr = (xc @ U, np.square(xc, dtype=np.float64).sum(1))
        ev = {}
        for n, x in sets.items():
            g = h.rff(x) - p.mu
            ev[n] = (g @ U, np.square(g, dtype=np.float64).sum(1))
        out.append({'pca': p, 'train': tr, 'eval': ev})
    return out


def q_of(fh, r):
    return min(fh['pca'].with_ratio(r).q, fh['pca'].q)


def resid(c, n2, q):
    return np.sqrt(np.maximum(n2 - np.square(c[:, :q], dtype=np.float64).sum(1), 0))


def mhksa_r2(fit, r1, r2s, names):
    """MHKSA scores for one r1 and every r2: stage 2 = PCA of the concatenated stage-1 coords of ID train."""
    qs = [q_of(fh, r1) for fh in fit]
    e1 = {n: sum(resid(fh['eval'][n][0], fh['eval'][n][1], q) ** 2 for fh, q in zip(fit, qs)) for n in names}
    Z = np.concatenate([fh['train'][0][:, :q] for fh, q in zip(fit, qs)], 1)
    p2 = PCA(min(r2s)).fit(Z)
    Zev = {n: np.concatenate([fh['eval'][n][0][:, :q] for fh, q in zip(fit, qs)], 1) for n in names}
    out = {}
    for r2 in r2s:
        if r2 >= 1.0:
            out[r2] = {n: -np.sqrt(e1[n]) for n in names}
            continue
        pr = p2.with_ratio(r2)
        out[r2] = {n: -np.sqrt(e1[n] + pr.transform(Zev[n])[1].astype(np.float64) ** 2) for n in names}
    return out


def ens_z(fit, r1, names):
    qs = [q_of(fh, r1) for fh in fit]
    ref = [resid(fh['train'][0], fh['train'][1], q) for fh, q in zip(fit, qs)]
    mu = np.array([r.mean() for r in ref])
    sd = np.array([r.std() for r in ref]) + 1e-10
    return {n: -np.mean([(resid(fh['eval'][n][0], fh['eval'][n][1], q) - m) / s
                         for fh, q, m, s in zip(fit, qs, mu, sd)], 0) for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', required=True, choices=['ce', 'supcon'])
    ap.add_argument('--config', default=None, help='config file (default configs/c10_r18_<mode>.yaml), e.g. configs/c100_r18_ce.yaml')
    ap.add_argument('--cache', required=True, help='feature cache dir (data.cache_dir)')
    ap.add_argument('--gamma-base', default='', help='fixed base gamma (kernel.gamma_base) instead of the median')
    ap.add_argument('--mult-max', nargs='+', type=float, default=[2, 3, 5, 7, 10])
    ap.add_argument('--r1', nargs='+', type=float, default=[0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument('--r2', nargs='+', type=float, default=[0.9, 0.95, 0.99, 1.0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--H', type=int, default=10)
    ap.add_argument('--out', default='results')
    ap.add_argument('--tag', default=None, help='file-name suffix (default: from the ranges and ratios, so runs do not overwrite each other)')
    a = ap.parse_args()
    sets_ = [f'data.cache_dir={a.cache}'] + ([f'kernel.gamma_base={a.gamma_base}'] if a.gamma_base else [])
    cfg = load_config(a.config or f'configs/c10_r18_{a.mode}.yaml', sets_)
    base = f'gb{a.gamma_base}' if a.gamma_base else 'median'
    if cfg.data.in_data != 'CIFAR10':                  # other datasets get their own file names
        base = f'{cfg.data.in_data.lower()}_{base}'
    f_in, f_out = load_eval(cfg)
    sets = {'ID': f_in, **f_out}
    oods = list(f_out)
    x_all = load_id_train(cfg)
    dim = x_all.shape[1]
    r_max = max(a.r1)
    os.makedirs('logs/sweep', exist_ok=True)
    g = lambda v: '-'.join(f'{x:g}' for x in v)
    tag = a.tag or f'm{g(a.mult_max)}_r1-{g(a.r1)}_r2-{g(a.r2)}'
    csv_path = f'logs/sweep/sweep_{a.mode}_{base}_{tag}.csv'
    w = csv.writer(open(csv_path, 'w', newline=''))
    w.writerow(['mode', 'base', 'mult_max', 'seed', 'method', 'r1', 'r2', 'dataset', 'FPR95', 'AUROC'])
    rows = []

    def record(mm, seed, meth, r1, r2, s):
        per, avg = evaluate_all(s['ID'], {n: s[n] for n in oods})
        for n, v in list(per.items()) + [('AVG', avg)]:
            w.writerow([a.mode, base, mm, seed, meth, r1, r2, n, f'{v["FPR"]:.4f}', f'{v["AUROC"]:.4f}'])
        rows.append((mm, seed, meth, r1, r2, avg['FPR'], avg['AUROC']))

    for seed in a.seeds:
        x_fit, _ = split_calib(x_all, cfg.data.calib_holdout, set_seed(seed))       # as train.py
        g_med, _ = median_heuristic(x_fit, cfg.kernel.median_subsample, cfg.kernel.median_mode,
                                    np.random.RandomState([seed, 0]))
        if cfg.kernel.get('gamma_base') is not None:
            g_med = float(cfg.kernel.gamma_base)
        one = fit_heads(heads_for(cfg, [g_med], 1, seed, dim), x_fit, sets, r_max)[0]
        for r1 in a.r1:                                                                 # H = 1 baseline per r1
            q = q_of(one, r1)
            record(1, seed, 'KPCA_H1', r1, '', {n: -resid(*one['eval'][n], q) for n in sets})
        for mm in a.mult_max:
            cfg.kernel.mult_max = mm
            fit = fit_heads(heads_for(cfg, gamma_schedule(a.H, cfg.kernel, g_med), a.H, seed, dim), x_fit, sets, r_max)
            for r1 in a.r1:
                record(mm, seed, 'Ens_zscore', r1, '', ens_z(fit, r1, list(sets)))
                for r2, s in mhksa_r2(fit, r1, a.r2, list(sets)).items():
                    record(mm, seed, 'MHKSA', r1, r2, s)
            print(f'[{a.mode} seed {seed}] x[{cfg.kernel.mult_min:g}, {mm:g}] done', flush=True)
    print('saved', csv_path)

    # mean over seeds -> Markdown tables (FPR95, AVG over OOD sets)
    def mean(mm, meth, r1, r2):
        v = [r[5] for r in rows if r[0] == mm and r[2] == meth and r[3] == r1 and r[4] == r2]
        return np.mean(v) if v else np.nan
    L = [f'# Range x PCA ratio sweep: {a.mode}, base {base}, H = {a.H}, seeds {a.seeds}', '',
         'FPR95 (AVG over OOD sets, mean over seeds). Best cell is picked on the test OOD sets: analysis only.', '']
    L += ['## single-head KPCA (H = 1, base gamma) per r1', '', '| r1 | ' + ' | '.join(f'{r:g}' for r in a.r1) + ' |',
          '| --- |' + ' --- |' * len(a.r1), '| FPR95 | ' + ' | '.join(f'{mean(1, "KPCA_H1", r, ""):.2f}' for r in a.r1) + ' |', '']
    L += ['## ensemble z-score (rows: range, columns: r1)', '', '| range | ' + ' | '.join(f'r1 {r:g}' for r in a.r1) + ' |',
          '| --- |' + ' --- |' * len(a.r1)]
    L += [f'| x{mm:g} | ' + ' | '.join(f'{mean(mm, "Ens_zscore", r, ""):.2f}' for r in a.r1) + ' |' for mm in a.mult_max]
    for r2 in a.r2:
        L += ['', f'## MHKSA, r2 = {r2:g} (rows: range, columns: r1)', '',
              '| range | ' + ' | '.join(f'r1 {r:g}' for r in a.r1) + ' |', '| --- |' + ' --- |' * len(a.r1)]
        L += [f'| x{mm:g} | ' + ' | '.join(f'{mean(mm, "MHKSA", r, r2):.2f}' for r in a.r1) + ' |' for mm in a.mult_max]
    cells = [(mean(mm, 'MHKSA', r1, r2), f'MHKSA x{mm:g} r1 {r1:g} r2 {r2:g}') for mm in a.mult_max for r1 in a.r1 for r2 in a.r2]
    cells += [(mean(mm, 'Ens_zscore', r1, ''), f'Ens z x{mm:g} r1 {r1:g}') for mm in a.mult_max for r1 in a.r1]
    best = sorted(c for c in cells if not np.isnan(c[0]))[:5]
    L += ['', '## lowest 5 cells (test-picked, analysis only)', ''] + [f'- {n}: {v:.2f}' for v, n in best]
    md = os.path.join(a.out, 'markdown', f'sweep_{a.mode}_{base}_{tag}.md')
    os.makedirs(os.path.dirname(md), exist_ok=True)
    open(md, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', md)


if __name__ == '__main__':
    main()

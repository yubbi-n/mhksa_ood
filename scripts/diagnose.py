"""Checkpoint-based analyses of the heads (largest-H checkpoints; results as Markdown on stdout).

    python scripts/diagnose.py heads     --mode ce --cache <feature cache> [--gamma-base 2] [--set kernel.mult_max=10] [--seeds 0 1 2 3 4]
    python scripts/diagnose.py influence --mode ce --cache <feature cache> ...
(the old environment variables still work as defaults: MODE, C, GB, EXTRA, SEEDS)
Output: printed and saved to <out>/markdown/<heads|influence>_<mode>_<median|gb..>_m<min>-<max>.md (default <out> = results).

heads      why combining heads does not beat the best head
             ① FPR95 of every single head, per OOD set
             ③ the same heads under raw-sum (MHKSA stage 1) / z-score mean (ensemble) / z-score max
             ② oracle pair and greedy-subset search on the test OOD sets (upper bound of what a z-mean subset can reach),
                head score correlations, and the ID tail of the max rule
influence  how much the bad heads weigh in the combined score
             ① per head: share of the score on ID (weight), of the OOD-ID gap (signal), of the ID variance (noise)
             ② FPR95 change when one head is left out (negative = removing it helps = harmful head)
             ③ all heads vs the better / worse half by single-head FPR
           rules: "raw" = MHKSA stage-1 residuals squared and summed (stage 2 would need refitting), "zmean" = ensemble [3]
Both are analyses: the oracle / halves are picked with the test OOD sets and are not methods.
"""
import os
import sys
import argparse
import itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, for src/
import numpy as np
from src.config import load_config, run_name
from src.data import load_eval
from src.utils import ckpt_path, load_pickle
from src.metrics import evaluate_all

def heads(cfg, name, H, f_in, f_out, oods, SEEDS):
    def fpr(s_in, s_out):
        per, avg = evaluate_all(s_in, s_out)
        return np.array([per[n]['FPR'] for n in oods] + [avg['FPR']])     # 5 sets + AVG


    R = {k: [] for k in ['single', 'zmean', 'zmax', 'raw', 'pairs', 'greedy', 'corr_in', 'corr_out', 'idtail']}
    for seed in SEEDS:
        m = load_pickle(ckpt_path(cfg, name, seed, H))['model']
        g = m.stage1.gammas
        mu, sd = m.ens.ref_mu[:, None], m.ens.ref_sd[:, None]
        Ein = m.stage1.transform(f_in)['corp'][1]
        Eout = {n: m.stage1.transform(x)['corp'][1] for n, x in f_out.items()}
        Zin, Zout = (Ein - mu) / sd, {n: (e - mu) / sd for n, e in Eout.items()}

        def comb(idx, rule='zmean'):
            f = {'zmean': lambda Z: -Z[idx].mean(0), 'zmax': lambda Z: -Z[idx].max(0),
                 'raw': lambda Z: None}[rule]
            if rule == 'raw':                                  # MHKSA-like: raw squared residuals summed
                return fpr(-np.sqrt((Ein[idx] ** 2).sum(0)), {n: -np.sqrt((e[idx] ** 2).sum(0)) for n, e in Eout.items()})
            return fpr(f(Zin), {n: f(z) for n, z in Zout.items()})

        allh = list(range(H))
        R['single'].append(np.stack([comb([h]) for h in allh]))
        R['zmean'].append(comb(allh)); R['zmax'].append(comb(allh, 'zmax')); R['raw'].append(comb(allh, 'raw'))
        R['pairs'].append({p: comb(list(p)) for p in itertools.combinations(allh, 2)})
        # greedy forward selection on test AVG FPR (oracle: shows the best a z-mean subset can reach)
        chosen, path = [], []
        for _ in range(H):
            best = min((h for h in allh if h not in chosen), key=lambda h: comb(chosen + [h])[-1])
            chosen.append(best); path.append((list(chosen), comb(chosen)))
        R['greedy'].append(path)
        # score correlation between heads (Spearman via ranks), ID test and pooled OOD
        rk = lambda Z: np.argsort(np.argsort(Z, 1), 1).astype(float)
        R['corr_in'].append(np.corrcoef(rk(Zin)))
        R['corr_out'].append(np.corrcoef(rk(np.concatenate(list(Zout.values()), 1))))
        # ID tail: fraction of ID test with z > 2 for one head vs for the max over heads
        R['idtail'].append(((Zin > 2).mean(1), (Zin.max(0) > 2).mean()))

    hdr = '| ' + ' | '.join(oods + ['AVG']) + ' |'
    sep = '| --- ' * (len(oods) + 2) + '|'
    fmt = lambda v: ' | '.join(f'{x:.2f}' for x in v)
    mean = lambda L: np.mean(L, 0)

    print('## ① single heads (FPR95, mean over seeds)\n')
    print('| head | gamma ' + hdr); print(sep + ' --- |')
    S = mean(R['single'])
    for h in range(H):
        print(f'| {h} | {g[h]:.3f} | {fmt(S[h])} |')
    best_single = S.min(0)
    print(f'| best single per column | - | {fmt(best_single)} |')
    print(f'| best single on AVG (head {S[:, -1].argmin()}) | - | {fmt(S[S[:, -1].argmin()])} |')

    print('\n## ③ same H heads, different combination rules (FPR95)\n')
    print('| rule ' + hdr); print(sep)
    for k, lab in [('raw', 'raw residual sum (MHKSA-like, e1 only)'), ('zmean', 'z-score mean (ensemble)'), ('zmax', 'z-score max')]:
        print(f'| {lab} | {fmt(mean(R[k]))} |')

    print('\n## ② complementarity: best subsets (oracle on test OOD = upper bound)\n')
    P = {p: mean([d[p] for d in R['pairs']]) for p in R['pairs'][0]}
    top = sorted(P, key=lambda p: P[p][-1])[:5]
    print('| pair (z-mean) | gammas ' + hdr); print(sep + ' --- |')
    for p in top + [(0, H - 1)]:
        print(f'| {p} | {g[p[0]]:.2f}+{g[p[1]]:.2f} | {fmt(P[p])} |')
    better = sum(P[p][-1] < min(S[p[0], -1], S[p[1], -1]) for p in P)
    print(f'\npairs whose AVG beats BOTH of their own heads: {better}/{len(P)}')
    print('\ngreedy subset (seed-wise oracle, AVG FPR by subset size):  ' +
          '  '.join(f'k={k + 1}:{np.mean([r[k][1][-1] for r in R["greedy"]]):.2f}' for k in range(H)))
    print('heads picked first (seed 0): ' + ' -> '.join(str(h) for h in R['greedy'][0][-1][0]))

    print('\n## head score correlation (Spearman, mean over seeds): ID test / pooled OOD\n')
    Ci, Co = mean(R['corr_in']), mean(R['corr_out'])
    print('| head | ' + ' | '.join(str(h) for h in range(H)) + ' |'); print('| --- ' * (H + 1) + '|')
    for i in range(H):
        print(f'| {i} | ' + ' | '.join(f'{Ci[i, j]:.2f}/{Co[i, j]:.2f}' for j in range(H)) + ' |')

    print('\n## why max hurts: share of ID test samples with z > 2\n')
    one = mean([t[0] for t in R['idtail']]); mx = np.mean([t[1] for t in R['idtail']])
    print('per head: ' + '  '.join(f'{x * 100:.1f}%' for x in one) + f'   |   max over heads: {mx * 100:.1f}%')


def influence(cfg, name, H, f_in, f_out, oods, SEEDS):
    def fpr(s_in, s_out):
        return evaluate_all(s_in, s_out)[1]['FPR']


    RULES = ('raw', 'zmean')
    acc = {k: [] for k in ['single', 'share', 'loo', 'group', 'all', 'gammas']}
    for seed in SEEDS:
        m = load_pickle(ckpt_path(cfg, name, seed, H))['model']
        out_in = m.stage1.transform(f_in)
        out_o = {n: m.stage1.transform(x) for n, x in f_out.items()}
        mu, sd = m.ens.ref_mu[:, None], m.ens.ref_sd[:, None]
        # per-head terms of each rule (higher = more OOD); the score is -(sum or mean of terms)
        T = {'raw': (out_in['mhksa'][1] ** 2, {n: o['mhksa'][1] ** 2 for n, o in out_o.items()}),
             'zmean': ((out_in['corp'][1] - mu) / sd, {n: (o['corp'][1] - mu) / sd for n, o in out_o.items()})}

        def score(rule, idx):
            ti, to = T[rule]
            return fpr(-ti[idx].sum(0), {n: -t[idx].sum(0) for n, t in to.items()})

        allh = list(range(H))
        single = np.array([fpr(-out_in['corp'][1][h], {n: -o['corp'][1][h] for n, o in out_o.items()}) for h in allh])
        order = np.argsort(single)                      # best -> worst single head
        good, bad = sorted(order[:H // 2]), sorted(order[H // 2:])
        share, loo, group, full = {}, {}, {}, {}
        for r in RULES:
            ti, to = T[r]
            to_all = np.concatenate(list(to.values()), 1)
            w = ti.mean(1) if r == 'raw' else np.full(H, 1.0)              # weight on ID (z terms all centred at 0)
            gap = to_all.mean(1) - ti.mean(1)                                # OOD - ID: separation signal
            noise = ti.var(1)                                                # spread of the term on ID
            share[r] = np.stack([w / w.sum(), gap / gap.sum(), noise / noise.sum()])   # (3, H)
            full[r] = score(r, allh)
            loo[r] = np.array([score(r, [h for h in allh if h != k]) - full[r] for k in allh])
            group[r] = (score(r, good), score(r, bad))
        acc['single'].append(single); acc['share'].append(share); acc['loo'].append(loo)
        acc['group'].append(group); acc['all'].append(full); acc['gammas'].append(m.stage1.gammas)

    g = np.mean(acc['gammas'], 0)
    S = np.mean(acc['single'], 0)
    rank = np.argsort(np.argsort(S))
    print('## single heads (FPR95, AVG over OOD sets) and rank (0 = best)\n')
    print('| head | gamma | FPR95 | rank |\n| --- | --- | --- | --- |')
    for h in range(H):
        print(f'| {h} | {g[h]:.3f} | {S[h]:.2f} | {rank[h]} |')

    for r, lab in [('raw', 'raw residual sum (MHKSA stage 1)'), ('zmean', 'z-score mean (ensemble)')]:
        sh = np.mean([a[r] for a in acc['share']], 0)
        lo = np.mean([a[r] for a in acc['loo']], 0)
        print(f'\n## rule: {lab}   (all {H} heads: FPR95 {np.mean([a[r] for a in acc["all"]]):.2f})\n')
        print('### ① influence vs contribution, and ② leave-one-head-out\n')
        print('| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |')
        print('| --- | --- | --- | --- | --- | --- | --- |')
        for h in range(H):
            print(f'| {h} | {g[h]:.3f} | {S[h]:.2f} | {100 * sh[0, h]:.1f}% | {100 * sh[1, h]:.1f}% | {100 * sh[2, h]:.1f}% | {lo[h]:+.2f} |')
        gd = np.mean([a[r][0] for a in acc['group']]); bd = np.mean([a[r][1] for a in acc['group']])
        al = np.mean([a[r] for a in acc['all']])
        print(f'\n### ③ groups (halves by single-head FPR, per seed)\n')
        print(f'| all {H} heads | better half only | worse half only | damage from the worse half (all - better half) |')
        print('| --- | --- | --- | --- |')
        print(f'| {al:.2f} | {gd:.2f} | {bd:.2f} | {al - gd:+.2f} |')


class _Tee:
    """print to the terminal and to the Markdown file."""
    def __init__(self, f, s):
        self.f, self.s = f, s

    def write(self, m):
        self.f.write(m); self.s.write(m)

    def flush(self):
        self.f.flush(); self.s.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['heads', 'influence'])
    ap.add_argument('--mode', default=os.environ.get('MODE'), choices=['ce', 'supcon'])
    ap.add_argument('--cache', default=os.environ.get('C'), help='feature cache dir (data.cache_dir)')
    ap.add_argument('--gamma-base', default=os.environ.get('GB', ''), help='fixed base gamma run (kernel.gamma_base)')
    ap.add_argument('--set', nargs='*', default=os.environ.get('EXTRA', '').split(), help='extra config overrides')
    ap.add_argument('--seeds', nargs='*', type=int, default=[int(s) for s in os.environ.get('SEEDS', '0 1 2 3 4').split()])
    ap.add_argument('--out', default='results', help='Markdown goes to <out>/markdown/')
    a = ap.parse_args()
    if not a.mode or not a.cache:
        ap.error('--mode and --cache are required (or MODE / C in the environment)')
    sets = list(a.set) + [f'data.cache_dir={a.cache}'] + ([f'kernel.gamma_base={a.gamma_base}'] if a.gamma_base else [])
    cfg = load_config(f'configs/c10_r18_{a.mode}.yaml', sets)
    name = run_name(cfg)
    H = max(cfg.kernel.heads)
    f_in, f_out = load_eval(cfg)
    base = f'gb{a.gamma_base}' if a.gamma_base else 'median'
    path = os.path.join(a.out, 'markdown', f'{a.what}_{a.mode}_{base}_m{cfg.kernel.mult_min}-{cfg.kernel.mult_max}.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sys.stdout = _Tee(open(path, 'w'), sys.stdout)
    print(f'# {a.what}: {a.mode}   run={name}   H={H}   seeds={a.seeds}\n')
    {'heads': heads, 'influence': influence}[a.what](cfg, name, H, f_in, f_out, list(f_out), a.seeds)
    sys.stdout.f.close()
    sys.stdout = sys.stdout.s
    print(f'saved {path}')


if __name__ == '__main__':
    main()

# Diagnose why combining heads does not beat the best head (analysis only; reads checkpoints of the largest H).
#   ① head pool    : FPR95 of every single head
#   ② complementarity: can ANY subset of heads beat the best single head?  (oracle search on test OOD = upper bound)
#                     + how correlated the heads' scores are
#   ③ combination rule: what each rule (z-mean, z-max, raw-sum like MHKSA) gets with the SAME heads
# usage (from ~/mhksa_ood):  MODE=ce C=<cache> [GB=2] [SEEDS="0 1 2 3 4"] python scripts/diag_heads.py
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, for src/
import numpy as np
from src.config import load_config, run_name
from src.data import load_eval
from src.utils import ckpt_path, load_pickle
from src.metrics import evaluate_all

MODE, GB = os.environ['MODE'], os.environ.get('GB', '')
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0 1 2 3 4').split()]
sets = os.environ.get('EXTRA', '').split() + [f'data.cache_dir={os.environ["C"]}'] + ([f'kernel.gamma_base={GB}'] if GB else [])
cfg = load_config(f'configs/c10_r18_{MODE}.yaml', sets)
name = run_name(cfg)
H = max(cfg.kernel.heads)
f_in, f_out = load_eval(cfg)
oods = list(f_out)
print(f'# {MODE} {"gamma_base=" + GB if GB else "median gamma"}   run={name}   H={H}   seeds={SEEDS}\n')


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

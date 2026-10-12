# How much do the bad heads weigh in the combined score?  (analysis only; reads checkpoints of the largest H)
#   ① influence vs contribution per head: share of the score on ID (weight), of the OOD-ID gap (signal),
#      and of the ID score variance (noise)
#   ② leave-one-head-out: FPR95 change when one head is removed (negative = removing it helps = harmful head)
#   ③ bad-head group: all heads vs the better half only vs the worse half only (halves ranked by single-head FPR)
# Rules: "raw" = MHKSA stage-1 residuals squared and summed (MHKSA without stage 2, which would need refitting),
#        "zmean" = ensemble [3] z-score mean.
# usage (from ~/mhksa_ood):  MODE=ce C=<cache> [GB=2] [EXTRA="kernel.mult_max=10"] [SEEDS="0 1 2 3 4"] python scripts/diag_influence.py
import os, sys
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
print(f'# {MODE}   run={name}   H={H}   seeds={SEEDS}\n')


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

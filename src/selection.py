"""[2-variant] choose ONE head among H.

oracle : head with the lowest avg FPR on the test OOD sets  -> upper bound, not a fair number
loo    : leave-one-OOD-out; for each OOD set, the head is chosen on the other OOD sets
"""
import numpy as np


def best_head_oracle(head_results):
    """head_results: list over heads of (per_ood dict, avg dict). returns (idx, per, avg)"""
    b = int(np.argmin([avg['FPR'] for _, avg in head_results]))
    return b, head_results[b][0], head_results[b][1]


def best_head_loo(head_results, ood_names):
    fpr = np.array([[per[n]['FPR'] for n in ood_names] for per, _ in head_results])   # (H, n_ood)
    chosen, per = [], {}
    for j, n in enumerate(ood_names):
        others = [k for k in range(len(ood_names)) if k != j]
        b = int(np.argmin(fpr[:, others].mean(1))) if others else int(np.argmin(fpr[:, j]))
        chosen.append(b)
        per[n] = head_results[b][0][n]
    avg = {k: float(np.mean([v[k] for v in per.values()])) for k in ['FPR', 'AUROC']}
    return chosen, per, avg

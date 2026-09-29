"""FPR@TPR95 / AUROC.  Uses ood-kernel-pca's metrics.py if it is importable (identical numbers to the paper)."""
import numpy as np

try:
    import metrics as _repo_metrics          # ood-kernel-pca/metrics.py (put repo root on PYTHONPATH)
    _cal = _repo_metrics.cal_metric
except Exception:
    _cal = None


def _fallback(known, novel):
    known, novel = np.sort(known), np.sort(novel)
    thr = known[round(0.05 * len(known))]
    fpr = np.sum(novel > thr) / float(len(novel))
    lo = np.searchsorted(novel, known, side='left')
    hi = np.searchsorted(novel, known, side='right')
    auroc = (lo + 0.5 * (hi - lo)).sum() / (len(known) * len(novel))
    return {'FPR': fpr, 'AUROC': auroc}


def ood_metrics(score_in, score_out):
    """score: higher = ID. returns {'FPR','AUROC'} in %."""
    r = _cal(score_in.copy(), score_out.copy()) if _cal else _fallback(score_in, score_out)
    return {'FPR': 100 * float(r['FPR']), 'AUROC': 100 * float(r['AUROC'])}


def evaluate_all(score_in, scores_out):
    """scores_out: {ood_name: score}. returns ({ood_name: metrics}, avg metrics)"""
    per = {n: ood_metrics(score_in, s) for n, s in scores_out.items()}
    avg = {k: float(np.mean([v[k] for v in per.values()])) for k in ['FPR', 'AUROC']}
    return per, avg

"""Head-combination variants, scored on existing checkpoints (no retraining).   run.methods: combine

Why: in MHKSA the stage-1 term e1^2 = sum_h e_h^2 is dominated by the large-gamma heads (their
residuals are larger: ~3 % of e1^2 for the 0.3x head vs ~19 % for the 3x head), and the ensemble [3]
gives every head the same weight even when it is bad.  Variants (higher = more ID):

  Comb_MHKSA_norm         every stage-1 residual e_h and the stage-2 residual e2 divided by its ID RMS,
                          then sqrt(sum_h (e_h/s_h)^2 + (e2/s2)^2): all H+1 terms weigh the same on ID data
  Comb_MHKSA_norm_e1only  the same, stage-1 terms only
  Comb_Ens_max            ensemble [3] with max instead of mean over heads: max_h (e_h - mu_h) / sd_h
                          (same per-head z-score as CoRP_Ens_zscore, so the two differ only in sum vs max)

The MHKSA scales s_h, s2 come from a fixed ID-train subset (model.combine_ref samples, same subset for
every H of a seed).  Comb_Ens_max reuses the ensemble's stored reference statistics (all ID train).
"""
import numpy as np


def rms(e, axis=-1):
    return np.sqrt(np.mean(np.square(e, dtype=np.float64), axis=axis)) + 1e-12


class Combiner:
    def __init__(self, model, x_ref, ref_out=None):
        self.model = model
        out = ref_out or model.stage1.transform(x_ref)
        coords, E1 = out['mhksa']
        self.s1 = rms(E1, axis=1)                                         # (H,)
        self.s2 = rms(model.mhksa.pca2.transform(np.concatenate(coords, 1))[1])

    def scores(self, x, out=None):
        m = self.model
        out = out or m.stage1.transform(x)
        coords, E1 = out['mhksa']
        e2 = m.mhksa.pca2.transform(np.concatenate(coords, 1))[1]
        t1 = (np.square(E1 / self.s1[:, None])).sum(0)
        t2 = np.square(e2 / self.s2)
        _, E = out['corp']
        z = (E - m.ens.ref_mu[:, None]) / m.ens.ref_sd[:, None]
        return {'Comb_MHKSA_norm': -np.sqrt(t1 + t2),
                'Comb_MHKSA_norm_e1only': -np.sqrt(t1),
                'Comb_Ens_max': -z.max(0)}

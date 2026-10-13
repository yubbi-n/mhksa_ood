"""Head-combination variants, scored on existing checkpoints (no retraining).
run.methods: combine (Combiner), mhksa_corp (MHKSACorpStage1, at the end of this file)

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


class MHKSACorpStage1:
    """MHKSA with stage 1 = the CoRP truncation of every head (exp_var_ratio, 0.8 CE / 0.7 SupCon) instead of
    mhksa_evr1, so its per-head residuals are exactly those of the single heads / ensemble.   run.methods: mhksa_corp

    Stage 2 is refitted here on the concatenated stage-1 coords of the ID training set (no retraining of the heads),
    for each ratio in `ratios`.  ratio 1.0 keeps everything in stage 2 (e2 = 0): the score is then the plain
    residual sum sqrt(sum_h e_h^2).   Score names: MHKSA_s1corp_s2<ratio>.
    """

    def __init__(self, model, x_train, ratios, train_out=None):
        from .model import PCA
        self.model = model
        coords, _ = (train_out or model.stage1.transform(x_train))['corp']
        Z = np.concatenate(coords, 1)
        self.ratios = [float(r) for r in ratios]
        fit = [r for r in self.ratios if r < 1.0]
        self.pca2 = {}
        if fit:
            base = PCA(fit[0]).fit(Z)
            self.pca2 = {r: base.with_ratio(r) for r in fit}
            base.drop_spectrum()
        self.q2 = {r: p.q for r, p in self.pca2.items()}

    def scores(self, x, out=None):
        coords, E = (out or self.model.stage1.transform(x))['corp']
        e1sq = (np.square(E, dtype=np.float64)).sum(0)
        Z = np.concatenate(coords, 1) if self.pca2 else None
        s = {}
        for r in self.ratios:
            e2sq = np.square(self.pca2[r].transform(Z)[1], dtype=np.float64) if r < 1.0 else 0.0
            s[f'MHKSA_s1corp_s2{r:g}'] = -np.sqrt(e1sq + e2sq)
        return s

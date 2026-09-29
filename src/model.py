"""Models.  All scores follow the convention: higher = more ID.

  PCA            linear PCA with q chosen by explained-variance ratio (same rule as CoRP code)
  KernelHead     RFF_gamma -> centre -> PCA            (one head = CoRP with that gamma)
                 one eigen-decomposition, two truncations: exp_var_ratio (CoRP) and mhksa_evr1 (MHKSA stage 1)
  MultiHead      H KernelHeads (stage 1, shared by every method below)
  MHKSA          [2]  concat stage-1 coords -> PCA (stage 2); error = sqrt(e1^2 + e2^2)
  CoRPEnsemble   [3]  per-head error -> percentile/z-score vs ID reference -> weighted sum
  (best single head [2-variant] is a selection over MultiHead's per-head scores -> see selection.py)
"""
import numpy as np

from .kernels import RFF


class PCA:
    def __init__(self, exp_var_ratio):
        self.ratio = exp_var_ratio

    def fit(self, x):
        self.mu = x.mean(0)
        xc = (x - self.mu).astype(np.float64)
        s, U = np.linalg.eigh(xc.T @ xc)
        self._s, self._U = s[::-1], U[:, ::-1]
        return self._truncate()

    def _truncate(self):
        cum = np.cumsum(self._s) / self._s.sum()
        q = int(np.searchsorted(cum, self.ratio) + 1)      # smallest q with cum >= ratio
        self.q = min(max(q, 1), len(self._s) - 1)
        self.U = np.ascontiguousarray(self._U[:, :self.q]).astype(np.float32)
        return self

    def with_ratio(self, ratio):
        """same fitted eigen-decomposition, different explained-variance ratio."""
        p = PCA(ratio)
        p.mu, p._s, p._U = self.mu, self._s, self._U
        return p._truncate()

    def drop_spectrum(self):
        del self._s, self._U                                # keep checkpoints small
        return self

    def transform(self, x):
        """returns (coords in principal subspace, reconstruction-error norm)"""
        xc = x - self.mu
        c = xc @ self.U
        return c, np.linalg.norm(xc - c @ self.U.T, axis=1)


class KernelHead:
    def __init__(self, gamma, M, exp_var_ratio, mhksa_evr1, dim, rng):
        self.gamma = float(gamma)
        self.rff = RFF(gamma, M, dim, rng)
        self.pca = PCA(exp_var_ratio)          # CoRP / single head / ensemble
        self.mhksa_evr1 = mhksa_evr1

    def fit(self, x):
        self.pca.fit(self.rff(x))
        self.pca_mh = self.pca.with_ratio(self.mhksa_evr1).drop_spectrum()   # MHKSA stage 1
        self.pca.drop_spectrum()
        return self

    def transform(self, x):
        """returns {'corp': (coords, err), 'mhksa': (coords, err)} ; RFF computed once."""
        f = self.rff(x)
        return {'corp': self.pca.transform(f), 'mhksa': self.pca_mh.transform(f)}


class MultiHead:
    """stage 1: independent RFF+PCA per head."""

    def __init__(self, gammas, M, exp_var_ratio, mhksa_evr1, dim, rng):
        self.gammas = np.asarray(gammas, dtype=float)
        self.heads = [KernelHead(g, M, exp_var_ratio, mhksa_evr1, dim, rng) for g in self.gammas]

    def fit(self, x):
        for h in self.heads:
            h.fit(x)
        return self

    def transform(self, x):
        """returns {'corp' | 'mhksa': (list of coords per head, errors (H, n))}"""
        out = [h.transform(x) for h in self.heads]
        return {k: ([o[k][0] for o in out], np.stack([o[k][1] for o in out])) for k in ('corp', 'mhksa')}

    @property
    def qs(self):
        return [h.pca.q for h in self.heads]

    @property
    def qs_mh(self):
        return [h.pca_mh.q for h in self.heads]


class MHKSA:
    """[2] multi-head PCA -> concat -> PCA.
    e1 (stage-1 residual) and e2 (stage-2 residual) are orthogonal, so sqrt(e1^2 + e2^2)
    is the reconstruction error of the whole two-stage projection in the concatenated RFF space."""

    def __init__(self, stage1, mhksa_evr2):
        self.stage1 = stage1
        self.pca2 = PCA(mhksa_evr2)

    def fit(self, x, stage1_out=None):
        coords, _ = (stage1_out or self.stage1.transform(x))['mhksa']
        self.pca2.fit(np.concatenate(coords, 1)).drop_spectrum()
        self.concat_dim = sum(c.shape[1] for c in coords)
        return self

    def scores(self, x, stage1_out=None):
        coords, E = (stage1_out or self.stage1.transform(x))['mhksa']
        e1 = np.sqrt((E ** 2).sum(0))
        e2 = self.pca2.transform(np.concatenate(coords, 1))[1]
        return {'MHKSA': -np.sqrt(e1 ** 2 + e2 ** 2), 'MHKSA_e1only': -e1, 'MHKSA_e2only': -e2}


class CoRPEnsemble:
    """[3] per-head errors normalised w.r.t. ID reference errors, then weighted sum."""

    def __init__(self, stage1, norm='percentile', weights='equal'):
        self.stage1, self.norm, self.weights = stage1, norm, weights

    def fit(self, x_ref, stage1_out=None):
        _, E = (stage1_out or self.stage1.transform(x_ref))['corp']
        self.ref = np.sort(E, axis=1)                 # (H, n_ref)
        self.ref_mu, self.ref_sd = E.mean(1), E.std(1) + 1e-10
        H = len(self.stage1.heads)
        self.w = np.ones(H) / H                        # TODO: other weighting schemes
        return self

    def _normalize(self, E):
        if self.norm == 'percentile':
            return np.stack([np.searchsorted(r, e, side='right') / len(r) for r, e in zip(self.ref, E)])
        return (E - self.ref_mu[:, None]) / self.ref_sd[:, None]

    def scores(self, x, stage1_out=None):
        _, E = (stage1_out or self.stage1.transform(x))['corp']
        return {f'CoRP_Ens_{self.norm}': -(self.w[:, None] * self._normalize(E)).sum(0)}


class MultiKernelOOD:
    """Bundles stage 1 + all methods so stage 1 is computed once per input."""

    def __init__(self, gammas, cfg, dim, rng):
        m = cfg.model
        self.stage1 = MultiHead(gammas, cfg.kernel.M, m.exp_var_ratio, m.mhksa_evr1, dim, rng)
        self.mhksa = MHKSA(self.stage1, m.mhksa_evr2)
        self.ens = CoRPEnsemble(self.stage1, m.ens_norm, m.ens_weights)

    def fit(self, x_train, x_calib=None):
        self.stage1.fit(x_train)
        out = self.stage1.transform(x_train)
        self.mhksa.fit(x_train, out)
        self.ens.fit(x_calib, None) if x_calib is not None else self.ens.fit(x_train, out)
        return self

    def scores(self, x):
        """dict: method name -> score (n,) ; plus 'head{h}' for each single head."""
        out = self.stage1.transform(x)
        s = {f'head{h}': -e for h, e in enumerate(out['corp'][1])}
        s.update(self.mhksa.scores(x, out))
        s.update(self.ens.scores(x, out))
        return s

    def info(self):
        return {'gammas': self.stage1.gammas.tolist(), 'q1': self.stage1.qs, 'q1_mh': self.stage1.qs_mh,
                'concat_dim': self.mhksa.concat_dim, 'q2': self.mhksa.pca2.q}

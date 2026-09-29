"""Gaussian-kernel RFF and gamma schedules.  k(x, y) = exp(-gamma ||x - y||^2),  w ~ N(0, 2 gamma I)."""
import numpy as np


class RFF:
    def __init__(self, gamma, M, dim, rng):
        self.gamma, self.M = float(gamma), int(M)
        self.w = (np.sqrt(2 * gamma) * rng.normal(size=(M, dim))).astype(np.float32)
        self.u = (2 * np.pi * rng.rand(M)).astype(np.float32)

    def __call__(self, x):
        return (np.sqrt(2.0 / self.M) * np.cos(x @ self.w.T + self.u[None, :])).astype(np.float32)


def median_heuristic(x, n_sub=5000, mode='half', rng=None):
    """gamma from the median pairwise distance of (a subsample of) x.
    half : sigma = median dist, k = exp(-d^2 / 2 sigma^2) -> gamma = 1 / (2 med^2)
    plain: gamma = 1 / med^2
    """
    rng = rng or np.random
    idx = rng.choice(len(x), size=min(n_sub, len(x)), replace=False)
    z = x[idx].astype(np.float64)
    sq = (z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    med = float(np.sqrt(np.median(d2[np.triu_indices(len(z), 1)])))
    gamma = 1.0 / (2 * med ** 2) if mode == 'half' else 1.0 / med ** 2
    return gamma, med


def gamma_schedule(H, kcfg, x_train=None, rng=None):
    """H = 1 -> median heuristic ; H >= 2 -> H values spread over [gamma_min, gamma_max]."""
    if H == 1:
        g, _ = median_heuristic(x_train, kcfg.median_subsample, kcfg.median_mode, rng)
        return np.array([g])
    if kcfg.gamma_spacing == 'log':
        return np.geomspace(kcfg.gamma_min, kcfg.gamma_max, H)
    return np.linspace(kcfg.gamma_min, kcfg.gamma_max, H)

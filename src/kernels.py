"""Gaussian-kernel RFF and gamma schedules.  k(x, y) = exp(-gamma ||x - y||^2),  w ~ N(0, 2 gamma I)."""
import numpy as np


class RFF:
    def __init__(self, gamma, M, dim, rng):
        self.gamma, self.M, self.dim = float(gamma), int(M), int(dim)
        self._rng_state = rng.get_state()                   # w, u are redrawn from it when unpickled
        self._draw(rng)

    def _draw(self, rng):
        self.w = (np.sqrt(2 * self.gamma) * rng.normal(size=(self.M, self.dim))).astype(np.float32)
        self.u = (2 * np.pi * rng.rand(self.M)).astype(np.float32)

    def __getstate__(self):                                 # keep checkpoints small: w is M x dim
        return {k: v for k, v in self.__dict__.items() if k not in ('w', 'u')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'w' not in state:                                # (old checkpoints store w, u)
            rng = np.random.RandomState()
            rng.set_state(self._rng_state)
            self._draw(rng)

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


def gamma_schedule(H, kcfg, g_med):
    """gamma_h = g_med * m_h.  H = 1 -> m = 1 (median heuristic) ; H >= 2 -> m_h over [mult_min, mult_max]."""
    if H == 1:
        return np.array([g_med])
    space = np.geomspace if kcfg.gamma_spacing == 'log' else np.linspace
    return g_med * space(kcfg.mult_min, kcfg.mult_max, H)

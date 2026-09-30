"""Fit multi-kernel models on ID train features and save checkpoints.

    python train.py --config configs/c10_r18_ce.yaml
    python train.py --config configs/c10_r18_ce.yaml --set kernel.heads=[1,5,10] run.seeds=[0]
"""
import sys
import time
import argparse

import numpy as np

from src.config import load_config, run_name
from src.data import load_id_train, split_calib
from src.kernels import gamma_schedule, median_heuristic
from src.model import MultiKernelOOD
from src.utils import set_seed, Tee, ckpt_path, save_pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[], help='overrides, e.g. kernel.M=4096')
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    name = run_name(cfg)
    sys.stdout = Tee(f'{cfg.run.logs_dir}/{name}/train.log')
    print(f'\n==== train: {name} ====')

    x_all = load_id_train(cfg)
    print(f'ID train {x_all.shape}')
    t0 = time.time()
    for seed in cfg.run.seeds:
        rng = set_seed(seed)
        x_fit, x_cal = split_calib(x_all, cfg.data.calib_holdout, rng)
        # one median-heuristic gamma per seed, shared by every H (the H = 1 baseline is the m = 1 head)
        g_med, med = median_heuristic(x_fit, cfg.kernel.median_subsample, cfg.kernel.median_mode,
                                      np.random.RandomState([seed, 0]))
        print(f'[seed {seed}] median dist {med:.4f} -> gamma_med {g_med:.4f}')
        for H in cfg.kernel.heads:
            # own RNG per (seed, H): results for a given H do not depend on which other H's are run
            rng_h = np.random.RandomState([seed, H])
            gammas = gamma_schedule(H, cfg.kernel, g_med)
            model = MultiKernelOOD(gammas, cfg, x_fit.shape[1], rng_h).fit(x_fit, x_cal)
            info = model.info()
            print(f"[seed {seed}] H={H:2d} gammas={[round(g, 4) for g in info['gammas']]} "
                  f"q1={info['q1']} | MHKSA q1={info['q1_mh']} concat={info['concat_dim']} q2={info['q2']}  ({time.time() - t0:.0f}s)")
            save_pickle({'model': model, 'cfg': dict(cfg), 'seed': seed, 'H': H},
                        ckpt_path(cfg, name, seed, H))
    print(f'checkpoints -> {cfg.run.ckpt_dir}/{name}/')


if __name__ == '__main__':
    main()

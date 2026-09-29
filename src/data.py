"""Cached penultimate features (from ood-kernel-pca/feat_extract.py) + cosine normalisation."""
import os
import numpy as np


def cosine_normalize(x):
    # l2 normalisation == feature map of the cosine kernel (CoP / CoRP first step)
    return np.ascontiguousarray(x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)).astype(np.float32)


def _load(path):
    return np.load(path, allow_pickle=True).T.astype(np.float32)   # cache is stored as (d, n)


def feature_dir(cfg):
    d = cfg.data
    return os.path.join(d.cache_dir, d.in_data, d.arch, d.train_mode)


def load_id_train(cfg):
    return cosine_normalize(_load(os.path.join(feature_dir(cfg), 'train_in.npy')))


def load_eval(cfg):
    """returns (id_test, {ood_name: feats})"""
    root = feature_dir(cfg)
    f_in = cosine_normalize(_load(os.path.join(root, 'val_in.npy')))
    f_out = {n: cosine_normalize(_load(os.path.join(root, f'{n}_out.npy'))) for n in cfg.data.out_datasets}
    return f_in, f_out


def split_calib(x, frac, rng):
    """split ID train into (fit, calib). frac=0 -> calib is None."""
    n_cal = int(round(frac * len(x)))
    if n_cal == 0:
        return x, None
    perm = rng.permutation(len(x))
    return x[perm[n_cal:]], x[perm[:n_cal]]

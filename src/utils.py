import os
import sys
import csv
import pickle
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.RandomState(seed)


class Tee:
    """print to stdout and a log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f, self.s = open(path, 'a'), sys.stdout

    def write(self, m):
        self.s.write(m); self.f.write(m)

    def flush(self):
        self.s.flush(); self.f.flush()


def ckpt_path(cfg, name, seed, H):
    return os.path.join(cfg.run.ckpt_dir, name, f'seed{seed}_H{H}.pkl')


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


CSV_HEADER = ['seed', 'H', 'method', 'gammas', 'extra', 'dataset', 'FPR95', 'AUROC']


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

"""YAML config with `_base_` inheritance and command-line overrides.

    cfg = load_config('configs/c10_r18_ce.yaml', ['kernel.heads=[1,2]', 'run.seeds=[0]'])
    cfg.kernel.M  -> 2048
"""
import os
import copy
import yaml


class Config(dict):
    """dict with attribute access (cfg.model.exp_var_ratio)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def __setattr__(self, k, v):
        self[k] = v


def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('_base_', None)
    if base:
        cfg = _merge(_load(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(path, overrides=()):
    cfg = _load(path)
    for ov in overrides or []:           # "a.b.c=value" (value parsed as YAML)
        key, val = ov.split('=', 1)
        d = cfg
        *parents, last = key.split('.')
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    if cfg['model'].get('mhksa_evr2') is None:
        cfg['model']['mhksa_evr2'] = cfg['model']['mhksa_evr1']
    return _to_cfg(cfg)


def _to_cfg(d):
    return Config({k: _to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def run_name(cfg):
    k, m = cfg.kernel, cfg.model
    return (f"{cfg.data.in_data}-{cfg.data.arch}-{cfg.data.train_mode}"
            f"-evr{m.exp_var_ratio}-mh{m.mhksa_evr1}_{m.mhksa_evr2}-M{k.M}"
            f"-m{k.mult_min}_{k.mult_max}{k.gamma_spacing}"
            + (f"-gb{k.gamma_base}" if k.get('gamma_base') is not None else '') + f"{cfg.run.tag}")

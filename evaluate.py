"""Load checkpoints, score ID test / OOD sets, report every method.

    python evaluate.py --config configs/c10_r18_ce.yaml      (same config/--set as train.py)

methods reported per (seed, H)  (select with run.methods, e.g. --set run.methods=[mhksa]):
    head{h}              each single head (= CoRP with that gamma)
    MHKSA / _e1only / _e2only                                    [2]
    best_head_oracle / best_head_loo                             [2-variant]
    CoRP_Ens_percentile / CoRP_Ens_zscore                        [3]
"""
import sys
import argparse
import numpy as np

from src.config import load_config, run_name
from src.data import load_eval
from src.metrics import evaluate_all, METRICS_SOURCE
from src.selection import best_head_oracle, best_head_loo
from src.utils import Tee, ckpt_path, load_pickle, write_csv

ALL_METHODS = ['mhksa', 'best_head', 'ensemble']


def fmt_g(gs):
    return ' '.join(f'{g:.4f}' for g in gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    name = run_name(cfg)
    methods = list(cfg.run.get('methods', ALL_METHODS))
    assert set(methods) <= set(ALL_METHODS), f'run.methods must be a subset of {ALL_METHODS}'
    suffix = '' if set(methods) == set(ALL_METHODS) else '_' + '_'.join(methods)   # keep full-run logs intact
    sys.stdout = Tee(f'{cfg.run.logs_dir}/{name}/eval{suffix}.log')
    print(f'\n==== eval: {name} ====  (FPR95 / AUROC, %)  metrics: {METRICS_SOURCE}  methods: {methods}')

    f_in, f_out = load_eval(cfg)
    oods = list(f_out)
    rows = []

    def add(seed, H, method, gammas, per, avg, extra=''):
        for n in oods:
            rows.append([seed, H, method, fmt_g(gammas), extra, n, per[n]['FPR'], per[n]['AUROC']])
        rows.append([seed, H, method, fmt_g(gammas), extra, 'AVG', avg['FPR'], avg['AUROC']])
        cells = ' '.join(f"{n[:6]:>6s} {per[n]['FPR']:5.2f}/{per[n]['AUROC']:5.2f}" for n in oods)
        print(f'  {method:22s} | {cells} | AVG {avg["FPR"]:5.2f}/{avg["AUROC"]:5.2f}')

    for seed in cfg.run.seeds:
        for H in cfg.kernel.heads:
            model = load_pickle(ckpt_path(cfg, name, seed, H))['model']
            info = model.info()
            gammas = info['gammas']
            print(f"\n[seed {seed}] H={H} gammas={[round(g, 4) for g in gammas]} q1={info['q1']} | MHKSA q1={info['q1_mh']} q2={info['q2']}")

            s_in = model.scores(f_in)
            s_out = {n: model.scores(x) for n, x in f_out.items()}

            head_results = []
            if 'best_head' in methods or H == 1:            # H = 1 single head = median-heuristic baseline
                for h in range(H):
                    per, avg = evaluate_all(s_in[f'head{h}'], {n: s_out[n][f'head{h}'] for n in oods})
                    head_results.append((per, avg))
                    add(seed, H, 'single_head', [gammas[h]], per, avg, extra=f'head{h}_q{info["q1"][h]}')

            for mth in [k for k in s_in if not k.startswith('head')]:
                if (mth.startswith('MHKSA') and 'mhksa' in methods) or (mth.startswith('CoRP_Ens') and 'ensemble' in methods):
                    per, avg = evaluate_all(s_in[mth], {n: s_out[n][mth] for n in oods})
                    add(seed, H, mth, gammas, per, avg, extra=f"q2={info['q2']}" if mth.startswith('MHKSA') else '')

            if 'best_head' in methods:
                b, per, avg = best_head_oracle(head_results)
                add(seed, H, 'best_head_oracle', [gammas[b]], per, avg, extra=f'head{b}')
                chosen, per, avg = best_head_loo(head_results, oods)
                add(seed, H, 'best_head_loo', [gammas[c] for c in chosen], per, avg,
                    extra='heads=' + '/'.join(map(str, chosen)))

    csv_path = f'{cfg.run.logs_dir}/{name}/results{suffix}.csv'
    write_csv(rows, csv_path)

    # ---- summary: AVG over OOD sets, mean ± std over seeds
    cols = (['MHKSA', 'MHKSA_e1only', 'MHKSA_e2only'] if 'mhksa' in methods else []) \
        + (['best_head_oracle', 'best_head_loo'] if 'best_head' in methods else []) \
        + (['CoRP_Ens_percentile', 'CoRP_Ens_zscore'] if 'ensemble' in methods else [])
    print('\n==== summary: AVG over OOD sets, mean±std over seeds (FPR95 / AUROC) ====')
    print(f"{'H':>3s} " + ' '.join(f'{m[:20]:>22s}' for m in cols))
    for H in cfg.kernel.heads:
        cells = []
        for m in cols:
            v = np.array([[r[6], r[7]] for r in rows if r[1] == H and r[2] == m and r[5] == 'AVG'])
            cells.append(f'{v[:, 0].mean():5.2f}±{v[:, 0].std():4.2f}/{v[:, 1].mean():5.2f}' if len(v) else '-')
        print(f'{H:>3d} ' + ' '.join(f'{c:>22s}' for c in cells))
    base = np.array([r[6] for r in rows if r[1] == 1 and r[2] == 'single_head' and r[5] == 'AVG'])
    if len(base):
        how = (f'gamma_base={cfg.kernel.gamma_base}' if cfg.kernel.get('gamma_base') is not None
               else 'median heuristic')
        print(f'baseline (H=1, {how}, evr {cfg.model.exp_var_ratio}): '
              f'FPR95 {base.mean():.2f}±{base.std():.2f}  <- lower FPR than this = better than the baseline')
    print(f'\nCSV -> {csv_path}')


if __name__ == '__main__':
    main()

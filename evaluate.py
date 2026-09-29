"""Load checkpoints, score ID test / OOD sets, report every method.

    python evaluate.py --config configs/c10_r18_ce.yaml      (same config/--set as train.py)

methods reported per (seed, H):
    head{h}              each single head (= CoRP with that gamma)
    MHKSA / _e1only / _e2only                                    [2]
    best_head_oracle / best_head_loo                             [2-variant]
    CoRP_Ens_<norm>                                              [3]
"""
import sys
import argparse
import numpy as np

from src.config import load_config, run_name
from src.data import load_eval
from src.metrics import evaluate_all, METRICS_SOURCE
from src.selection import best_head_oracle, best_head_loo
from src.utils import Tee, ckpt_path, load_pickle, write_csv

SUMMARY_METHODS = ['MHKSA', 'MHKSA_e1only', 'MHKSA_e2only', 'best_head_oracle', 'best_head_loo']


def fmt_g(gs):
    return ' '.join(f'{g:.4f}' for g in gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    name = run_name(cfg)
    sys.stdout = Tee(f'{cfg.run.logs_dir}/{name}/eval.log')
    print(f'\n==== eval: {name} ====  (FPR95 / AUROC, %)  metrics: {METRICS_SOURCE}')

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
            print(f"\n[seed {seed}] H={H} gammas={[round(g, 4) for g in gammas]} q1={info['q1']} q2={info['q2']}")

            s_in = model.scores(f_in)
            s_out = {n: model.scores(x) for n, x in f_out.items()}

            head_results = []
            for h in range(H):
                per, avg = evaluate_all(s_in[f'head{h}'], {n: s_out[n][f'head{h}'] for n in oods})
                head_results.append((per, avg))
                add(seed, H, 'single_head', [gammas[h]], per, avg, extra=f'head{h}_q{info["q1"][h]}')

            for mth in [k for k in s_in if not k.startswith('head')]:
                per, avg = evaluate_all(s_in[mth], {n: s_out[n][mth] for n in oods})
                add(seed, H, mth, gammas, per, avg, extra=f"q2={info['q2']}" if mth.startswith('MHKSA') else '')

            b, per, avg = best_head_oracle(head_results)
            add(seed, H, 'best_head_oracle', [gammas[b]], per, avg, extra=f'head{b}')
            chosen, per, avg = best_head_loo(head_results, oods)
            add(seed, H, 'best_head_loo', [gammas[c] for c in chosen], per, avg,
                extra='heads=' + '/'.join(map(str, chosen)))

    csv_path = f'{cfg.run.logs_dir}/{name}/results.csv'
    write_csv(rows, csv_path)

    # ---- summary: AVG over OOD sets, mean ± std over seeds
    methods = SUMMARY_METHODS + [f'CoRP_Ens_{cfg.model.ens_norm}']
    print('\n==== summary: AVG over OOD sets, mean±std over seeds (FPR95 / AUROC) ====')
    print(f"{'H':>3s} " + ' '.join(f'{m[:20]:>22s}' for m in methods))
    for H in cfg.kernel.heads:
        cells = []
        for m in methods:
            v = np.array([[r[6], r[7]] for r in rows if r[1] == H and r[2] == m and r[5] == 'AVG'])
            cells.append(f'{v[:, 0].mean():5.2f}±{v[:, 0].std():4.2f}/{v[:, 1].mean():5.2f}' if len(v) else '-')
        print(f'{H:>3d} ' + ' '.join(f'{c:>22s}' for c in cells))
    print(f'\nCSV -> {csv_path}')


if __name__ == '__main__':
    main()

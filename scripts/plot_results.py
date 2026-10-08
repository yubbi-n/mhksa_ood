"""Plot MHKSA [2] / best-head [2-variant] / ensemble [3] results from evaluate.py CSVs.

    python scripts/plot_results.py                       # auto-find logs/*-ce-* and logs/*-supcon-*
    python scripts/plot_results.py --h 3 --out figures   # per-OOD bars at H = 3
    python scripts/plot_results.py --ce logs/<run> --supcon logs/<run>

Reads  every <run>/results*.csv written by evaluate.py (results.csv, results_mhksa.csv, results_best_head.csv, ...).
Writes <out>/fig1_vs_heads.png      FPR95 / AUROC vs number of heads (MHKSA, best head LOO, ensemble, baseline)
       <out>/fig2_mhksa_parts.png   MHKSA vs stage-1-only / stage-2-only error
       <out>/fig3_per_ood_H{h}.png  per-OOD FPR95 at one H (baseline vs MHKSA vs best head LOO vs ensemble)
       <out>/summary.md             every plotted number as a table (mean ± std over seeds)
"""
import os
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reference palette (light surface); slots validated as sets {1,2,3} and {1,4,5}
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BASE = '#8a8984'                                    # baseline reference: neutral, not a series hue
STYLE = {   # method -> (label, colour, marker, linestyle)
    'MHKSA':            ('MHKSA [2]',                  '#2a78d6', 'o', '-'),
    'best_head_loo':    ('Best head, LOO [2-var]',     '#eb6834', 's', '-'),
    'best_head_oracle': ('Best head, oracle [2-var]',  '#1baf7a', '^', '--'),
    'CoRP_Ens_percentile': ('Ensemble, percentile [3]', '#1baf7a', '^', '-'),
    'CoRP_Ens_zscore':  ('Ensemble, z-score [3]',      '#1baf7a', 'v', '--'),
    'MHKSA_e1only':     ('Stage-1 error only',         '#eda100', 'D', '-'),
    'MHKSA_e2only':     ('Stage-2 error only',         '#e87ba4', 'v', '-'),
}
OOD_ORDER = ['SVHN', 'LSUN', 'iSUN', 'Texture', 'places365']
MODES = [('ce', 'CE'), ('supcon', 'SupCon')]
BASE_GAMMA = {}       # mode title -> 'median γ' or 'γ=<value>' (from the run folder name, '-gb<value>')


def base_label(title):
    return BASE_GAMMA.get(title, 'median γ')

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
    'legend.frameon': False,
})


# ---------------------------------------------------------------- data
def find_run(logs, mode):
    runs = sorted(glob.glob(os.path.join(logs, f'*-{mode}-*')), key=os.path.getmtime)
    if len(runs) > 1:
        print(f'[{mode}] {len(runs)} runs found, using the newest: {runs[-1]}  (pick one with --{mode})')
    return runs[-1] if runs else None


def load(run):
    """{(method, H, dataset): [(FPR, AUROC) per seed]} from both CSVs of a run."""
    d = defaultdict(list)
    seen = set()
    for path in sorted(glob.glob(os.path.join(run, 'results*.csv'))):
        for r in csv.DictReader(open(path)):
            key = (r['method'], int(r['H']), r['dataset'])
            if (key, r['seed']) in seen:            # single_head H=1 appears in both files
                continue
            seen.add((key, r['seed']))
            d[key].append((float(r['FPR95']), float(r['AUROC'])))
    return d


def stat(d, method, H, dataset='AVG', col=0):
    v = np.array([x[col] for x in d.get((method, H, dataset), [])])
    return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)


def heads_of(d):
    return sorted({k[1] for k in d})


# ---------------------------------------------------------------- plotting helpers
def series(ax, d, method, Hs, col):
    m = np.array([stat(d, method, H, col=col)[:2] for H in Hs])
    if np.isnan(m[:, 0]).all():
        return None
    lab, c, mk, ls = STYLE[method]
    ax.errorbar(Hs, m[:, 0], yerr=m[:, 1], color=c, marker=mk, ls=ls, lw=2, ms=6, capsize=2.5,
                elinewidth=1, mec=SURFACE, mew=1.2, label=lab, zorder=3)
    return m


def baseline(ax, d, col):
    mu, sd, n = stat(d, 'single_head', 1, col=col)
    if n == 0:
        return
    ax.axhspan(mu - sd, mu + sd, color=BASE, alpha=0.08, lw=0, zorder=1)
    ax.axhline(mu, color=BASE, ls=(0, (4, 3)), lw=1.5, zorder=2, label='Baseline (H=1 head) ± std')


def base_txt(d, col=0):
    mu, sd, n = stat(d, 'single_head', 1, col=col)
    return f'  (baseline {mu:.2f} ± {sd:.2f})' if n else ''


def label_ends(ax, Hs, ends):
    """direct labels at the right end of each line; nudged apart when they collide."""
    if not ends:
        return
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (txt, _, c), y in zip(ends, ys):
        ax.annotate(txt, xy=(Hs[-1], y), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color=INK2, annotation_clip=False)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


# ---------------------------------------------------------------- figures
def fig_vs_heads(data, out):
    methods = ['MHKSA', 'best_head_loo', 'CoRP_Ens_percentile']   # oracle (picked on the test OOD sets) is left out of the plots
    fig, axes = plt.subplots(2, len(data), figsize=(5.4 * len(data), 7.2), squeeze=False)
    for j, (mode, title, d) in enumerate(data):
        Hs = heads_of(d)
        for i, (col, ylab) in enumerate([(0, 'FPR95 (%) ↓'), (1, 'AUROC (%) ↑')]):
            ax = axes[i, j]
            baseline(ax, d, col)
            ends = []
            for m in methods:
                r = series(ax, d, m, Hs, col)
                if r is not None:
                    ends.append((f'{r[-1, 0]:.2f}', r[-1, 0], STYLE[m][1]))
            ax.set_xticks(Hs)
            ax.set_xlim(Hs[0] - 0.4, Hs[-1] + 0.4)
            ax.set_ylabel(ylab)
            ax.set_title(f'{title} · {ylab.split()[0]}' + base_txt(d, col))
            if i == 1:
                ax.set_xlabel('Number of heads H')
            label_ends(ax, Hs, ends)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    bases = ', '.join(f'{t} {base_label(t)}' for _, t, _ in data)
    fig.text(0.5, -0.01, f'CIFAR10 · ResNet18. Mean ± std over seeds, averaged over 5 OOD sets. '
             f'Base γ ({bases}) is the H = 1 head; H ≥ 2 uses base × [0.3, 3] (log-spaced).',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(h_pad=1.5, w_pad=3)
    save(fig, out, 'fig1_vs_heads.png')


def fig_mhksa_parts(data, out):
    fig, axes = plt.subplots(2, len(data), figsize=(5.4 * len(data), 6.6), squeeze=False,
                             gridspec_kw={'height_ratios': [1.6, 1]})
    for j, (mode, title, d) in enumerate(data):
        Hs = heads_of(d)
        for i, methods in enumerate([['MHKSA', 'MHKSA_e1only'], ['MHKSA_e2only']]):
            ax = axes[i, j]
            if i == 0:
                baseline(ax, d, 0)
            ends = []
            for m in methods:
                r = series(ax, d, m, Hs, 0)
                if r is not None:
                    ends.append((f'{r[-1, 0]:.2f}', r[-1, 0], STYLE[m][1]))
            ax.set_xticks(Hs)
            ax.set_xlim(Hs[0] - 0.4, Hs[-1] + 0.4)
            ax.set_ylabel('FPR95 (%) ↓')
            ax.set_title(f'{title} · ' + ('MHKSA vs stage-1 error' + base_txt(d) if i == 0
                                          else 'stage-2 error only (own scale)'))
            if i == 1:
                ax.set_xlabel('Number of heads H')
            label_ends(ax, Hs, ends)
    hs, ls = [], []
    for ax in axes[:, 0]:
        h, l = ax.get_legend_handles_labels()
        hs += h; ls += l
    fig.legend(hs, ls, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.text(0.5, -0.01, 'MHKSA score = −√(e1² + e2²); stage-1 only = e1, stage-2 only = e2. '
             'Stage-2 errors are on their own axis because they sit far above the others.',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(h_pad=1.5, w_pad=3)
    save(fig, out, 'fig2_mhksa_parts.png')


def fig_per_ood(data, out, H):
    bars = [('single_head', 1, 'Baseline (H=1 head)', BASE),
            ('MHKSA', H, f'MHKSA, H={H}', STYLE['MHKSA'][1]),
            ('best_head_loo', H, f'Best head LOO, H={H}', STYLE['best_head_loo'][1]),
            ('CoRP_Ens_percentile', H, f'Ensemble, H={H}', STYLE['CoRP_Ens_percentile'][1])]
    fig, axes = plt.subplots(1, len(data), figsize=(5.6 * len(data), 3.8), squeeze=False)
    for j, (mode, title, d) in enumerate(data):
        ax = axes[0, j]
        oods = [o for o in OOD_ORDER if any(k[2] == o for k in d)] or sorted({k[2] for k in d} - {'AVG'})
        groups = oods + ['AVG']
        bars_here = [b for b in bars if stat(d, b[0], b[1])[2]]
        w = 0.8 / len(bars_here)
        x = np.arange(len(groups))
        for k, (m, h, lab, c) in enumerate(bars_here):
            v = np.array([stat(d, m, h, g)[:2] for g in groups])
            ax.bar(x + (k - (len(bars_here) - 1) / 2) * w, v[:, 0], w, yerr=v[:, 1], color=c, label=lab,
                   edgecolor=SURFACE, linewidth=1.5, error_kw={'elinewidth': 1, 'capsize': 2, 'ecolor': INK2},
                   zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([g if g != 'places365' else 'Places365' for g in groups])
        ax.get_xticklabels()[-1].set_fontweight('bold')
        ax.grid(axis='x', visible=False)
        ax.set_ylabel('FPR95 (%) ↓')
        ax.set_title(f'{title} · per OOD set (H={H})')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.tight_layout(w_pad=3)
    save(fig, out, f'fig3_per_ood_H{H}.png')


def summary_md(data, out, H):
    lines = ['# Results summary (mean ± std over seeds; FPR95 ↓ / AUROC ↑, %)', '']
    for mode, title, d in data:
        Hs = heads_of(d)
        cols = [m for m in ['MHKSA', 'MHKSA_e1only', 'MHKSA_e2only', 'best_head_oracle', 'best_head_loo',
                            'CoRP_Ens_percentile', 'CoRP_Ens_zscore']
                if stat(d, m, Hs[-1])[2]]
        mu, sd, _ = stat(d, 'single_head', 1)
        au = stat(d, 'single_head', 1, col=1)[0]
        lines += [f'## {title}', '', f'Baseline (H=1, {base_label(title)}): FPR95 {mu:.2f} ± {sd:.2f} / AUROC {au:.2f}', '',
                  '| H | ' + ' | '.join(cols) + ' |', '| --- |' + ' --- |' * len(cols)]
        for h in Hs:
            cells = []
            for m in cols:
                f, fs, n = stat(d, m, h)
                a = stat(d, m, h, col=1)[0]
                cells.append(f'{f:.2f} ± {fs:.2f} / {a:.2f}' if n else '-')
            lines.append(f'| {h} | ' + ' | '.join(cells) + ' |')
        lines += ['', f'Per OOD set, FPR95 at H={H}:', '', '| method | ' + ' | '.join(OOD_ORDER) + ' |',
                  '| --- |' + ' --- |' * len(OOD_ORDER)]
        for m, h, lab in [('single_head', 1, 'Baseline (H=1)'), ('MHKSA', H, f'MHKSA (H={H})'),
                          ('best_head_loo', H, f'Best head LOO (H={H})'),
                          ('CoRP_Ens_percentile', H, f'Ensemble percentile (H={H})')]:
            if stat(d, m, h)[2]:
                lines.append(f'| {lab} | ' + ' | '.join('{:.2f} ± {:.2f}'.format(*stat(d, m, h, o)[:2])
                                                       for o in OOD_ORDER) + ' |')
        lines.append('')
    path = os.path.join(out, 'summary.md')
    open(path, 'w').write('\n'.join(lines))
    print('saved', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='logs')
    ap.add_argument('--ce', default=None, help='run folder for CE (default: newest logs/*-ce-*)')
    ap.add_argument('--supcon', default=None, help='run folder for SupCon (default: newest logs/*-supcon-*)')
    ap.add_argument('--h', type=int, default=2, help='H used for the per-OOD bar chart')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = []
    for mode, title in MODES:
        run = getattr(a, mode) or find_run(a.logs, mode)
        if run and glob.glob(os.path.join(run, 'results*.csv')):
            print(f'[{mode}] {run}')
            gb = os.path.basename(os.path.normpath(run)).split('-gb')
            BASE_GAMMA[title] = f"γ={gb[1].split('-')[0].split('_')[0]}" if len(gb) > 1 else 'median γ'
            data.append((mode, title, load(run)))
    if not data:
        raise SystemExit(f'no results_*.csv found under {a.logs}/')

    fig_vs_heads(data, a.out)
    fig_mhksa_parts(data, a.out)
    fig_per_ood(data, a.out, a.h)
    summary_md(data, a.out, a.h)


if __name__ == '__main__':
    main()

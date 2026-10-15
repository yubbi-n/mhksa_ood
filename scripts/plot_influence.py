"""Leave-one-head-out bars from `diagnose.py influence` reports (one panel per report).

    python scripts/plot_influence.py results/markdown/influence_ce_median_m0.3-3.0.md results/markdown/influence_ce_median_m0.3-10.md \
        results/markdown/influence_supcon_median_m0.3-3.0.md results/markdown/influence_supcon_median_m0.3-10.md
    python scripts/plot_influence.py ... --rule mhksa        # MHKSA (stage 2 refitted per subset) instead of the ensemble

Bar = FPR95 change when that head is left out (negative = removing it helps = harmful head).
x = head gamma as a multiple of the median gamma (head 0 is 0.3x).  -> results/graph/influence_loo_<rule>.png
"""
import os
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
HARM, HELP = '#e34948', '#2a78d6'
RULES = {'zmean': 'z-score mean (ensemble)', 'mhksa': 'MHKSA (stage 2 refitted per head subset)'}


def parse(path, rule):
    txt = open(path).read()
    head = txt.splitlines()[0]
    mode = 'SupCon' if 'supcon' in head.lower() else 'CE'
    m = re.search(r'-m([\d.]+)_([\d.]+)(log|lin)', head)
    rng = f'× [{m.group(1)}, {float(m.group(2)):g}]' if m else ''
    sec = txt[txt.index(f'## rule: {RULES[rule]}'):]
    rows = re.findall(r'^\| (\d+) \| ([\d.]+) \| ([\d.]+) \|.*\| ([+-][\d.]+) \|$', sec.split('### ③')[0], re.M)
    g = np.array([float(r[1]) for r in rows])
    return f'{mode} · γ_med {rng}', g / (g[0] / 0.3), np.array([float(r[3]) for r in rows]), np.array([float(r[2]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reports', nargs='+')
    ap.add_argument('--rule', default='zmean', choices=list(RULES))
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    n = len(a.reports)
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots((n + 1) // cols, cols, figsize=(6.2 * cols, 3.4 * ((n + 1) // cols)), squeeze=False,
                             facecolor=SURFACE)
    lim = 0
    for ax, path in zip(axes.flat, a.reports):
        title, mult, d, single = parse(path, a.rule)
        x = np.arange(len(d))
        ax.bar(x, d, 0.7, color=[HARM if v < 0 else HELP for v in d], zorder=3)
        ax.axhline(0, color=INK2, lw=1, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m:.2g}×' for m in mult], fontsize=8.5)
        best = single.argmin()
        ax.get_xticklabels()[best].set_fontweight('bold')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', color=INK)
        ax.set_facecolor(SURFACE)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.set_ylabel('FPR95 change if removed', color=INK2, fontsize=9)
        lim = max(lim, np.abs(d).max())
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat[:n]:
        ax.set_ylim(-lim * 1.15, lim * 1.15)
        ax.set_xlabel('head γ / median γ', color=INK2, fontsize=9)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HARM), plt.Rectangle((0, 0), 1, 1, color=HELP)],
               labels=['harmful (removing it lowers FPR)', 'needed (removing it raises FPR)'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.02, f'Rule: {RULES[a.rule]}, H = 10, mean over 5 seeds. Bold tick = best single head.',
             ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout()
    os.makedirs(os.path.join(a.out, 'graph'), exist_ok=True)
    path = os.path.join(a.out, 'graph', f'influence_loo_{a.rule}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    print('saved', path)


if __name__ == '__main__':
    main()

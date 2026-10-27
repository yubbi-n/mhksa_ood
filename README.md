# multihead_kpca

Multi-kernel (multi-head RFF) extensions of CoRP (Kernel PCA for OOD detection) on cached CIFAR10 features.

```
configs/
  base.yaml              all defaults (heads 1~10, gamma = gamma_med x [0.3, 3] log, M 2048, seeds 0~4)
  c10_r18_ce.yaml        CE      : evr 0.8, MHKSA 0.9 x 0.9
  c10_r18_supcon.yaml    SupCon  : evr 0.7, MHKSA 0.84 x 0.84
src/
  config.py      YAML (_base_ inheritance) + --set overrides
  data.py        cached feature loading, cosine normalisation, calib split
  kernels.py     RFF, median heuristic, gamma schedule
  model.py       PCA, KernelHead, MultiHead(stage 1), MHKSA [2], CoRPEnsemble [3], MultiKernelOOD
  selection.py   best single head [2-variant]: oracle / leave-one-OOD-out
  metrics.py     FPR95 / AUROC (uses ood-kernel-pca/metrics.py when on PYTHONPATH)
  utils.py       seed, logger, checkpoint, csv
train.py         fit on ID train -> checkpoints/<run>/seed{s}_H{H}.pkl
evaluate.py      load checkpoints -> logs/<run>/{eval.log, results.csv} + summary table
scripts/run_c10.sh, run_parallel.sh, plot_results.py, diagnose.py, visualize_images.py
```

## Run
```bash
export PYTHONPATH=/path/to/ood-kernel-pca:$PYTHONPATH     # for the paper's metrics.py
python train.py    --config configs/c10_r18_ce.yaml --set data.cache_dir=/path/to/ood-kernel-pca/cache
python evaluate.py --config configs/c10_r18_ce.yaml --set data.cache_dir=/path/to/ood-kernel-pca/cache
# or
bash scripts/run_c10.sh
```
Faster: one process per seed in parallel (finished (seed, H) checkpoints are skipped, so a killed run can be restarted):
```bash
CFG=configs/c10_r18_ce.yaml FEAT_CACHE=/path/to/cache KPCA_REPO=/path/to/ood-kernel-pca NPROC=32 \
  bash scripts/run_parallel.sh run.methods=[mhksa]
```
Method 2 (MHKSA) only: add `run.methods=[mhksa]` to evaluate.py (training is identical; outputs go to `eval_mhksa.log`, `results_mhksa.csv`).
Head-combination variants (`src/combine.py`), on existing checkpoints: `--set run.methods=[combine]` → `eval_combine.log`, `results_combine.csv`; `plot_results.py` then also writes `fig4_combine.png`.
Method 3 (ensemble) only, on existing checkpoints (no retraining): `python evaluate.py --config ... --set run.methods=[ensemble]` (add `kernel.gamma_base=2` etc. for a fixed-base run) → `eval_ensemble.log`, `results_ensemble.csv`.

Tables and figures from the result CSVs (needs `matplotlib`; one script for all of them):
```bash
python scripts/plot_results.py --ce logs/<ce run> --supcon logs/<supcon run>
# + a reference line (H = 1 single head of another run, e.g. the tuned-gamma runs) in fig5:
python scripts/plot_results.py --ce logs/<ce run> --supcon logs/<supcon run> \
    --ref-ce logs/<ce run>-gb2 --ref-supcon logs/<supcon run>-gb1 --figs vs_ref
```
→ `fig1_vs_heads` (FPR / AUROC vs H), `fig2_mhksa_parts` (stage-1 / stage-2), `fig3_per_ood_H{h}` (bars, `--h`),
`fig4_combine` (if `combine` was evaluated), `fig5_{ce,supcon}_vs_ref` (`--compare` methods vs the reference line),
`fig6_{mode}_{method}_per_ood` (one line per OOD set, `--per-ood` methods), `summary.md`, `per_dataset.md`.
`--figs` picks what is written: `vs_heads, mhksa_parts, per_ood, combine, summary` by default; `vs_ref` (fig5),
`per_ood_lines` (fig6) and `per_dataset` only when listed; `all` for everything.
Figures go to `results/graph/`, tables to `results/markdown/` (`--out` changes `results`); every file name ends with a
tag naming the runs (e.g. `_median_m0.3-10`, `--tag` to override). Without `--ce` / `--supcon` the newest
`logs/*-ce-*` / `logs/*-supcon-*` are used.

Checkpoint-based analyses (largest-H checkpoints):
```bash
python scripts/diagnose.py heads     --mode ce --cache /path/to/cache [--set kernel.mult_max=10] [--gamma-base 2]
python scripts/diagnose.py influence --mode ce --cache /path/to/cache [--set kernel.mult_max=10] [--gamma-base 2]
# -> results/markdown/<heads|influence>_<mode>_<median|gb..>_m<min>-<max>.md (also printed)
```
`heads`: single heads per OOD set, raw-sum / z-mean / z-max on the same heads, oracle pair and greedy subsets
(upper bound), head score correlations. `influence`: per-head weight / signal / noise shares, leave-one-head-out,
better vs worse half. Oracle subsets and halves are chosen on the test OOD sets: analysis only, not methods.
`python scripts/plot_influence.py <influence reports...>` draws the leave-one-head-out bars (`results/graph/influence_loo_<rule>.png`).

Image-level views (`torchvision` for images; `heatmap` also needs `torch`, the ood-kernel-pca repo and its ResNet18 checkpoint):
```bash
# grids of real images: most / least OOD-like, fixed / broken vs baseline, ID false alarms (+ picked.csv, summary.txt)
python scripts/visualize_images.py examples --config configs/c10_r18_ce.yaml \
    --set data.cache_dir=$HOME/ood-kernel-pca/cache --H 2 --data_root ~/data
# Grad-CAM of the OOD score (layer3, 8x8): image | baseline | MHKSA
python scripts/visualize_images.py heatmap --config configs/c10_r18_ce.yaml \
    --set data.cache_dir=$HOME/ood-kernel-pca/cache --H 2 --data_root ~/data \
    --kpca_repo ~/ood-kernel-pca --ckpt ~/ood-kernel-pca/save/CIFAR10/R18/ce/checkpoint_100.pth.tar
```
One image across runs (image | baseline | MHKSA H=1..10), auto-picking an image the baseline misses:
```bash
python scripts/visualize_images.py compare --config configs/c10_r18_ce.yaml \
    --set data.cache_dir=$HOME/ood-kernel-pca/cache --dataset SVHN \
    --ckpt ~/ood-kernel-pca/save/CIFAR10/R18/ce/checkpoint_100.pth.tar      # or --index 123
```
`heatmap`/`compare` re-extract features from the images and prints how far they are from the cache (should be ~1e-6),
which checks the checkpoint, data path and image order.

Fixed base γ instead of the median heuristic (e.g. the paper's tuned CoRP γ: CE 2, SupCon 1); heads = base × [0.3, 3],
results go to a separate `...-gb<γ>` folder:
```bash
CFG=configs/c10_r18_ce.yaml FEAT_CACHE=... KPCA_REPO=... bash scripts/run_parallel.sh kernel.gamma_base=2 run.methods=[mhksa,best_head]
```

Overrides: `--set kernel.heads=[1,5,10] run.seeds=[0] kernel.gamma_spacing=linear model.mhksa_evr1=0.95`

## Methods
| name | description |
|---|---|
| `single_head` | each head alone (= CoRP with that gamma) |
| `MHKSA` | [2] RFF_h → PCA_h → concat coords → PCA; score = −√(e1² + e2²) |
| `MHKSA_e1only / e2only` | stage-1 / stage-2 error only (analysis) |
| `best_head_oracle` | [2-variant] best head by test-OOD avg FPR (upper bound) |
| `best_head_loo` | [2-variant] head chosen on the other OOD sets (no test-set peeking, but still uses OOD data) |
| `CoRP_Ens_percentile` | [3] per-head error → percentile of the ID train errors of that head → equal-weight sum (linear beyond the ID range, so OOD samples past every ID error are not tied) |
| `CoRP_Ens_zscore` | [3] same, with (error − ID mean) / ID std per head |
| `Comb_MHKSA_norm` | MHKSA with every stage-1 residual and the stage-2 residual divided by its ID RMS (all H+1 terms weigh the same); `_e1only`: stage-1 terms only |
| `Comb_Ens_max` | ensemble with max over heads of the per-head z-score instead of the mean |
| `MHKSA_s1corp_s2<r>` | MHKSA with stage 1 = the CoRP ratio (0.8 CE / 0.7 SupCon, same residuals as the single heads) and stage 2 refitted with ratio r on ID train; r = 1.0 = no stage 2 (`run.methods=[mhksa_corp]`, ratios `model.mhksa_corp_s2`, default [CoRP ratio, 0.9, 1.0]) |

Gamma (senior's MHKSA): γ_h = γ_med · m_h, γ_med = 1/(2·med²) from the median heuristic (once per seed, shared by all H).
H = 1 → m = 1 (pure median heuristic); H ≥ 2 → m_h over [mult_min, mult_max] = [0.3, 3] (log-spaced).

PCA ratios: `exp_var_ratio` (0.8 CE / 0.7 SupCon) for single head / best head / ensemble (= CoRP).
MHKSA uses `mhksa_evr1`, `mhksa_evr2` (0.9 / 0.9 CE, 0.84 / 0.84 SupCon) so that stage1 × stage2 ≈ exp_var_ratio.

## ImageNet-1K (ResNet50, CE)
Features are extracted on the server GPU, streaming ImageNet from the Hugging Face hub one shard at a time
(needs ~1 GB free while running; accept the dataset terms and `huggingface-cli login` first):
```bash
python scripts/extract_imagenet.py train --cache ./cache --per-class 200 --gpu 0   # 200k train images
python scripts/extract_imagenet.py val   --cache ./cache --gpu 0
python scripts/extract_imagenet.py ood   --cache ./cache --gpu 0 --ood iNaturalist=<dir> SUN=<dir> Places=<dir> Textures=<dir>
python train.py --config configs/in1k_r50_ce.yaml --set data.cache_dir=./cache kernel.mult_max=10
```

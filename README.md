# multihead_kpca

Multi-kernel (multi-head RFF) extensions of CoRP (Kernel PCA for OOD detection) on cached CIFAR10 features.

```
configs/
  base.yaml              all defaults (heads 1~10, gamma 0.3~3 log, M 2048, seeds 0~4)
  c10_r18_ce.yaml        CE      : evr 0.8
  c10_r18_supcon.yaml    SupCon  : evr 0.7
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
scripts/run_c10.sh
```

## Run
```bash
export PYTHONPATH=/path/to/ood-kernel-pca:$PYTHONPATH     # for the paper's metrics.py
python train.py    --config configs/c10_r18_ce.yaml --set data.cache_dir=/path/to/ood-kernel-pca/cache
python evaluate.py --config configs/c10_r18_ce.yaml --set data.cache_dir=/path/to/ood-kernel-pca/cache
# or
bash scripts/run_c10.sh
```
Overrides: `--set kernel.heads=[1,5,10] run.seeds=[0] kernel.gamma_spacing=linear model.exp_var_ratio2=0.9`

## Methods
| name | description |
|---|---|
| `single_head` | each head alone (= CoRP with that gamma) |
| `MHKSA` | [2] RFF_h → PCA_h → concat coords → PCA; score = −√(e1² + e2²) |
| `MHKSA_e1only / e2only` | stage-1 / stage-2 error only (analysis) |
| `best_head_oracle` | [2-variant] best head by test-OOD avg FPR (upper bound) |
| `best_head_loo` | [2-variant] head chosen on the other OOD sets (fair) |
| `CoRP_Ens_percentile` | [3] per-head error → ID percentile → equal-weight sum |

Gamma: H = 1 → median heuristic (γ = 1/(2·med²)); H ≥ 2 → H values over [gamma_min, gamma_max].

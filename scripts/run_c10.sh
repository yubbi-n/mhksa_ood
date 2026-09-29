#!/bin/bash
# Run from the repo root.
# FEAT_CACHE : ood-kernel-pca feature cache,  KPCA_REPO : ood-kernel-pca root (for metrics.py)
FEAT_CACHE=${FEAT_CACHE:-../ood-kernel-pca/cache}
KPCA_REPO=${KPCA_REPO:-../ood-kernel-pca}
export PYTHONPATH=$KPCA_REPO:$PYTHONPATH

for CFG in configs/c10_r18_ce.yaml configs/c10_r18_supcon.yaml; do
  python train.py    --config $CFG --set data.cache_dir=$FEAT_CACHE "$@"
  python evaluate.py --config $CFG --set data.cache_dir=$FEAT_CACHE "$@"
done

# gamma range search later, e.g.
#   bash scripts/run_c10.sh kernel.gamma_min=0.5 kernel.gamma_max=5 run.tag=_range2

#!/bin/bash
# Train one process per seed in parallel, then evaluate.  Run from the repo root.
#   CFG=configs/c10_r18_ce.yaml FEAT_CACHE=... KPCA_REPO=... bash scripts/run_parallel.sh [extra --set overrides]
# NPROC : cores to use in total (default 32) ; SEEDS : seeds to train (default "0 1 2 3 4")
# Already finished (seed, H) checkpoints are skipped (run.resume=true), so a killed run can be restarted.
CFG=${CFG:-configs/c10_r18_ce.yaml}
FEAT_CACHE=${FEAT_CACHE:-../ood-kernel-pca/cache}
KPCA_REPO=${KPCA_REPO:-../ood-kernel-pca}
NPROC=${NPROC:-32}
SEEDS=${SEEDS:-"0 1 2 3 4"}
export PYTHONPATH=$KPCA_REPO:$PYTHONPATH

N=$(echo $SEEDS | wc -w)
T=$(( NPROC / N )); [ $T -lt 1 ] && T=1
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
echo "$(date '+%F %T') train $CFG : seeds [$SEEDS], $T threads each"

for S in $SEEDS; do
  python train.py --config $CFG --set data.cache_dir=$FEAT_CACHE run.seeds=[$S] run.resume=true "$@" \
    > train_$(basename $CFG .yaml)_seed$S.out 2>&1 &
done
wait

export OMP_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC
SEED_LIST=$(echo $SEEDS | tr ' ' ',')
echo "$(date '+%F %T') evaluate $CFG"
python evaluate.py --config $CFG --set data.cache_dir=$FEAT_CACHE run.seeds=[$SEED_LIST] "$@"
echo "$(date '+%F %T') done"

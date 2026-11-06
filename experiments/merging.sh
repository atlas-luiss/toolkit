#!/usr/bin/env bash
# MagMax setting, 3 seeds, sequentially on one GPU. Logs: runs/merging/magmax_cifar100_s<seed>.out
cd "$(dirname "$0")/.."                       # run from the repository root
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
PY=${PYTHON:-python}
CIFAR=${CL_DATA:-./data}/cifar100
mkdir -p runs/merging
for s in 0 1 2; do
  echo "[$(date +%T)] start seed $s"
  CUDA_VISIBLE_DEVICES=$1 $PY experiments/merging_magmax.py $s > runs/merging/magmax_cifar100_s$s.out 2>&1
  echo "[$(date +%T)] done seed $s exit $?"
done
echo DONE

#!/usr/bin/env bash
# 3 seeds (= 3 class orders) each. GPU 0: frozen-feature runs then L2P++; GPU 2: CODA-P; GPU 3: DualPrompt.
cd "$(dirname "$0")/.."                       # run from the repository root
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
PY=${PYTHON:-python}
CIFAR=${CL_DATA:-./data}/cifar100
mkdir -p runs/repro
r() { echo "[$(date +%T)] start $2 s$3 gpu$1"; CUDA_VISIBLE_DEVICES=$1 $PY experiments/repro.py $2 $3 > runs/repro/$2_s$3.out 2>&1; echo "[$(date +%T)] done $2 s$3 exit $?"; }
( for n in ncm-in21k ranpac-in21k ncm-in1k ranpac-in1k ncm-in21k-modelprep ranpac-in21k-modelprep; do
    for s in 0 1 2; do r 0 $n $s; done; done
  for s in 0 1 2; do r 0 l2p++ $s; done ) &
( for s in 0 1 2; do r 2 coda $s; done ) &
( for s in 0 1 2; do r 3 dualprompt $s; done ) &
wait
echo ALL DONE

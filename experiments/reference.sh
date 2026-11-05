#!/usr/bin/env bash
# README reference results: exactly the CLI commands shown in the README, toolkit defaults
# (ViT-B/16 IN21K, preprocess=model, AdamW lr 1e-3, batch 64, 1 epoch/task), 3 seeds.
# Usage: bash experiments/reference.sh <gpu> [names...]      logs: runs/reference/<name>_s<seed>/
cd /scratch/eric/projects/toolkit
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
GPU=$1; shift
C="--data cifar100 --root /scratch/eric/data/cifar100 --data-set download=False"
declare -A R=(
  [frozen-ranpac]="--strategy frozen --classifier ranpac --epochs 0"
  [frozen-klda]="--strategy frozen --classifier klda --epochs 0"
  [frozen-lda]="--strategy frozen --classifier lda --epochs 0"
  [frozen-fecam]="--strategy frozen --classifier fecam --epochs 0"
  [frozen-ncm]="--strategy frozen --classifier ncm --epochs 0"
  [naive-slca]="--strategy naive --classifier slca"
  [er-ace]="--strategy er-ace --set memory_size=2000"
  [er]="--strategy er --set memory_size=2000"
  [lora-er]="--strategy lora-er --set memory_size=2000"
  [naive]="--strategy naive"
  [routed-adaptformer-hide]="--experts adaptformer --router hide"
  [routed-lora-hide]="--experts lora --router hide"
  [routed-lora-hide-lda]="--experts lora --router hide --classifier lda"
  [routed-lora-hide-ncm]="--experts lora --router hide --classifier ncm"
  [routed-lora-hide-fecam]="--experts lora --router hide --classifier fecam"
  [routed-prompt-hide]="--experts prompt --router hide"
  [coda-prompt]="--strategy coda-prompt"
)
NAMES=${@:-${!R[@]}}
for n in $NAMES; do for s in 0 1 2; do
  echo "[$(date +%T)] $n s$s"
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m cl_toolkit $C ${R[$n]} --seed $s --out runs/reference/${n}_s$s \
    > runs/reference/${n}_s$s.out 2>&1 || echo "FAILED $n s$s"
done; done
echo DONE

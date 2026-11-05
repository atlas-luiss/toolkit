"""Reproduce published Split CIFAR-100 (10 tasks, Class-IL) numbers in the papers' own settings.

Usage: python experiments/repro.py <name> <seed>        logs: runs/repro/<name>_s<seed>/{log.txt,config.json,results.json}

Settings and the numbers they are compared with (see VERIFICATION.md):
  ncm-in21k / ranpac-in21k   frozen ViT-B/16 IN21K, no training      RanPAC paper Tab. 1/A4: NCM 83.4, no-Phase-1 89.0
  ncm-in1k  / ranpac-in1k    same with IN21K->IN1K weights           (weights ablation)
  coda / dualprompt / l2p++  CODA-Prompt repo config: ViT-B/16 IN1K (timm vit_base_patch16_224 of 2022 =
                             augreg_in21k_ft_in1k), Adam lr 1e-3, no wd, bs 128, cosine, 20 epochs/task
                             CODA paper Tab. 2a: CODA-P 86.25, DualPrompt 83.05, L2P++ 82.50
"""
import sys

import torch

import cl_toolkit as clt

IN21K, IN1K = "vit_base_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k_ft_in1k"
CIFAR = dict(data="cifar100", root="/scratch/eric/data/cifar100", scenario="class", num_tasks=10,
             batch_size=128, num_workers=8)
PILOT = {"download": False, "preprocess": "pilot"}      # transforms of the RanPAC / APER code base
CODA = {"download": False, "preprocess": "coda"}        # transforms of the CODA-Prompt code base
MODEL = {"download": False, "preprocess": "model"}      # toolkit default (for the preprocessing ablation)
PROMPT = dict(model=IN1K, epochs=20, lr=1e-3, optimizer_cls=torch.optim.Adam, weight_decay=0.0, scheduler="cosine",
              data_kwargs=CODA)

RUNS = {
    "ncm-in21k": dict(model=IN21K, strategy="simplecil", epochs=0, data_kwargs=PILOT),
    "ncm-in1k": dict(model=IN1K, strategy="simplecil", epochs=0, data_kwargs=PILOT),
    "ranpac-in21k": dict(model=IN21K, strategy="ranpac", epochs=0, data_kwargs=PILOT),
    "ranpac-in1k": dict(model=IN1K, strategy="ranpac", epochs=0, data_kwargs=PILOT),
    "ncm-in21k-modelprep": dict(model=IN21K, strategy="simplecil", epochs=0, data_kwargs=MODEL),
    "ranpac-in21k-modelprep": dict(model=IN21K, strategy="ranpac", epochs=0, data_kwargs=MODEL),
    "coda": dict(strategy="coda-prompt", **PROMPT),
    "dualprompt": dict(strategy="dualprompt", **PROMPT),
    "l2p++": dict(strategy="l2p", **PROMPT),
}

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    clt.run(**CIFAR, **RUNS[name], seed=seed, out_dir=f"runs/repro/{name}_s{seed}")

"""
Shared setup for real-benchmark examples.

ViT-B/16 (frozen backbone) on Split-CIFAR-100, 5 tasks × 20 classes.
Import this from each strategy example.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import torch
import numpy as np

from ctk.benchmarks import SplitCIFAR100
from ctk.models     import ViTModel

DATA_ROOT   = "/leonardo_work/IscrC_ConFT/eric/data"
MODEL_CACHE = "/leonardo_work/IscrC_ConFT/eric/hf_cache"
NUM_TASKS   = 5
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.environ.setdefault("HF_HOME", MODEL_CACHE)

SCENARIO_KW = dict(batch_size=BATCH_SIZE, num_workers=4,
                   pin_memory=(DEVICE == "cuda"))

TRAINER_KW  = dict(
    optimizer_cls=torch.optim.AdamW,
    lr=LR,
    weight_decay=1e-4,
    epochs_per_task=EPOCHS,
    device=DEVICE,
    verbose=True,
)


def make_bench():
    return SplitCIFAR100(root=DATA_ROOT, num_tasks=NUM_TASKS,
                         image_size=224, seed=42, download=False)


def make_model(freeze_backbone=True):
    return ViTModel(
        model_name="vit_base_patch16_224",
        pretrained=True,
        freeze_backbone=freeze_backbone,
        device=torch.device(DEVICE),
    )


def print_header(strategy_name: str, extra: str = ""):
    print("=" * 62)
    print(f"  Split-CIFAR-100 | ViT-B/16 | {strategy_name} | Class-IL")
    if extra:
        print(f"  {extra}")
    print("=" * 62)
    print(f"  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  "
          f"Epochs/task: {EPOCHS}  |  BS: {BATCH_SIZE}")
    print()


def print_results(results: dict, elapsed: float):
    mat = results["accuracy_matrix"]
    n   = mat.shape[0]

    print("\n" + "=" * 62)
    print("ACCURACY MATRIX  (row=after task t, col=eval task j)")
    header = "       " + "  ".join(f"  T{j}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"After T{i}:"
        for j in range(n):
            v = mat[i, j]
            row += f"  {'---' if (v == 0 and j > i) else f'{v:4.1f}'}"
        print(row)
    print()
    print(f"  Avg accuracy   : {results['avg_accuracy']:.2f}%")
    print(f"  Avg forgetting : {results['avg_forgetting']:.2f}%")
    print(f"  Wall time      : {elapsed/60:.1f} min")
    print("=" * 62)

"""
Joint Training  ·  ImageNet-R  ·  upper bound
=============================================
Pools all task data and trains in one shot — no forgetting possible.
This is the ceiling. No continual learning method should exceed it.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, MODEL_CACHE, DEVICE, print_results

from ctk.benchmarks           import ImageNetR
from ctk.models               import DINOModel
from ctk.scenarios.finetuning import JointTraining
from ctk.strategies           import NaiveStrategy
from ctk.trainers             import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
# JointTraining merges all tasks into a single shuffled dataset.
# The model sees every class simultaneously — no forgetting can occur.
scenario = JointTraining(bench, batch_size=64, num_workers=4,
                         pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Joint Training  |  all tasks merged into one dataset")

# ── Strategy ──────────────────────────────────────────────────────────────────
# Plain cross-entropy is all that's needed when there's no forgetting problem.
strategy = NaiveStrategy()
print(f"Strategy   →  Naive  |  no CL overhead needed for joint training\n")

# ── Train ─────────────────────────────────────────────────────────────────────
t0      = time.time()
results = ContinualTrainer(
    model, scenario, strategy,
    optimizer_cls   = torch.optim.AdamW,
    lr              = 1e-4,
    weight_decay    = 1e-4,
    epochs_per_task = 2,
    device          = DEVICE,
    verbose         = True,
).fit()

# ── Results ───────────────────────────────────────────────────────────────────
# Expect: avg_forgetting ≈ 0%, avg_accuracy is the best achievable with this model.
# The gap between this and naive.py is your "continual learning challenge".
print_results(results, time.time() - t0)

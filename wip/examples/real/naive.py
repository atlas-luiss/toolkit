"""
Naive Fine-tuning  ·  ImageNet-R  ·  lower bound
=================================================
Trains sequentially on each task with no anti-forgetting mechanism.
Run this first — every other strategy should beat it.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, MODEL_CACHE, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import NaiveStrategy
from ctk.trainers   import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
# ImageNet-R: 200 classes, split into 10 sequential tasks of 20 classes each.
# The model sees each task once and is never allowed to revisit previous data.
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
# DINOv2-B/14: frozen self-supervised ViT backbone.
# Only the incremental classification head is trained.
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
# Class-Incremental: task identity is NOT given at test time.
# The model must distinguish all 200 classes from a single shared head.
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental  |  task ID hidden at inference")

# ── Strategy ──────────────────────────────────────────────────────────────────
# No strategy. Plain cross-entropy on the current task only.
# New gradients freely overwrite weights that were useful for old tasks.
strategy = NaiveStrategy()
print(f"Strategy   →  Naive  |  no anti-forgetting mechanism\n")

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
# Expect: high forgetting, low avg accuracy — the model mostly remembers task 9.
# This is your lower bound. Compare every other experiment against these numbers.
print_results(results, time.time() - t0)

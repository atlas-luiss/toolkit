"""
LoRA  ·  ImageNet-R  ·  Task-IL
=================================
Same LoRA setup as lora_dino.py but in the easier Task-Incremental setting
where the task identity IS provided at test time. The correct adapter can be
activated for each test sample, giving the model a perfect routing signal.

Comparing Task-IL (this file) vs Class-IL (lora_dino.py) with identical
hyperparameters shows the exact cost of not knowing the task at inference.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import TaskIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainers   import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
# Task-Incremental: task identity IS given at test time.
# Each task has its own output head — the model only ranks classes within a task.
scenario = TaskIncremental(bench, batch_size=64, num_workers=4,
                           pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Task-Incremental  |  task ID provided at inference")

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy = LoRAStrategy(rank=8, lora_alpha=16, merge_after_task=False)
print(f"Strategy   →  LoRA  |  rank=8  |  α=16  |  one adapter per task\n")

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
# Task-IL accuracy should be noticeably higher than lora_dino.py (Class-IL).
# The difference quantifies how much task identity is worth at test time.
print_results(results, time.time() - t0)

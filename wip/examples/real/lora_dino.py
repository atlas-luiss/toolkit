"""
LoRA on DINOv2  ·  ImageNet-R  ·  Class-IL
===========================================
Instead of fine-tuning all backbone weights, LoRA injects small trainable
rank-decomposition matrices alongside the frozen attention projections:

    W' = W + B · A     where  B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k},  r ≪ d

One adapter is created per task. The base weights are never modified, so
earlier adapters are unaffected by later training — no forgetting at the
weight level. Only ~0.5% of total parameters are trained per task.

In Class-IL, all adapters coexist at inference since task identity is unknown.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainers   import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
# freeze_backbone=True is required — LoRA adds its own trainable adapters,
# so the base weights must be frozen for the isolation guarantee to hold.
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental")

# ── Strategy ──────────────────────────────────────────────────────────────────
# rank            : low-rank dimension r — higher means more capacity per task
# lora_alpha      : scaling factor; effective LR scales as alpha / rank
# merge_after_task: fuse adapter into backbone weights after each task to save
#                   memory (irreversible — disable if you need per-task adapters)
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
# LoRA largely eliminates weight-level forgetting but the shared classification
# head can still be a bottleneck in Class-IL. See lora_replay_dino.py which adds
# a replay buffer to keep the head calibrated across all seen classes.
print_results(results, time.time() - t0)

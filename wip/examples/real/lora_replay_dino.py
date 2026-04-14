"""
LoRA + Replay on DINOv2  ·  ImageNet-R  ·  Class-IL
=====================================================
Combines two complementary ideas:

  · LoRA adapters  — per-task trainable parameters that don't interfere with
                     each other, preventing weight-level forgetting
  · Experience Replay — a memory buffer that mixes past examples into each
                        batch, keeping the shared classification head calibrated
                        across all seen classes

LoRA alone protects the backbone but the growing softmax head can still drift.
Replay alone trains all weights and may disturb adapter interactions.
Together they address both failure modes simultaneously.

This tends to be the strongest PEFT-based continual learning configuration.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAReplay
from ctk.trainers   import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental")

# ── Strategy ──────────────────────────────────────────────────────────────────
# rank         : LoRA low-rank dimension (same as lora_dino.py)
# memory_size  : replay buffer — 1000 ≈ 5 examples/class for 200 classes
# replay_ratio : fraction of each mini-batch drawn from the buffer
strategy = LoRAReplay(rank=8, lora_alpha=16, memory_size=1000, replay_ratio=0.5)
print(f"Strategy   →  LoRA + Replay  |  rank=8  |  memory=1000  |  replay_ratio=0.5\n")

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
# Expect this to outperform both lora_dino.py and er.py individually.
# The accuracy matrix diagonal should stay high even for the earliest tasks.
print_results(results, time.time() - t0)

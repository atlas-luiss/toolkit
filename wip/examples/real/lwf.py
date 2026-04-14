"""
LwF (Learning without Forgetting)  ·  ImageNet-R
=================================================
Before training on a new task, LwF records what the model currently predicts
on the incoming data (the "dark knowledge"). A distillation loss then keeps
the new model's predictions close to those frozen reference outputs throughout
training — no past data needs to be stored.

    loss = CE(new task) + λ · KL(current_logits ‖ reference_logits)
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LwF
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
# lam         : distillation loss weight; increase if forgetting remains high
# temperature : softens the reference logits — higher T → softer, more informative targets
strategy = LwF(lam=1.0, temperature=2.0)
print(f"Strategy   →  LwF  |  λ=1.0  |  temperature=2.0\n")

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
# LwF works best when new tasks are visually similar to old ones.
# On diverse benchmarks like ImageNet-R the distillation signal may be weaker.
print_results(results, time.time() - t0)

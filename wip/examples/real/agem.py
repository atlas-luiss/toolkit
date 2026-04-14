"""
A-GEM (Averaged Gradient Episodic Memory)  ·  ImageNet-R
=========================================================
Before each gradient step, A-GEM samples a reference batch from its memory
buffer and checks whether the proposed update would increase the loss on those
past examples. If it would, the gradient is projected to the feasible
half-space (made orthogonal to the reference gradient).

This gives a formal guarantee: average loss on past tasks never increases.
The trade-off is conservatism — plasticity on the current task may suffer.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import AGEM
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
# memory_size    : past examples used as the reference set
# ref_sample_size: how many buffer examples to sample per update step
#                  larger → more accurate projection constraint, slower steps
strategy = AGEM(memory_size=2000, ref_sample_size=256)
print(f"Strategy   →  A-GEM  |  memory=2000  |  ref_sample=256\n")

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
# A-GEM provides a theoretical guarantee on average past-task loss.
# In practice, ER (er.py) often achieves higher accuracy — compare both.
print_results(results, time.time() - t0)

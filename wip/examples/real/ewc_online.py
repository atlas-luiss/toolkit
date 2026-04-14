"""
EWC Online  ·  ImageNet-R
==========================
Standard EWC stores one Fisher matrix per past task — memory grows with the
number of tasks. EWC Online solves this by maintaining a single running Fisher
updated with an exponential moving average after each task:

    F ← γ · F_old + (1 − γ) · F_new_task

Memory cost stays constant regardless of how many tasks are seen.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import EWCOnline
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
# gamma : EMA decay — 0.9 means older tasks gradually lose influence
#         set to 1.0 to accumulate all tasks equally (= EWC with one matrix)
strategy = EWCOnline(lam=1000.0, gamma=0.9, fisher_samples=500)
print(f"Strategy   →  EWC Online  |  λ=1000  |  γ=0.9  |  fisher_samples=500\n")

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
# Should match ewc.py in quality with lower memory footprint.
# On long task sequences (>10 tasks), EWC Online typically holds up better.
print_results(results, time.time() - t0)

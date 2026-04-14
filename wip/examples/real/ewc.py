"""
EWC (Elastic Weight Consolidation)  ·  ImageNet-R
==================================================
After each task, EWC estimates which parameters mattered most using the
Fisher information matrix. Future updates are penalised for moving those
parameters far from where they landed after the previous task.

    loss = CE(current task) + λ · Σ F_i · (θ_i − θ*_i)²

No memory buffer needed — this is a regularization-only method.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import EWC
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
# lam           : penalty strength — higher = less forgetting, less plasticity
# fisher_samples: samples used to estimate the Fisher after each task
#                 more samples → smoother estimate, slower consolidation step
strategy = EWC(lam=1000.0, fisher_samples=500)
print(f"Strategy   →  EWC  |  λ=1000  |  fisher_samples=500\n")

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
# EWC should show lower forgetting than naive.py.
# If forgetting is still high, try increasing lam (e.g. 5000, 10000).
print_results(results, time.time() - t0)

"""
MAS (Memory Aware Synapses)  ·  ImageNet-R
==========================================
Estimates parameter importance by measuring how much each weight influences
the L2 norm of the model's output — no labels required. Important parameters
are penalised from changing on future tasks, similar to EWC.

The key difference from EWC: importance is computed from the output magnitude
gradient (unsupervised), not the loss gradient (supervised). This means MAS
can consolidate even on unlabelled data after each task.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import MAS
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
# lam                : regularization strength
# importance_samples : samples used to compute the output-gradient importance
strategy = MAS(lam=1.0, importance_samples=500)
print(f"Strategy   →  MAS  |  λ=1.0  |  importance_samples=500\n")

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
# Results typically track EWC closely. MAS tends to be preferred when labels
# are unavailable at consolidation time or compute is limited.
print_results(results, time.time() - t0)

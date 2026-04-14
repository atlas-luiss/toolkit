"""
DER++ (Dark Experience Replay++)  ·  ImageNet-R
================================================
Extends ER by storing not just (x, y) but also the model's output logits
at the time each sample was added to the buffer. During replay it optimises
two additional terms alongside the standard cross-entropy:

    α · MSE(current_logits, stored_logits)   ← self-consistency over time
    β · CE(current_logits, stored_labels)    ← standard classification replay

The logit-matching term acts as self-distillation: it prevents the model from
drifting away from past predictions even for classes the current task doesn't
directly supervise.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import DERPP
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
# memory_size : buffer stores (x, y, logits) — slightly more memory than plain ER
# alpha       : weight of the logit MSE (self-consistency) term
# beta        : weight of the label CE (replay) term
strategy = DERPP(memory_size=2000, alpha=0.2, beta=0.5)
print(f"Strategy   →  DER++  |  memory=2000  |  α=0.2  |  β=0.5\n")

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
# DER++ typically outperforms ER (er.py) on benchmarks with drifting features.
# The stored logits capture richer information than hard labels alone.
print_results(results, time.time() - t0)

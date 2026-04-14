"""
GDumb (Greedy Sampler + Dumb Learner)  ·  ImageNet-R
=====================================================
GDumb ignores the stream ordering entirely. As data arrives it fills a
class-balanced buffer (greedy sampling). At the end of each task it
discards all online learning and retrains the model from scratch using
only the buffer contents.

Counterintuitively simple — yet often competitive with complex methods.
It proves that a well-balanced exemplar set with enough offline epochs
can match many sophisticated strategies.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import GDumb
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
# memory_size   : class-balanced buffer — slots grow as new classes arrive
# replay_epochs : offline retraining epochs on the buffer after each task
#                 should be ≥ the online training epochs to actually learn
# lr            : learning rate used for the offline replay phase
strategy = GDumb(memory_size=2000, replay_epochs=10, lr=1e-3)
print(f"Strategy   →  GDumb  |  memory=2000  |  replay_epochs=10\n")

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
# Online training is effectively wasted — results reflect the offline replay only.
# Increasing replay_epochs or memory_size directly improves performance.
print_results(results, time.time() - t0)

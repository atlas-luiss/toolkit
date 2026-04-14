"""
Experience Replay (ER)  ·  ImageNet-R
======================================
Maintains a fixed-size memory buffer of past examples using reservoir sampling
(every sample has equal probability of being kept). During each training step,
a fraction of the mini-batch is drawn from this buffer and mixed with new data.

This is the simplest rehearsal method — and often the strongest baseline.
If you're allowed to store any past data, start here.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import ExperienceReplay
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
# memory_size       : total examples stored; reservoir sampling keeps class balance
#                     2000 samples ≈ 10 examples/class across all 200 classes
# replay_batch_ratio: fraction of each batch filled from the buffer
#                     0.5 → 32 new samples + 32 replayed samples per step (bs=64)
strategy = ExperienceReplay(memory_size=2000, replay_batch_ratio=0.5)
print(f"Strategy   →  Experience Replay  |  memory=2000  |  replay_ratio=0.5\n")

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
# ER should substantially reduce forgetting compared to naive.py.
# To push further: increase memory_size, or try derpp.py for richer replay.
print_results(results, time.time() - t0)

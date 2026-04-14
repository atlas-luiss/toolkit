"""
HOW-TO: Basic Continual Learning
=================================
The minimal CL setup: benchmark → scenario → model → strategy → trainer.

NaiveStrategy applies plain cross-entropy with no forgetting protection.
Run this first to see catastrophic forgetting — avg_forgetting will be high.

Dataset : ImageNet-R  (200 classes, 10 tasks of 20 classes each)
Model   : DINOv2-B/14 with frozen backbone (only the head is trained)
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer                          # alias for ContinualTrainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import NaiveStrategy
from ctk.callbacks import ProgressiveResultsCallback

from _setup import make_benchmark, make_model, DEVICE

# ---------------------------------------------------------------------------
# 1. Benchmark — defines the sequence of tasks and their data
# ---------------------------------------------------------------------------
benchmark = make_benchmark(num_tasks=10)
print(f"Benchmark  →  ImageNet-R  |  {benchmark.num_tasks} tasks  "
      f"|  {benchmark.classes_per_task} classes/task  "
      f"|  {benchmark.total_classes} total")

# ---------------------------------------------------------------------------
# 2. Scenario — defines *how* tasks are presented and the head type
#
#   ClassIncremental : single growing classifier, no task oracle at test time
#   TaskIncremental  : one head per task, task ID known at test time
# ---------------------------------------------------------------------------
scenario = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))

# ---------------------------------------------------------------------------
# 3. Model — DINOv2-B/14 with frozen backbone
# ---------------------------------------------------------------------------
model = make_model()
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ---------------------------------------------------------------------------
# 4. Strategy — NaiveStrategy just does cross-entropy, nothing more
# ---------------------------------------------------------------------------
strategy = NaiveStrategy()

# ---------------------------------------------------------------------------
# 5. Callbacks (optional but useful)
# ---------------------------------------------------------------------------
callbacks = [
    ProgressiveResultsCallback(),   # prints accuracy matrix after each task
]

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
trainer = Trainer(
    model           = model,
    scenario        = scenario,
    strategy        = strategy,
    optimizer_cls   = torch.optim.AdamW,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    epochs_per_task = 1,            # set higher (e.g. 5) for better results
    device          = DEVICE,
    callbacks       = callbacks,
    verbose         = True,
)

results = trainer.fit()

# ---------------------------------------------------------------------------
# 7. Key CL metrics
# ---------------------------------------------------------------------------
print("\nCL Metrics:")
print(f"  avg_accuracy     : {results['avg_accuracy']:.1f}%")
print(f"  avg_forgetting   : {results['avg_forgetting']:.1f}%")
print(f"  backward_transfer: {results['backward_transfer']:.1f}%")

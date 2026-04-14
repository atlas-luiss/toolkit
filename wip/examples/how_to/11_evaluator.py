"""
HOW-TO: ContinualEvaluator
===========================
ContinualEvaluator runs post-hoc evaluation on a trained model across all
tasks — useful when you want to evaluate a checkpoint without re-running
the full training loop.

It also exposes compute_cl_metrics() so you can compute avg_accuracy,
avg_forgetting, and backward_transfer from any accuracy matrix you already
have (e.g. from a results file).

Dataset : ImageNet-R  (3 tasks)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch
import numpy as np

from ctk import Trainer
from ctk.trainers import ContinualEvaluator
from ctk.scenarios import ClassIncremental
from ctk.strategies import EWC

from _setup import make_benchmark, make_model, DEVICE

# ---------------------------------------------------------------------------
# 1. Train a model first
# ---------------------------------------------------------------------------
benchmark = make_benchmark(num_tasks=3)
scenario  = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                             pin_memory=(DEVICE == "cuda"))
model     = make_model()
strategy  = EWC(lam=500.0, fisher_samples=200)

trainer = Trainer(
    model, scenario, strategy,
    optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
    epochs_per_task=1, device=DEVICE, verbose=True,
)
train_results = trainer.fit()

print("Training done.\n")

# ---------------------------------------------------------------------------
# 2. Evaluate the trained model on all tasks
# ---------------------------------------------------------------------------
evaluator = ContinualEvaluator(
    model    = model,
    scenario = scenario,
    strategy = strategy,   # optional — used for strategy-specific eval hooks
)

eval_results = evaluator.evaluate_all()

# Pretty-print the per-task table
evaluator.print_report(eval_results)

# ---------------------------------------------------------------------------
# 3. Evaluate a single task
# ---------------------------------------------------------------------------
task0_result = evaluator.evaluate_task(task_id=0)
print(f"Task 0 only — accuracy: {task0_result['accuracy']:.1f}%  "
      f"loss: {task0_result['loss']:.4f}")

# ---------------------------------------------------------------------------
# 4. Compute CL metrics from any accuracy matrix
#    (handy if you have stored results from a previous run)
# ---------------------------------------------------------------------------
# Suppose these are accuracy values from a previous experiment:
stored_matrix = np.array([
    [80.0,   0.0,   0.0],   # after task 0: only task 0 evaluated
    [72.0,  75.0,   0.0],   # after task 1
    [65.0,  68.0,  78.0],   # after task 2: all tasks evaluated
])

metrics = evaluator.compute_cl_metrics(stored_matrix)
print(f"\nMetrics from stored accuracy matrix:")
print(f"  avg_accuracy     : {metrics['avg_accuracy']:.1f}%")
print(f"  avg_forgetting   : {metrics['avg_forgetting']:.1f}%")
print(f"  backward_transfer: {metrics['backward_transfer']:.1f}%")

"""
HOW-TO: Callbacks
==================
Callbacks let you plug monitoring, checkpointing, and control logic into
the training loop without touching the trainer code.

Hook points (called in this order per task):
  on_train_begin → on_task_begin → on_epoch_begin → on_batch_begin
  → on_before_step → on_batch_end → on_epoch_end → on_eval_begin
  → on_eval_end → on_task_end → on_train_end

Built-in callbacks shown here:
  ProgressiveResultsCallback  — prints accuracy matrix after each task
  EarlyStopping               — stops a task early when loss plateaus
  ModelCheckpointing          — saves best checkpoint to disk
  MetricLogger                — logs metrics to console or CSV
  WarmupScheduler             — linear LR warmup per task
  GradientClipping            — clips gradient norm before each step

Dataset : ImageNet-R  (3 tasks of ~67 classes each)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import NaiveStrategy
from ctk.callbacks import (
    ProgressiveResultsCallback,
    EarlyStopping,
    ModelCheckpointing,
    MetricLogger,
    WarmupScheduler,
    GradientClipping,
)

from _setup import make_benchmark, make_model, DEVICE

benchmark = make_benchmark(num_tasks=3)
scenario  = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                             pin_memory=(DEVICE == "cuda"))
model     = make_model()
strategy  = NaiveStrategy()

# ---------------------------------------------------------------------------
# Assemble the callback list
# ---------------------------------------------------------------------------
callbacks = [
    # Prints the accuracy matrix and forgetting/BWT after every task finishes
    ProgressiveResultsCallback(),

    # Stop training within a task if loss doesn't improve for 2 epochs
    # (resets automatically at the start of each task)
    EarlyStopping(monitor="loss", patience=2, mode="min"),

    # Save the best model checkpoint (by accuracy) and a per-task checkpoint
    ModelCheckpointing(
        checkpoint_dir        = "/tmp/ctk_checkpoints",
        monitor               = "accuracy",
        mode                  = "max",
        save_best_only        = True,
        save_task_checkpoints = True,
    ),

    # Log metrics to console and to a CSV file
    MetricLogger(
        log_to   = ["console", "csv"],
        csv_path = "/tmp/ctk_metrics.csv",
    ),

    # Linear warmup from 0 → lr over the first `warmup_epochs` epochs of each task
    WarmupScheduler(warmup_epochs=1),

    # Clip gradient norm to 1.0 before each optimiser step
    GradientClipping(max_norm=1.0),
]

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
results = Trainer(
    model, scenario, strategy,
    optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
    epochs_per_task=2, device=DEVICE,
    callbacks=callbacks, verbose=False,
).fit()

print(f"\nFinal avg_accuracy : {results['avg_accuracy']:.1f}%")
print(f"Final avg_forgetting: {results['avg_forgetting']:.1f}%")

# ---------------------------------------------------------------------------
# Writing your own callback
# ---------------------------------------------------------------------------
from ctk.callbacks import Callback

class PrintTaskTime(Callback):
    """Prints how long each task took to train."""

    import time

    def on_task_begin(self, trainer, task_id, **kwargs):
        self._t0 = self.time.time()

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        elapsed = self.time.time() - self._t0
        print(f"  [PrintTaskTime] Task {task_id} took {elapsed:.1f}s")

# Add it to any trainer the same way as built-ins

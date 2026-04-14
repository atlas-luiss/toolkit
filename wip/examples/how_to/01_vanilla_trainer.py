"""
HOW-TO: Standard (non-CL) Training
=====================================
Two ways to do ordinary supervised training with CTK — no continual learning,
no sequential tasks.  Use these as your baseline before adding any CL strategy.

─────────────────────────────────────────────────────────────
Path A — JointTraining + Trainer  (recommended when you have a CTK benchmark)
  JointTraining concatenates all task data into one dataset and runs a
  single standard training pass.  No strategy needed.

Path B — StandardTrainer  (use when you already have your own DataLoaders)
  Accepts any CLModel plus standard DataLoaders.
  Pass num_classes and the head is set up automatically.
─────────────────────────────────────────────────────────────

Dataset : ImageNet-R  (all 200 classes, trained jointly)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from _setup import make_benchmark, make_model, DEVICE

# ===========================================================================
# Path A — JointTraining + Trainer
# ===========================================================================
from ctk import Trainer
from ctk.scenarios import JointTraining
from ctk.callbacks import MetricLogger

benchmark = make_benchmark(num_tasks=10)
scenario  = JointTraining(benchmark, batch_size=128, num_workers=4,
                          pin_memory=(DEVICE == "cuda"))
model     = make_model()

print("Path A — JointTraining (all 200 classes at once)\n")

results_a = Trainer(
    model,
    scenario,
    optimizer_cls   = torch.optim.AdamW,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    epochs_per_task = 5,
    device          = DEVICE,
    callbacks       = [MetricLogger(log_to=["console"])],
    verbose         = True,
).fit()

print(f"\n  avg_accuracy: {results_a['avg_accuracy']:.1f}%\n")


# ===========================================================================
# Path B — StandardTrainer with your own DataLoaders
# ===========================================================================
from torch.utils.data import ConcatDataset, DataLoader
from ctk.trainers import StandardTrainer

bench2 = make_benchmark(num_tasks=10)
splits = bench2.get_splits()

train_loader = DataLoader(ConcatDataset([s.train_dataset for s in splits]),
                          batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))
val_loader   = DataLoader(ConcatDataset([s.test_dataset for s in splits]),
                          batch_size=128,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))

print("Path B — StandardTrainer with explicit DataLoaders\n")

results_b = StandardTrainer(
    model        = make_model(),
    train_loader = train_loader,
    val_loader   = val_loader,
    num_classes  = bench2.total_classes,   # head is set up automatically
    epochs       = 5,
    lr           = 1e-3,
    weight_decay = 1e-4,
    verbose      = True,
).fit()

print(f"\n  val_accuracy: {results_b['val_accuracy'][-1]:.1f}%")

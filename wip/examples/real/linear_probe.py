"""
Linear Probe  ·  ImageNet-R
============================
The simplest way to evaluate a pretrained backbone: freeze all weights and
train only a single linear layer on top of the extracted features.

A strong linear probe means the backbone already organises the feature space
well — classes are linearly separable without any task-specific adaptation.
This is the standard way to benchmark self-supervised and pretrained models.

All 200 ImageNet-R classes are seen at once (not sequentially).
This is a standard supervised classification experiment, not a CL experiment.
"""

import time
import torch
from torch.utils.data import DataLoader
from _setup import IMAGENET_R_ROOT, DEVICE

from ctk.benchmarks  import ImageNetR
from ctk.models      import DINOModel
from ctk.models.base import IncrementalClassifier
from ctk.trainers    import VanillaTrainer

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 256   # larger batch is fine — no gradient flows through the backbone
LR         = 1e-3

# ── Data ──────────────────────────────────────────────────────────────────────
# Load all 200 classes at once by using num_tasks=1.
# get_splits() returns TaskSplit objects with .train_dataset / .test_dataset.
bench  = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=1, seed=42)
splits = bench.get_splits()
train_ds, test_ds = splits[0].train_dataset, splits[0].test_dataset

print(f"Data       →  ImageNet-R  |  {len(train_ds):,} train  |  {len(test_ds):,} test")
print(f"           →  {bench.total_classes} classes  (all seen simultaneously)")

# ── Model ─────────────────────────────────────────────────────────────────────
# freeze_backbone=True: only the linear head trains.
# The backbone is a fixed feature extractor — no gradient flows through it.
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
model.set_head(IncrementalClassifier(model.embed_dim))
model.grow_head(bench.total_classes)   # single linear head over all 200 classes

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model      →  DINOv2-B/14 (frozen)  +  Linear head")
print(f"Trainable  →  {n_trainable:,} params  ({model.embed_dim} × {bench.total_classes} + bias)\n")

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))

# ── Train ─────────────────────────────────────────────────────────────────────
t0      = time.time()
results = VanillaTrainer(
    model, train_loader, val_loader=test_loader,
    epochs       = EPOCHS,
    lr           = LR,
    weight_decay = 0.0,   # no regularisation for a linear probe
    device       = DEVICE,
    verbose      = True,
).fit()

# ── Results ───────────────────────────────────────────────────────────────────
best_val = max(results["val_accuracy"], default=0.0)
print(f"\n  Best val accuracy : {best_val:.1f}%")
print(f"  Wall time         : {(time.time() - t0) / 60:.1f} min")
print()
print("  This is your DINOv2 feature quality baseline on ImageNet-R.")
print("  Run finetune.py to see how much full fine-tuning helps.")

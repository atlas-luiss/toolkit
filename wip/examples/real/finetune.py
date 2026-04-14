"""
Fine-tuning Comparison  ·  ImageNet-R
=======================================
Compares three training regimes on the same data:

  1. Linear probe  — backbone fully frozen, only the head trains
  2. Head + LoRA   — backbone mostly frozen, small LoRA adapters trained
  3. Full fine-tune — all backbone weights unlocked and trained

Running all three shows the accuracy vs compute/memory trade-off of each
approach on ImageNet-R with DINOv2. Not a CL experiment — all 200 classes
are trained on simultaneously.
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
EPOCHS     = 5
BATCH_SIZE = 64
LR         = 1e-4

# ── Data ──────────────────────────────────────────────────────────────────────
bench  = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=1, seed=42)
splits = bench.get_splits()
train_ds, test_ds = splits[0].train_dataset, splits[0].test_dataset
NUM_CLASSES = bench.total_classes

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))

print(f"Data  →  ImageNet-R  |  {len(train_ds):,} train  |  {len(test_ds):,} test  |  {NUM_CLASSES} classes\n")


def run(label, freeze_backbone, use_lora=False):
    model = DINOModel("dinov2_vitb14", freeze_backbone=freeze_backbone,
                      device=torch.device(DEVICE))
    model.set_head(IncrementalClassifier(model.embed_dim))
    model.grow_head(NUM_CLASSES)

    if use_lora:
        model.add_lora(task_id=0, rank=8, lora_alpha=16)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"── {label}")
    print(f"   Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.1f}%)")

    t0      = time.time()
    results = VanillaTrainer(
        model, train_loader, val_loader=test_loader,
        epochs       = EPOCHS,
        lr           = LR,
        weight_decay = 1e-4,
        device       = DEVICE,
        verbose      = False,   # suppress per-epoch logs; we print the summary below
    ).fit()
    elapsed = time.time() - t0

    best_val = max(results["val_accuracy"], default=0.0)
    train_curve = "  ".join(f"{a:.1f}%" for a in results["val_accuracy"])
    print(f"   Val accuracy by epoch: {train_curve}")
    print(f"   Best val: {best_val:.1f}%  |  Time: {elapsed/60:.1f} min\n")
    return best_val


results = {}
results["Linear probe"]   = run("Linear probe   (frozen backbone)",  freeze_backbone=True)
results["LoRA"]           = run("LoRA           (frozen + adapters)", freeze_backbone=True,  use_lora=True)
results["Full fine-tune"] = run("Full fine-tune (all weights)",       freeze_backbone=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("─" * 50)
print("  Summary")
for name, acc in results.items():
    print(f"  {name:<20}  {acc:.1f}%")
print("─" * 50)

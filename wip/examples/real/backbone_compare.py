"""
Backbone Comparison  ·  ImageNet-R
=====================================
Runs the same linear probe experiment with three different pretrained backbones
and compares their feature quality on ImageNet-R.

  DINOv2-B/14  — self-supervised ViT, trained on ImageNet-21k with DINO
  DINOv2-S/14  — smaller variant, faster but less capacity
  CLIP ViT-B/32 — vision-language model, trained on 400M image-text pairs

All backbones are frozen. Only a linear head is trained.
This tells you which pretrained features transfer best to this domain.
"""

import time
import torch
from torch.utils.data import DataLoader
from _setup import IMAGENET_R_ROOT, MODEL_CACHE, DEVICE

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel, CLIPModel
from ctk.models.base import IncrementalClassifier
from ctk.trainers   import VanillaTrainer

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3

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


def run_linear_probe(label, model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    # Attach an IncrementalClassifier head then grow it to NUM_CLASSES
    model.set_head(IncrementalClassifier(model.embed_dim))
    model.grow_head(NUM_CLASSES)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"── {label}")
    print(f"   embed_dim={model.embed_dim}  |  "
          f"trainable={n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.1f}%)")

    t0      = time.time()
    results = VanillaTrainer(
        model, train_loader, val_loader=test_loader,
        epochs       = EPOCHS,
        lr           = LR,
        weight_decay = 0.0,
        device       = DEVICE,
        verbose      = False,
    ).fit()
    elapsed = time.time() - t0

    best_val = max(results["val_accuracy"], default=0.0)
    print(f"   Best val: {best_val:.1f}%  |  Time: {elapsed/60:.1f} min\n")
    return best_val


# ── Run each backbone ─────────────────────────────────────────────────────────
scores = {}

scores["DINOv2-B/14"] = run_linear_probe(
    "DINOv2-B/14  (self-supervised, ViT-Base)",
    DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE)),
)

scores["DINOv2-S/14"] = run_linear_probe(
    "DINOv2-S/14  (self-supervised, ViT-Small)",
    DINOModel("dinov2_vits14", freeze_backbone=True, device=torch.device(DEVICE)),
)

scores["CLIP ViT-B/32"] = run_linear_probe(
    "CLIP ViT-B/32  (language-supervised)",
    CLIPModel("ViT-B-32", pretrained="openai", freeze_backbone=True,
              device=torch.device(DEVICE)),
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("─" * 50)
print("  Backbone comparison  (linear probe, 10 epochs)")
for name, acc in scores.items():
    bar = "█" * int(acc / 2)
    print(f"  {name:<18}  {acc:5.1f}%  {bar}")
print("─" * 50)

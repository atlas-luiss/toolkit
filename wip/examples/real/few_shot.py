"""
Few-Shot Evaluation  ·  ImageNet-R
=====================================
Tests how well a frozen DINOv2 backbone generalises to new classes with very
limited labelled data: 1, 5, and 20 shots per class.

Uses a nearest-centroid classifier (also called Nearest Mean of Exemplars):
  1. Compute the mean feature vector for each class from the K support samples
  2. Assign each test image to the class with the closest centroid

No training is needed — this is pure feature-space geometry. Strong few-shot
performance means the backbone transfers well to novel distributions.
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from _setup import IMAGENET_R_ROOT, DEVICE

from ctk.benchmarks import ImageNetR
from ctk.models     import DINOModel

# ── Config ────────────────────────────────────────────────────────────────────
K_SHOTS    = [1, 5, 20]   # shots per class to evaluate
BATCH_SIZE = 256

# ── Data ──────────────────────────────────────────────────────────────────────
bench  = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=1, seed=42)
splits = bench.get_splits()
train_ds, test_ds = splits[0].train_dataset, splits[0].test_dataset
NUM_CLASSES = bench.total_classes
print(f"Data  →  ImageNet-R  |  {NUM_CLASSES} classes  |  {len(test_ds):,} test samples\n")

# ── Model ─────────────────────────────────────────────────────────────────────
# Backbone fully frozen — we only extract features, never update weights.
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
model.eval()
print(f"Model →  DINOv2-B/14 (frozen)  |  embed_dim={model.embed_dim}\n")


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(dataset):
    """Return (features, labels) tensors for an entire dataset."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                        pin_memory=(DEVICE == "cuda"))
    all_feats, all_labels = [], []
    for x, y in loader:
        feats = model._encode(x.to(DEVICE))   # (B, embed_dim)
        all_feats.append(feats.cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


print("Extracting features …")
t0 = time.time()
train_feats, train_labels = extract_features(train_ds)
test_feats,  test_labels  = extract_features(test_ds)
print(f"  Done in {time.time()-t0:.1f}s  |  "
      f"train={train_feats.shape}  test={test_feats.shape}\n")


# ── Nearest-centroid classifier ───────────────────────────────────────────────

def nearest_centroid_accuracy(train_feats, train_labels, test_feats, test_labels,
                               k_shot, num_classes, seed=0):
    """Sample k examples per class, compute centroids, classify test set."""
    rng = np.random.default_rng(seed)
    support_feats  = []
    support_labels = []

    for cls in range(num_classes):
        idx = (train_labels == cls).nonzero(as_tuple=True)[0].numpy()
        chosen = rng.choice(idx, size=min(k_shot, len(idx)), replace=False)
        support_feats.append(train_feats[chosen])
        support_labels.extend([cls] * len(chosen))

    support_feats  = torch.cat(support_feats)                      # (k*C, D)
    support_labels = torch.tensor(support_labels)

    # Compute per-class centroids
    centroids = torch.stack([
        support_feats[support_labels == cls].mean(0)
        for cls in range(num_classes)
    ])  # (C, D)

    # Classify: nearest centroid in L2 space
    # Normalise for cosine similarity (often better with DINOv2)
    centroids_n  = torch.nn.functional.normalize(centroids,   dim=1)
    test_feats_n = torch.nn.functional.normalize(test_feats,  dim=1)
    sims   = test_feats_n @ centroids_n.T                          # (N, C)
    preds  = sims.argmax(dim=1)
    acc    = (preds == test_labels).float().mean().item() * 100
    return acc


# ── Evaluate ──────────────────────────────────────────────────────────────────
print("─" * 45)
print("  Few-shot results  (nearest-centroid, cosine)")
print("─" * 45)
for k in K_SHOTS:
    acc = nearest_centroid_accuracy(
        train_feats, train_labels, test_feats, test_labels,
        k_shot=k, num_classes=NUM_CLASSES,
    )
    print(f"  {k:>2}-shot  →  {acc:.1f}%")
print("─" * 45)
print()
print("  Compare the 20-shot number with linear_probe.py (full training data).")
print("  A small gap means DINOv2 features are well-structured for this domain.")

"""
Task Generalization  ·  ImageNet-R
=====================================
Trains on a subset of ImageNet-R tasks and evaluates on held-out tasks
the model has never seen. Measures how well the learned representations
transfer across the class distribution.

Setup:
  - Train tasks : 0–6  (140 classes, seen during training)
  - Test tasks  : 7–9  (60 classes, never seen)

This is different from standard CL evaluation: we care about zero-shot
transfer to new classes, not retention of old ones.

Useful for: domain generalization, open-world recognition research.
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from _setup import IMAGENET_R_ROOT, DEVICE

from ctk.benchmarks  import ImageNetR
from ctk.models      import DINOModel
from ctk.models.base import IncrementalClassifier
from ctk.trainers    import VanillaTrainer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_TASKS   = 10
TRAIN_TASKS = list(range(7))   # tasks 0–6: used for training
TEST_TASKS  = list(range(7, NUM_TASKS))  # tasks 7–9: held out entirely
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 1e-4

# ── Data ──────────────────────────────────────────────────────────────────────
bench  = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=NUM_TASKS, seed=42)
splits = bench.get_splits()

train_classes = sum([splits[t].classes for t in TRAIN_TASKS], [])
test_classes  = sum([splits[t].classes for t in TEST_TASKS],  [])

train_ds = ConcatDataset([splits[t].train_dataset for t in TRAIN_TASKS])
test_ds  = ConcatDataset([splits[t].test_dataset  for t in TEST_TASKS])

print(f"Train  →  tasks {TRAIN_TASKS[0]}–{TRAIN_TASKS[-1]}  |  {len(train_classes)} classes  |  {len(train_ds):,} samples")
print(f"Test   →  tasks {TEST_TASKS[0]}–{TEST_TASKS[-1]}   |  {len(test_classes)} classes  |  {len(test_ds):,} samples  (never seen)\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
model.set_head(IncrementalClassifier(model.embed_dim))
model.grow_head(len(train_classes))   # head only covers training classes
print(f"Model  →  DINOv2-B/14 (frozen)  |  head covers {len(train_classes)} train classes\n")

# ── Train ─────────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))

t0      = time.time()
results = VanillaTrainer(
    model, train_loader,
    epochs       = EPOCHS,
    lr           = LR,
    weight_decay = 1e-4,
    device       = DEVICE,
    verbose      = True,
).fit()
print(f"\nTrain accuracy: {results['train_accuracy'][-1]:.1f}%")

# ── Evaluate on held-out tasks ────────────────────────────────────────────────
# The test classes are different from training classes — use nearest-centroid
# so we don't need to resize the head (which was trained on train classes only).

print("\nEvaluating generalisation to held-out tasks via nearest-centroid …")

model.eval()

@torch.no_grad()
def get_features(ds):
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model._encode(x.to(DEVICE)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)

# Centroids from test-task train splits
test_train_ds = ConcatDataset([splits[t].train_dataset for t in TEST_TASKS])
support_feats, support_labels = get_features(test_train_ds)
query_feats,   query_labels   = get_features(test_ds)

# Remap labels to 0-based within test classes
label_map = {c: i for i, c in enumerate(sorted(set(support_labels.tolist())))}
support_labels = torch.tensor([label_map[l.item()] for l in support_labels])
query_labels   = torch.tensor([label_map[l.item()] for l in query_labels])

centroids = torch.stack([
    support_feats[support_labels == i].mean(0)
    for i in range(len(test_classes))
])
centroids_n = F.normalize(centroids, dim=1)
query_n     = F.normalize(query_feats, dim=1)
preds       = (query_n @ centroids_n.T).argmax(dim=1)
gen_acc     = (preds == query_labels).float().mean().item() * 100

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  Train task accuracy  : {results['train_accuracy'][-1]:.1f}%")
print(f"  Held-out task acc    : {gen_acc:.1f}%  (nearest-centroid, cosine)")
print(f"  Wall time            : {(time.time()-t0)/60:.1f} min")
print(f"{'─'*50}")
print()
print("  The gap between train and held-out accuracy reveals how well")
print("  the frozen DINOv2 features transfer across unseen class groups.")

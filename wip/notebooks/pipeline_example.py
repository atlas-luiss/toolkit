"""
pipeline_example.py — Practical end-to-end pipeline with CTK.

Scenario: an image classification system that needs to support new categories
over time without retraining from scratch.

Stages:
  1. Initial deployment  — fine-tune DINOv2 on the first batch of classes
  2. Class expansion     — incrementally add new classes using LoRA (no forgetting)
  3. Inference           — classify new images across all learned classes
  4. Analysis            — compare forgetting, plot accuracy evolution
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from ctk.models          import DINOModel
from ctk.models.base     import IncrementalClassifier
from ctk.benchmarks      import SplitCIFAR100
from ctk.scenarios       import ClassIncremental
from ctk.strategies      import LoRAStrategy
from ctk.trainers        import VanillaTrainer, ContinualTrainer
from ctk.trainers.cl_evaluator import ContinualEvaluator
from ctk.callbacks       import (
    GradientClipping,
    MetricLogger,
    ModelCheckpointing,
    TaskMetricsCallback,
    ProgressiveResultsCallback,
)
from ctk.utils           import check_trainable_params

# =========================================================================
# Config
# =========================================================================
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT   = '/leonardo_work/IscrC_ConFT/eric/data'
OUTPUT_DIR  = Path('outputs/pipeline')
LORA_RANK   = 8
EPOCHS_INIT = 2    # epochs for initial deployment
EPOCHS_CL   = 1    # epochs per new task during expansion
BATCH       = 64
LR          = 1e-3
NUM_CL_TASKS = 5   # number of expansion rounds after initial deployment

os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 65)
print('  CTK Practical Pipeline — Incremental Class Expansion')
print('=' * 65)
print(f'  Device : {DEVICE}')
print(f'  Stages : 1 initial deployment + {NUM_CL_TASKS} expansion rounds')
print()

# =========================================================================
# Shared transforms (DINOv2 expects ImageNet normalisation)
# =========================================================================
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =========================================================================
# STAGE 1 — Initial deployment
# =========================================================================
# The system launches knowing only the first 10 classes of CIFAR-100.
# We fine-tune a frozen DINOv2 with a single LoRA adapter.
# =========================================================================
print('─' * 65)
print('STAGE 1 — Initial deployment (classes 0–9, frozen DINO + LoRA)')
print('─' * 65)

INIT_CLASSES = list(range(10))

full_train = datasets.CIFAR100(root=DATA_ROOT, train=True,  transform=train_tf, download=False)
full_val   = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=val_tf,   download=False)

def class_subset(dataset, classes):
    idx = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, idx)

init_train_loader = DataLoader(class_subset(full_train, INIT_CLASSES),
                               batch_size=BATCH, shuffle=True,
                               num_workers=1, pin_memory=True)
init_val_loader   = DataLoader(class_subset(full_val,   INIT_CLASSES),
                               batch_size=BATCH, shuffle=False,
                               num_workers=1, pin_memory=True)

model = DINOModel(model_name='dinov2_vitb14', pretrained=True,
                  freeze_backbone=True, device=torch.device(DEVICE))
model.add_lora(task_id=0, rank=LORA_RANK, lora_alpha=16.0, lora_dropout=0.1)
model.set_head(IncrementalClassifier(embed_dim=model.embed_dim,
                                      initial_classes=len(INIT_CLASSES)))

stats = check_trainable_params(model, verbose=False)
print(f'Trainable params: {stats["trainable_params"]:,} '
      f'({stats["trainable_percent"]:.2f}%)')
print()

t0 = time.time()
init_results = VanillaTrainer(
    model, init_train_loader, val_loader=init_val_loader,
    epochs=EPOCHS_INIT, lr=LR, weight_decay=1e-4,
    device=DEVICE, verbose=True,
    callbacks=[
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing(str(OUTPUT_DIR / 'stage1'), monitor='accuracy',
                           mode='max', save_best_only=True,
                           save_task_checkpoints=False),
    ],
).fit()

print(f'\nStage 1 complete in {time.time() - t0:.1f}s')
print(f'  Initial val accuracy : {init_results["val_accuracy"][-1]:.2f}%')
print()

# =========================================================================
# STAGE 2 — Class expansion via continual learning
# =========================================================================
# New categories arrive in {NUM_CL_TASKS} rounds of 10 classes each.
# LoRAStrategy adds a fresh adapter per round — no forgetting by design.
# The model from Stage 1 is reused directly.
# =========================================================================
print('─' * 65)
print(f'STAGE 2 — Incremental expansion ({NUM_CL_TASKS} rounds × 10 classes)')
print('─' * 65)

# Build a CL benchmark for the expansion classes (tasks 1..NUM_CL_TASKS)
# num_tasks covers all classes including the initial 10
bench = SplitCIFAR100(root=DATA_ROOT, num_tasks=NUM_CL_TASKS + 1,
                       image_size=224, seed=42, download=False)
scenario = ClassIncremental(bench, batch_size=BATCH, num_workers=1,
                             pin_memory=(DEVICE == 'cuda'))

# LoRAStrategy will add a new adapter for each CL task (tasks 1..N)
# Task 0 already has its adapter from Stage 1
strategy = LoRAStrategy(rank=LORA_RANK, lora_alpha=16.0,
                         lora_dropout=0.1, merge_after_task=False)

metrics_cb = TaskMetricsCallback(
    save_json=str(OUTPUT_DIR / 'cl_metrics.json')
)

t1 = time.time()
cl_results = ContinualTrainer(
    model, scenario, strategy,
    optimizer_cls=torch.optim.AdamW,
    lr=LR, weight_decay=1e-4,
    epochs_per_task=EPOCHS_CL,
    device=DEVICE, verbose=True,
    callbacks=[
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing(str(OUTPUT_DIR / 'stage2'), monitor='accuracy',
                           mode='max', save_best_only=True,
                           save_task_checkpoints=True),
        metrics_cb,
        ProgressiveResultsCallback(),
    ],
).fit()

print(f'\nStage 2 complete in {(time.time() - t1) / 60:.1f} min')
print(f'  Avg accuracy   : {cl_results["avg_accuracy"]:.2f}%')
print(f'  Avg forgetting : {cl_results["avg_forgetting"]:.2f}%')
print()

# =========================================================================
# STAGE 3 — Inference on unseen images
# =========================================================================
# Simulate a production inference call: pick random images from the val set
# and run them through the trained model.
# =========================================================================
print('─' * 65)
print('STAGE 3 — Inference on held-out images')
print('─' * 65)

CIFAR100_CLASSES = full_val.classes   # list of 100 class name strings

model.eval()
sample_loader = DataLoader(full_val, batch_size=16, shuffle=True,
                            num_workers=1)
images, labels = next(iter(sample_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)

# For Class-IL the head covers all learned classes; infer task_id from
# the total number of tasks seen (last task id)
last_task_id = NUM_CL_TASKS   # scenario set up model up to this task

with torch.no_grad():
    logits = model(images, task_id=last_task_id)
    probs  = F.softmax(logits, dim=1)
    preds  = probs.argmax(dim=1)
    confs  = probs.max(dim=1).values

print(f'{"Image":>5}  {"True label":>20}  {"Predicted":>20}  {"Conf":>6}  {"OK"}')
print('─' * 65)
for i in range(min(16, len(labels))):
    true_name = CIFAR100_CLASSES[labels[i].item()]
    pred_name = CIFAR100_CLASSES[preds[i].item()] if preds[i].item() < len(CIFAR100_CLASSES) else str(preds[i].item())
    ok        = '✓' if preds[i] == labels[i] else '✗'
    print(f'{i:>5}  {true_name:>20}  {pred_name:>20}  {confs[i]:.3f}  {ok}')

batch_acc = (preds == labels).float().mean().item() * 100
print(f'\nBatch accuracy: {batch_acc:.1f}%')
print()

# =========================================================================
# STAGE 4 — Analysis
# =========================================================================
print('─' * 65)
print('STAGE 4 — Analysis')
print('─' * 65)

mat = cl_results['accuracy_matrix']
n   = mat.shape[0]

# Accuracy matrix heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

masked = np.where(np.tril(np.ones_like(mat, dtype=bool)), mat, np.nan)
im = axes[0].imshow(masked, vmin=0, vmax=100, cmap='RdYlGn')
plt.colorbar(im, ax=axes[0], label='Accuracy (%)')
for i in range(n):
    for j in range(n):
        if j <= i:
            axes[0].text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center',
                         fontsize=8)
axes[0].set_xticks(range(n))
axes[0].set_xticklabels([f'T{j}' for j in range(n)])
axes[0].set_yticks(range(n))
axes[0].set_yticklabels([f'After T{i}' for i in range(n)])
axes[0].set_title('Accuracy matrix')

# Per-task accuracy evolution
for task_j in range(n):
    x, y = [], []
    for after_i in range(task_j, n):
        x.append(after_i)
        y.append(mat[after_i, task_j])
    axes[1].plot(x, y, marker='o', ms=4, label=f'T{task_j}')
axes[1].set_xticks(range(n))
axes[1].set_xticklabels([f'After T{i}' for i in range(n)])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Per-task accuracy over time')
axes[1].legend(loc='lower left', ncol=2, fontsize=8)

plt.suptitle('CTK Pipeline — DINOv2 + LoRA | Class-IL | CIFAR-100', y=1.01)
plt.tight_layout()
plot_path = OUTPUT_DIR / 'pipeline_analysis.png'
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
print(f'Plot saved to {plot_path}')

# Summary JSON
summary = {
    'stage1_val_accuracy':   init_results['val_accuracy'][-1],
    'cl_avg_accuracy':       cl_results['avg_accuracy'],
    'cl_avg_forgetting':     cl_results['avg_forgetting'],
    'cl_backward_transfer':  cl_results['backward_transfer'],
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print('=' * 65)
print('PIPELINE COMPLETE')
print('=' * 65)
print(f'  Stage 1 val accuracy : {summary["stage1_val_accuracy"]:.2f}%')
print(f'  CL avg accuracy      : {summary["cl_avg_accuracy"]:.2f}%')
print(f'  CL avg forgetting    : {summary["cl_avg_forgetting"]:.2f}%')
print(f'  Outputs saved to     : {OUTPUT_DIR}/')
print('=' * 65)

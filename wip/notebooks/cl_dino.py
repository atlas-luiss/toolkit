"""
cl_dino.py — Class-Incremental continual learning with DINOv2 on Split-CIFAR-100.
Uses ContinualTrainer, NaiveStrategy, 10 tasks x 2 epochs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch

from ctk.models          import DINOModel
from ctk.benchmarks      import SplitCIFAR100
from ctk.scenarios       import ClassIncremental
from ctk.strategies      import NaiveStrategy
from ctk.trainers        import ContinualTrainer
from ctk.callbacks       import (
    ProgressiveResultsCallback,
    GradientClipping,
    WarmupScheduler,
    LRSchedulerCallback,
    MetricLogger,
    ModelCheckpointing,
    TaskMetricsCallback,
)
from ctk.utils           import check_trainable_params

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/leonardo_work/IscrC_ConFT/eric/data'
NUM_TASKS = 10     # 10 tasks × 10 classes each (100 classes total)
EPOCHS    = 2
LR        = 1e-3
BATCH     = 64

os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')

print('=' * 62)
print('  Split-CIFAR-100 | DINOv2-B/14 | Naive | Class-IL')
print('=' * 62)
print(f'  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  '
      f'Epochs/task: {EPOCHS}  |  BS: {BATCH}')
print()

# -------------------------------------------------------------------------
# Components
# -------------------------------------------------------------------------
bench = SplitCIFAR100(
    root=DATA_ROOT,
    num_tasks=NUM_TASKS,
    image_size=224,
    seed=42,
    download=False,
)

model = DINOModel(
    model_name='dinov2_vitb14',
    pretrained=True,
    freeze_backbone=True,
    device=torch.device(DEVICE),
)

scenario = ClassIncremental(
    bench,
    batch_size=BATCH,
    num_workers=1,
    pin_memory=(DEVICE == 'cuda'),
)

strategy = NaiveStrategy()

stats = check_trainable_params(model, verbose=False)
print(f'Embed dim       : {model.embed_dim}')
print(f'Total params    : {stats["all_params"]:,}')
print(f'Trainable params: {stats["trainable_params"]:,} ({stats["trainable_percent"]:.2f}%)')
print()

# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------
t0 = time.time()

results = ContinualTrainer(
    model,
    scenario,
    strategy,
    optimizer_cls=torch.optim.AdamW,
    lr=LR,
    weight_decay=1e-4,
    epochs_per_task=EPOCHS,
    device=DEVICE,
    verbose=True,
    callbacks=[
        WarmupScheduler(warmup_epochs=1, warmup_factor=0.1),
        LRSchedulerCallback(torch.optim.lr_scheduler.CosineAnnealingLR,
                            T_max=EPOCHS, eta_min=1e-5),
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing('checkpoints/cl_dino', monitor='accuracy',
                           mode='max', save_best_only=True,
                           save_task_checkpoints=True),
        TaskMetricsCallback(save_json='checkpoints/cl_dino/metrics.json'),
        ProgressiveResultsCallback(),
    ],
).fit()

elapsed = time.time() - t0

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print()
print('=' * 62)
print('FINAL RESULTS')
print('=' * 62)
print(f'  Avg accuracy   : {results["avg_accuracy"]:.2f}%')
print(f'  Avg forgetting : {results["avg_forgetting"]:.2f}%')
print(f'  BWT            : {results["backward_transfer"]:.2f}%')
print(f'  Wall time      : {elapsed / 60:.1f} min')
print('=' * 62)

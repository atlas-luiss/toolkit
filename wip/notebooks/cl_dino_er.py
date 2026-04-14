"""
cl_dino_er.py — Class-Incremental continual learning with DINOv2 + Experience Replay.
ContinualTrainer, 10 tasks x 1 epoch, CIFAR-100.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch

from ctk.models          import DINOModel
from ctk.benchmarks      import SplitCIFAR100
from ctk.scenarios       import ClassIncremental
from ctk.strategies      import ExperienceReplay
from ctk.trainers        import ContinualTrainer
from ctk.callbacks       import (
    ProgressiveResultsCallback,
    GradientClipping,
    MetricLogger,
    ModelCheckpointing,
    TaskMetricsCallback,
)
from ctk.utils           import check_trainable_params

# -------------------------------------------------------------------------
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/leonardo_work/IscrC_ConFT/eric/data'
NUM_TASKS = 10
EPOCHS    = 1
LR        = 1e-3
BATCH     = 64
os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')

print('=' * 62)
print('  Split-CIFAR-100 | DINOv2-B/14 | ExperienceReplay | Class-IL')
print('=' * 62)
print(f'  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  '
      f'Epochs/task: {EPOCHS}  |  BS: {BATCH}')
print()

# -------------------------------------------------------------------------
bench    = SplitCIFAR100(root=DATA_ROOT, num_tasks=NUM_TASKS, image_size=224,
                          seed=42, download=False)
model    = DINOModel(model_name='dinov2_vitb14', pretrained=True,
                     freeze_backbone=True, device=torch.device(DEVICE))
scenario = ClassIncremental(bench, batch_size=BATCH, num_workers=1,
                             pin_memory=(DEVICE == 'cuda'))
strategy = ExperienceReplay(
    memory_size=500,          # total exemplars across all seen tasks
    replay_batch_ratio=0.5,   # 50% of each batch comes from the replay buffer
)

stats = check_trainable_params(model, verbose=False)
print(f'Embed dim       : {model.embed_dim}')
print(f'Total params    : {stats["all_params"]:,}')
print(f'Trainable params: {stats["trainable_params"]:,} ({stats["trainable_percent"]:.2f}%)')
print()

# -------------------------------------------------------------------------
t0 = time.time()

results = ContinualTrainer(
    model, scenario, strategy,
    optimizer_cls=torch.optim.AdamW,
    lr=LR, weight_decay=1e-4,
    epochs_per_task=EPOCHS,
    device=DEVICE, verbose=True,
    callbacks=[
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing('checkpoints/cl_dino_er', monitor='accuracy',
                           mode='max', save_best_only=True,
                           save_task_checkpoints=True),
        TaskMetricsCallback(save_json='checkpoints/cl_dino_er/metrics.json'),
        ProgressiveResultsCallback(),
    ],
).fit()

elapsed = time.time() - t0

print()
print('=' * 62)
print('FINAL RESULTS')
print('=' * 62)
print(f'  Avg accuracy   : {results["avg_accuracy"]:.2f}%')
print(f'  Avg forgetting : {results["avg_forgetting"]:.2f}%')
print(f'  BWT            : {results["backward_transfer"]:.2f}%')
print(f'  Wall time      : {elapsed / 60:.1f} min')
print('=' * 62)

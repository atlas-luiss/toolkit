"""
basic_dino.py — Standard single-task fine-tuning with DINOv2 on CIFAR-100.
Uses VanillaTrainer (no continual learning constraints).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ctk.models          import DINOModel
from ctk.models.base     import IncrementalClassifier
from ctk.trainers        import VanillaTrainer
from ctk.callbacks       import (
    EarlyStopping,
    GradientClipping,
    WarmupScheduler,
    LRSchedulerCallback,
    MetricLogger,
    ModelCheckpointing,
)
from ctk.utils           import check_trainable_params

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/leonardo_work/IscrC_ConFT/eric/data'
EPOCHS    = 2
LR        = 1e-3
BATCH     = 64

os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')

print('=' * 60)
print('  CIFAR-100 | DINOv2-B/14 | VanillaTrainer | 2 epochs')
print('=' * 60)
print(f'  Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH}')
print()

# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
# DINOv2 uses 224×224 inputs with ImageNet normalisation
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds     = datasets.CIFAR100(root=DATA_ROOT, train=True,  transform=train_tf, download=False)
val_ds       = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=val_tf,   download=False)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=1, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=1, pin_memory=True)

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
model = DINOModel(
    model_name='dinov2_vitb14',
    pretrained=True,
    freeze_backbone=True,
    device=torch.device(DEVICE),
)

# Attach a classification head for 100 classes
model.set_head(IncrementalClassifier(embed_dim=model.embed_dim, initial_classes=100))

stats = check_trainable_params(model, verbose=False)
print(f'Embed dim       : {model.embed_dim}')
print(f'Total params    : {stats["all_params"]:,}')
print(f'Trainable params: {stats["trainable_params"]:,} ({stats["trainable_percent"]:.2f}%)')
print()

# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------
trainer = VanillaTrainer(
    model,
    train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    optimizer_cls=torch.optim.AdamW,
    lr=LR,
    weight_decay=1e-4,
    device=DEVICE,
    verbose=True,
    callbacks=[
        WarmupScheduler(warmup_epochs=1, warmup_factor=0.1),
        LRSchedulerCallback(torch.optim.lr_scheduler.CosineAnnealingLR,
                            T_max=EPOCHS, eta_min=1e-5),
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing('checkpoints/basic_dino', monitor='accuracy',
                           mode='max', save_best_only=True,
                           save_task_checkpoints=False),
    ],
)

results = trainer.fit()

print()
print('=' * 60)
print(f'  Best train accuracy : {max(results["train_accuracy"]):.2f}%')
print(f'  Best val accuracy   : {max(results["val_accuracy"]):.2f}%')
print('=' * 60)

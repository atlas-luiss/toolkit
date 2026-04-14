"""
basic_dino_lora.py — Single-task fine-tuning with DINOv2 + LoRA on CIFAR-100.
VanillaTrainer, 1 epoch. Only the LoRA adapter and head are trained.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ctk.models          import DINOModel
from ctk.models.base     import IncrementalClassifier
from ctk.trainers        import VanillaTrainer
from ctk.callbacks       import GradientClipping, MetricLogger, ModelCheckpointing
from ctk.utils           import check_trainable_params

# -------------------------------------------------------------------------
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/leonardo_work/IscrC_ConFT/eric/data'
EPOCHS    = 1
LR        = 1e-3
BATCH     = 64
LORA_RANK = 8
os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')

print('=' * 60)
print('  CIFAR-100 | DINOv2-B/14 | LoRA | Vanilla | 1 epoch')
print('=' * 60)
print(f'  Device: {DEVICE}  |  LoRA rank: {LORA_RANK}  |  Batch: {BATCH}')
print()

# -------------------------------------------------------------------------
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

train_loader = DataLoader(
    datasets.CIFAR100(root=DATA_ROOT, train=True, download=False,
                      transform=transforms.Compose([
                          transforms.RandomResizedCrop(224),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std),
                      ])),
    batch_size=BATCH, shuffle=True, num_workers=1, pin_memory=True,
)
val_loader = DataLoader(
    datasets.CIFAR100(root=DATA_ROOT, train=False, download=False,
                      transform=transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std),
                      ])),
    batch_size=BATCH, shuffle=False, num_workers=1, pin_memory=True,
)

# -------------------------------------------------------------------------
model = DINOModel(model_name='dinov2_vitb14', pretrained=True,
                  freeze_backbone=True, device=torch.device(DEVICE))

# Attach LoRA adapter (task_id=0 for single-task use)
model.add_lora(task_id=0, rank=LORA_RANK, lora_alpha=16.0, lora_dropout=0.1)

# Attach classification head
model.set_head(IncrementalClassifier(embed_dim=model.embed_dim, initial_classes=100))

stats = check_trainable_params(model, verbose=False)
print(f'Embed dim       : {model.embed_dim}')
print(f'Total params    : {stats["all_params"]:,}')
print(f'Trainable params: {stats["trainable_params"]:,} ({stats["trainable_percent"]:.2f}%)')
print()

# -------------------------------------------------------------------------
results = VanillaTrainer(
    model, train_loader, val_loader=val_loader,
    epochs=EPOCHS, lr=LR, weight_decay=1e-4, device=DEVICE, verbose=True,
    callbacks=[
        GradientClipping(max_norm=1.0),
        MetricLogger(log_to=['console']),
        ModelCheckpointing('checkpoints/basic_dino_lora', monitor='accuracy',
                           mode='max', save_best_only=True, save_task_checkpoints=False),
    ],
).fit()

print()
print('=' * 60)
print(f'  Train accuracy : {results["train_accuracy"][-1]:.2f}%')
print(f'  Val accuracy   : {results["val_accuracy"][-1]:.2f}%')
print('=' * 60)

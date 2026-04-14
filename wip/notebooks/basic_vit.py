import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))





import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = '/leonardo_work/IscrC_ConFT/eric/data'

os.environ.setdefault('HF_HOME', '/leonardo_work/IscrC_ConFT/eric/hf_cache')

print(f'Device : {DEVICE}')





# load model

from ctk.models import ViTModel

# Vision Transformer — loaded via timm
model = ViTModel(
    model_name='vit_base_patch16_224',
    pretrained=True,
    freeze_backbone=True,   # freeze backbone, only train the head
    device=torch.device(DEVICE),
)

print(f'Backbone  : vit_base_patch16_224')
print(f'Embed dim : {model.embed_dim}')



def inspect_model(model):
    print('Model architecture:')
    print(model)

    # Print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')

    # Print the output shape of the backbone
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)  # Batch size of 1, image size 224x224
    with torch.no_grad():
        backbone_output = model.encode(dummy_input)
    print(f'Backbone output shape: {backbone_output.shape}')
    
    


inspect_model(model)







# Extract features from a real dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408],
                         [0.2675, 0.2565, 0.2761]),
])

dataset = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=tf, download=False)
loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

all_features, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        feats  = model.encode(images).cpu()
        all_features.append(feats)
        all_labels.append(labels)

all_features = torch.cat(all_features)   # (N, 768)
all_labels   = torch.cat(all_labels)     # (N,)

print(f'Extracted features: {all_features.shape}')





# Quick t-SNE visualisation of a subset of classes
from sklearn.manifold import TSNE

n_classes_to_show = 10
mask  = all_labels < n_classes_to_show
feats = all_features[mask].numpy()
lbls  = all_labels[mask].numpy()

coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats)

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(n_classes_to_show):
    sel = lbls == c
    ax.scatter(coords[sel, 0], coords[sel, 1], s=8, alpha=0.6, label=str(c))
ax.legend(markerscale=2, bbox_to_anchor=(1, 1))
ax.set_title(f't-SNE of ViT-B/16 features (first {n_classes_to_show} CIFAR-100 classes)')
plt.tight_layout()
plt.savefig('tsne.png', dpi=120, bbox_inches='tight')
print('t-SNE saved to tsne.png')


# -------------------------------------------------------------------------
# Fine-tune with VanillaTrainer
# -------------------------------------------------------------------------
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from ctk.trainers import VanillaTrainer
from ctk.callbacks import (
    EarlyStopping,
    GradientClipping,
    WarmupScheduler,
    LRSchedulerCallback,
    MetricLogger,
    ModelCheckpointing,
)
from ctk.models.base import IncrementalClassifier

# Build train/val loaders
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

train_ds = datasets.CIFAR100(root=DATA_ROOT, train=True,  transform=train_tf, download=False)
val_ds   = datasets.CIFAR100(root=DATA_ROOT, train=False, transform=val_tf,   download=False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Attach a classification head to the frozen ViT
NUM_CLASSES = 100
head = IncrementalClassifier(embed_dim=model.embed_dim, initial_classes=NUM_CLASSES)
model.set_head(head)

print(f'\nFine-tuning ViT-B/16 (frozen backbone) on CIFAR-100 with VanillaTrainer...')

EPOCHS = 10
trainer = VanillaTrainer(
    model,
    train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    optimizer_cls=torch.optim.AdamW,
    lr=1e-3,
    weight_decay=1e-4,
    device=DEVICE,
    verbose=True,
    callbacks=[
        WarmupScheduler(warmup_epochs=1, warmup_factor=0.1),
        LRSchedulerCallback(torch.optim.lr_scheduler.CosineAnnealingLR,
                            T_max=EPOCHS, eta_min=1e-5),
        GradientClipping(max_norm=1.0),
        EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        MetricLogger(log_to=['csv'], csv_path='training_log.csv'),
        ModelCheckpointing('checkpoints/', monitor='accuracy', mode='max',
                           save_best_only=True, save_task_checkpoints=False),
    ],
)

results = trainer.fit()

print(f'\nBest train accuracy : {max(results["train_accuracy"]):.2f}%')
print(f'Best val accuracy   : {max(results["val_accuracy"]):.2f}%')
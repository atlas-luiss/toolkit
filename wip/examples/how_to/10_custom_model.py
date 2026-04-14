"""
HOW-TO: Custom Model
=====================
Wrap any PyTorch model as a CLModel so it works with all CTK scenarios
and strategies 

You must implement:
  _encode(x)              → Tensor of shape (B, embed_dim)
  embed_dim               → int property
  _default_lora_targets() → list of submodule names to apply LoRA to

The base class handles:
  - Head management (IncrementalClassifier / MultiHeadClassifier)
  - LoRA adapter lifecycle (add_lora, freeze_lora, merge_lora)
  - forward(x, task_id) routing
  - Device tracking

This example wraps a pretrained torchvision ResNet-18 and runs it on
ImageNet-R.  The same pattern works for any CNN or transformer backbone.

Dataset : ImageNet-R  (3 tasks)
Model   : Custom ResNet-18 CLModel (pretrained torchvision)
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch
import torch.nn as nn
from typing import List

from ctk.models.base import CLModel
from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import NaiveStrategy

from _setup import make_benchmark, DEVICE


# ---------------------------------------------------------------------------
# Example 1 — wrapping a pretrained torchvision ResNet
# ---------------------------------------------------------------------------

class ResNetCLModel(CLModel):
    """
    torchvision ResNet-18 with its final FC layer removed, used as a feature
    extractor.  Works with any torchvision classification model.

    Input : (B, 3, H, W)  — standard ImageNet-normalised tensors
    Output: (B, 512)       — feature embeddings
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False,
                 device=None):
        super().__init__(device=device)

        import torchvision.models as tvm
        resnet = tvm.resnet18(weights="DEFAULT" if pretrained else None)

        self._embed_dim = resnet.fc.in_features  # 512 for ResNet-18

        # Remove the final classification head — CTK manages heads instead
        resnet.fc = nn.Identity()
        self._backbone = resnet.to(self._device)

        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def _default_lora_targets(self) -> List[str]:
        # Target the last two residual blocks for LoRA
        return ["layer4.0.conv1", "layer4.0.conv2",
                "layer4.1.conv1", "layer4.1.conv2"]


# ---------------------------------------------------------------------------
# Example 2 — wrapping a plain MLP (for flat-feature datasets)
# ---------------------------------------------------------------------------

class MyMLP(CLModel):
    """
    A two-hidden-layer MLP backbone.
    Suitable for flat feature vectors (e.g. embeddings, tabular data).

    Input : (B, input_dim)
    Output: (B, embed_dim)
    """

    def __init__(self, input_dim: int = 512, hidden: int = 256,
                 embed_dim: int = 128, device=None):
        super().__init__(device=device)
        self._embed_dim = embed_dim

        self._backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        ).to(self._device)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def _default_lora_targets(self) -> List[str]:
        return ["0", "2"]   # Linear layer indices inside the Sequential


# ---------------------------------------------------------------------------
# Run with ResNetCLModel on ImageNet-R
# ---------------------------------------------------------------------------
benchmark = make_benchmark(num_tasks=3)
scenario  = ClassIncremental(benchmark, batch_size=64, num_workers=4,
                             pin_memory=(DEVICE == "cuda"))

# Pretrained ResNet-18, backbone frozen (only the CL head is trained)
model = ResNetCLModel(pretrained=True, freeze_backbone=True,
                      device=torch.device(DEVICE))
print(f"Model  →  ResNet-18  |  embed_dim={model.embed_dim}  |  backbone frozen")

results = Trainer(
    model, scenario, NaiveStrategy(),
    optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
    epochs_per_task=1, device=DEVICE, verbose=True,
).fit()

print(f"\nResNetCLModel — avg_accuracy: {results['avg_accuracy']:.1f}%  "
      f"avg_forgetting: {results['avg_forgetting']:.1f}%")

# ---------------------------------------------------------------------------
# Built-in models (require timm / open_clip / HuggingFace)
# ---------------------------------------------------------------------------
# from ctk.models import ViTModel, CLIPModel, DINOModel, LLMModel
#
# ViTModel  — Vision Transformer via timm
#   model = ViTModel("vit_base_patch16_224", pretrained=True, freeze_backbone=True)
#
# CLIPModel — CLIP vision encoder via open_clip
#   model = CLIPModel("ViT-B-32", pretrained="openai")
#
# DINOModel — DINOv2 self-supervised encoder via torch.hub
#   model = DINOModel("dinov2_vitb14", freeze_backbone=True)
#
# LLMModel  — HuggingFace causal/seq2seq LM
#   model = LLMModel("meta-llama/Llama-2-7b-hf")

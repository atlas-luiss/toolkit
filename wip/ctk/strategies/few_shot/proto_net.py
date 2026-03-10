"""Prototypical Networks for few-shot / continual learning."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctk.strategies.base import Strategy


class ProtoNet(Strategy):
    """Nearest-prototype classifier; prototypes grow with each task (no forgetting by construction)."""

    def __init__(
        self,
        distance:          str  = "cosine",
        finetune_backbone: bool = False,
    ):
        if distance not in ("cosine", "l2"):
            raise ValueError("distance must be 'cosine' or 'l2'")
        self.distance          = distance
        self.finetune_backbone = finetune_backbone

        self._prototypes: Dict[int, torch.Tensor] = {}
        self._class_order: List[int] = []

    # ------------------------------------------------------------------

    def on_task_start(self, task_id: int, train_loader, model) -> None:
        """Build prototypes from the training data for this task."""
        self._build_prototypes(train_loader, model)

    def compute_loss(self, batch, model, task_id: int) -> torch.Tensor:
        """Returns zero loss when finetune_backbone=False (no update needed)."""
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        if not self.finetune_backbone:
            return torch.tensor(0.0, device=model._device, requires_grad=True)

        features = F.normalize(model.encode(x), dim=1) \
            if self.distance == "cosine" else model.encode(x)
        logits = self._proto_logits(features, model._device)
        return F.cross_entropy(logits, y)

    def on_task_end(self, task_id: int, train_loader, model) -> None:
        """Refresh prototypes after fine-tuning (only if backbone was updated)."""
        if self.finetune_backbone:
            self._build_prototypes(train_loader, model)

    def evaluate_task(self, test_loader, model, task_id: int) -> dict:
        """Nearest-prototype classification over all seen classes."""
        device = model._device
        model.eval()

        correct = total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                features = (F.normalize(model.encode(x), dim=1)
                            if self.distance == "cosine"
                            else model.encode(x))

                logits = self._proto_logits(features, device)

                if logits.size(1) > 0:
                    total_loss += F.cross_entropy(logits, y).item()
                    preds       = logits.argmax(1)
                    correct    += (preds == y).sum().item()
                total += y.size(0)

        model.train()
        return {
            "accuracy": 100.0 * correct / max(total, 1),
            "loss":     total_loss / max(len(test_loader), 1),
        }

    # ------------------------------------------------------------------

    def _build_prototypes(self, train_loader, model) -> None:
        """Compute mean feature embedding per class and update self._prototypes."""
        device = model._device
        model.eval()

        sums:   Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(model.embed_dim, device=device)
        )
        counts: Dict[int, int] = defaultdict(int)

        with torch.no_grad():
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                feats = model.encode(x)
                for feat, label in zip(feats, y):
                    cls = int(label.item())
                    sums[cls]   = sums[cls] + feat
                    counts[cls] += 1

        for cls, total in counts.items():
            proto = sums[cls] / total
            if self.distance == "cosine":
                proto = F.normalize(proto, dim=0)
            self._prototypes[cls] = proto.detach()
            if cls not in self._class_order:
                self._class_order.append(cls)

        model.train()

    # ------------------------------------------------------------------

    def _proto_logits(
        self,
        features: torch.Tensor,
        device:   torch.device,
    ) -> torch.Tensor:
        """Return (B, n_classes) similarity/distance logits against stored prototypes."""
        if not self._prototypes:
            return features.new_zeros(features.size(0), 0)

        proto_matrix = torch.stack(
            [self._prototypes[c] for c in self._class_order]
        ).to(device)   # (n_classes, D)

        if self.distance == "cosine":
            f_norm = F.normalize(features, dim=1)
            return f_norm @ proto_matrix.T
        else:
            dists = torch.cdist(features, proto_matrix)
            return -dists

    # ------------------------------------------------------------------

    @property
    def num_prototypes(self) -> int:
        return len(self._prototypes)

    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        """Return a copy of the current prototype memory."""
        return {k: v.clone() for k, v in self._prototypes.items()}

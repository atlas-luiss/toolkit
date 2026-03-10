"""iCaRL — Rebuffi et al., CVPR 2017."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional, List, Dict

from ctk.strategies.base import Strategy
from ctk.strategies.replay.buffer import HerdingBuffer


class iCaRL(Strategy):
    """iCaRL: herding exemplar selection + NMC classifier + BCE with KD loss."""

    def __init__(
        self,
        exemplars_per_class: int   = 20,
        temperature:         float = 2.0,
        kd_weight:           float = 1.0,
    ):
        self.exemplars_per_class = exemplars_per_class
        self.temperature         = temperature
        self.kd_weight           = kd_weight

        self.herding_buf: Optional[HerdingBuffer]  = None
        self._old_model:  Optional[nn.Module]      = None
        self._seen_classes: List[int]              = []

    def on_task_start(self, task_id, train_loader, model):
        """Snapshot the current model before training begins (for KD)."""
        if self.herding_buf is None:
            self.herding_buf = HerdingBuffer(self.exemplars_per_class, model._device)

        if task_id > 0:
            class _Snapshot(nn.Module):
                def __init__(self, backbone, head):
                    super().__init__()
                    self.backbone = backbone
                    self.head     = head
                def forward(self, x):
                    return self.head(self.backbone.encode(x))

            self._old_model = _Snapshot(
                deepcopy(model),
                deepcopy(model.get_head()),
            ).eval().to(model._device)

            for p in self._old_model.parameters():
                p.requires_grad = False

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        logits = model(x, task_id)

        y_oh = torch.zeros_like(logits)
        y_oh.scatter_(1, y.unsqueeze(1), 1.0)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y_oh)

        kd_loss = self._kd_loss(logits, x, model._device)

        return bce_loss + self.kd_weight * kd_loss

    def _kd_loss(self, logits_new: torch.Tensor, x: torch.Tensor, device) -> torch.Tensor:
        """KL divergence between current and old model's tempered sigmoid outputs."""
        if self._old_model is None:
            return torch.tensor(0.0, device=device)

        with torch.no_grad():
            old_logits = self._old_model(x)
            old_probs  = torch.sigmoid(old_logits / self.temperature)

        if self._seen_classes:
            idx = torch.tensor(self._seen_classes, device=device)
            return F.binary_cross_entropy_with_logits(
                logits_new[:, idx] / self.temperature,
                old_probs[:, idx],
            )
        return torch.tensor(0.0, device=device)

    def on_task_end(self, task_id, train_loader, model):
        """Build herding exemplars for the classes seen in this task."""
        device = model._device
        model_ref = model
        model_ref.eval()
        all_features: List[torch.Tensor] = []
        all_inputs:   List[torch.Tensor] = []
        all_labels:   List[torch.Tensor] = []

        with torch.no_grad():
            for batch in train_loader:
                x = batch[0].to(device)
                y = batch[1]
                feats = model_ref.encode(x).cpu()
                all_features.append(feats)
                all_inputs.append(batch[0])  # keep CPU copy
                all_labels.append(y)

        features = torch.cat(all_features)
        inputs   = torch.cat(all_inputs)
        labels   = torch.cat(all_labels)

        new_classes = sorted(labels.unique().tolist())
        for cls in new_classes:
            mask     = labels == cls
            cls_feat = features[mask]
            cls_inp  = inputs[mask]
            if cls_feat.size(0) == 0:
                continue
            self.herding_buf.update_class(cls, cls_feat, cls_inp)
            if cls not in self._seen_classes:
                self._seen_classes.append(cls)

        model_ref.train()

    def evaluate_task(
        self,
        test_loader: torch.utils.data.DataLoader,
        model,
        task_id: int,
    ) -> Optional[Dict]:
        """NMC (nearest-mean-of-exemplars) classifier; returns None if no exemplars stored."""
        if self.herding_buf is None or not self.herding_buf.store:
            return None

        device = model._device
        model.eval()

        class_ids  = sorted(self.herding_buf.store.keys())
        means_mat  = torch.stack([
            self.herding_buf.class_mean(cid).to(device)
            for cid in class_ids
        ])  # (K, D)

        correct, total, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                feats = F.normalize(model.encode(x), dim=1)

                sims      = feats @ means_mat.T
                pred_idx  = sims.argmax(1)
                preds     = torch.tensor(
                    [class_ids[i] for i in pred_idx.tolist()],
                    device=device,
                )

                logits = model(x, task_id)
                total_loss += F.cross_entropy(logits, y).item()
                correct    += (preds == y).sum().item()
                total      += y.size(0)

        model.train()
        return {
            'accuracy': 100.0 * correct / max(total, 1),
            'loss':     total_loss / max(len(test_loader), 1),
        }

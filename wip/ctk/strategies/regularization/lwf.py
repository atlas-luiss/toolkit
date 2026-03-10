"""LwF — Learning without Forgetting (Li & Hoiem, TPAMI 2018)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional

from ctk.strategies.base import Strategy


class LwF(Strategy):
    """LwF: CE + KL distillation from a frozen snapshot of the previous model."""

    def __init__(self, lam: float = 1.0, temperature: float = 2.0):
        self.lam         = lam
        self.temperature = temperature
        self._old_model: Optional[nn.Module] = None

    def on_task_start(self, task_id, train_loader, model):
        """Snapshot the current model (backbone + head) before training begins."""
        if task_id == 0:
            self._old_model = None
            return

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
        ce_loss = F.cross_entropy(logits, y)
        kd_loss = self._distillation_loss(logits, x, model._device)

        return ce_loss + self.lam * kd_loss

    def _distillation_loss(
        self,
        logits_new: torch.Tensor,
        x:          torch.Tensor,
        device,
    ) -> torch.Tensor:
        """KL divergence between current and old model's tempered softmax outputs."""
        if self._old_model is None:
            return torch.tensor(0.0, device=device)

        with torch.no_grad():
            old_logits = self._old_model(x)
            old_probs  = F.softmax(old_logits / self.temperature, dim=1)

        new_log_probs = F.log_softmax(logits_new / self.temperature, dim=1)
        kl = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        return (self.temperature ** 2) * kl

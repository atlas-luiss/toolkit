"""MAS — Memory Aware Synapses (Aljundi et al., ECCV 2018)."""

import torch
import torch.nn.functional as F
from typing import Dict

from ctk.strategies.base import Strategy


class MAS(Strategy):
    """MAS: EWC-style penalty with unsupervised importance (gradient of output L2 norm)."""

    def __init__(self, lam: float = 1000.0, importance_samples: int = 200):
        self.lam                = lam
        self.importance_samples = importance_samples

        self._importance:  Dict[str, torch.Tensor] = {}
        self._prev_params: Dict[str, torch.Tensor] = {}

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        loss  = F.cross_entropy(model(x, task_id), y)
        loss  = loss + self._regularization_loss(model)
        return loss

    def _regularization_loss(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=model._device)
        for name, param in model.named_parameters():
            if name in self._importance and name in self._prev_params:
                imp  = self._importance[name].to(model._device)
                prev = self._prev_params[name].to(model._device)
                loss = loss + (imp * (param - prev).pow(2)).sum()
        return self.lam * loss

    def on_task_end(self, task_id, train_loader, model):
        """Compute MAS importance and accumulate; snapshot parameters."""
        new_omega = self._compute_importance(train_loader, model)

        for name, w in new_omega.items():
            if name in self._importance:
                self._importance[name] = self._importance[name] + w
            else:
                self._importance[name] = w.clone()

        self._prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    def _compute_importance(
        self,
        train_loader: torch.utils.data.DataLoader,
        model,
    ) -> Dict[str, torch.Tensor]:
        """Estimate MAS importance: |d ||f(x)||^2 / d theta| (label-free)."""
        device = model._device
        model.eval()

        omega: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device)
            model.zero_grad()

            out  = model(x, None)
            loss = out.norm(dim=1).pow(2).mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    omega[name] += param.grad.detach().abs()

            n_batches += 1
            if n_batches * x.size(0) >= self.importance_samples:
                break

        for name in omega:
            omega[name] /= max(n_batches, 1)

        model.train()
        return omega

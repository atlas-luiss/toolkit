"""EWC (Kirkpatrick et al., PNAS 2017) and EWC Online (Schwarz et al., ICML 2018)."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from ctk.strategies.base import Strategy


class EWC(Strategy):
    """EWC: quadratic penalty on parameter drift weighted by accumulated Fisher diagonal."""

    def __init__(self, lam: float = 1000.0, fisher_samples: int = 200):
        self.lam             = lam
        self.fisher_samples  = fisher_samples

        self._importance:    Dict[str, torch.Tensor] = {}
        self._prev_params:   Dict[str, torch.Tensor] = {}

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        loss  = F.cross_entropy(model(x, task_id), y)
        loss  = loss + self._regularization_loss(model)
        return loss

    def _regularization_loss(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=model._device)
        for name, param in model.named_parameters():
            if name in self._importance and name in self._prev_params:
                imp   = self._importance[name].to(model._device)
                prev  = self._prev_params[name].to(model._device)
                loss  = loss + (imp * (param - prev).pow(2)).sum()
        return (self.lam / 2.0) * loss

    def on_task_end(self, task_id, train_loader, model):
        """Compute Fisher diagonal and accumulate; snapshot parameters."""
        new_fisher = self._compute_fisher(train_loader, model, task_id)

        for name, f in new_fisher.items():
            if name in self._importance:
                self._importance[name] = self._importance[name] + f
            else:
                self._importance[name] = f.clone()

        self._prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    def _compute_fisher(
        self,
        train_loader: torch.utils.data.DataLoader,
        model,
        task_id=None,
    ) -> Dict[str, torch.Tensor]:
        """Empirical Fisher diagonal via squared gradients of log-likelihood."""
        device = model._device
        model.eval()

        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        n_batches = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            model.zero_grad()

            log_prob  = F.log_softmax(model(x, task_id), dim=1)
            sampled_y = torch.multinomial(log_prob.exp().detach(), 1).squeeze(1)
            F.nll_loss(log_prob, sampled_y).backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

            n_batches += 1
            if n_batches * x.size(0) >= self.fisher_samples:
                break

        for name in fisher:
            fisher[name] /= max(n_batches, 1)

        model.train()
        return fisher


class EWCOnline(Strategy):
    """EWC Online: same penalty as EWC but uses a single running Fisher that decays by gamma each task."""

    def __init__(
        self,
        lam:            float = 1000.0,
        gamma:          float = 0.9,
        fisher_samples: int   = 200,
    ):
        self.lam            = lam
        self.gamma          = gamma
        self.fisher_samples = fisher_samples

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
        return (self.lam / 2.0) * loss

    def on_task_end(self, task_id, train_loader, model):
        """Update running Fisher with decay; snapshot parameters."""
        new_fisher = self._compute_fisher(train_loader, model, task_id)

        for name, f in new_fisher.items():
            if name in self._importance:
                self._importance[name] = self.gamma * self._importance[name] + f
            else:
                self._importance[name] = f.clone()

        self._prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    def _compute_fisher(
        self,
        train_loader: torch.utils.data.DataLoader,
        model,
        task_id=None,
    ) -> Dict[str, torch.Tensor]:
        """Empirical Fisher diagonal."""
        device = model._device
        model.eval()

        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        n_batches = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            model.zero_grad()

            log_prob  = F.log_softmax(model(x, task_id), dim=1)
            sampled_y = torch.multinomial(log_prob.exp().detach(), 1).squeeze(1)
            F.nll_loss(log_prob, sampled_y).backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

            n_batches += 1
            if n_batches * x.size(0) >= self.fisher_samples:
                break

        for name in fisher:
            fisher[name] /= max(n_batches, 1)

        model.train()
        return fisher

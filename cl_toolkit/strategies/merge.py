"""MergeAfterTask: fine-tune on each task, then merge all task vectors."""

from __future__ import annotations

from typing import Dict

import torch

from cl_toolkit import merging
from cl_toolkit.strategies.base import Strategy


class MergeAfterTask(Strategy):
    """Wrap a strategy; after each task, the model used for evaluation is

        theta_base + merge(tau_1, ..., tau_t)

    where tau_i = (weights after task i) - theta_base over the backbone
    parameters that are trainable (the head is kept as trained).

    ``restart``: where task t+1 starts training from —
      ``"finetuned"`` the previous fine-tuned weights (MagMax-style sequential FT),
      ``"merged"`` the merged model, or ``"base"`` the pre-trained weights (independent FT).

    ``method`` is any name in :data:`cl_toolkit.merging.METHODS` (magmax, ties, ta, dare,
    breadcrumbs, consensus, tsv, iso_c, iso_cts, average); ``merge_kw`` go to that method.
    The default restart ("finetuned") is MagMax's sequential fine-tuning, used for every
    method here so they differ only in the merge rule.

    Example (MagMax)::

        MergeAfterTask(NaiveStrategy(), method="magmax")
    """

    def __init__(self, strategy: Strategy, method: str = "magmax", restart: str = "finetuned", **merge_kw):
        if restart not in ("finetuned", "merged", "base"):
            raise ValueError("restart must be 'finetuned', 'merged' or 'base'")
        self.strategy, self.method, self.restart, self.merge_kw = strategy, method, restart, merge_kw
        self.scenarios, self.train_mask, self.eval_mask = strategy.scenarios, strategy.train_mask, strategy.eval_mask
        self._base: Dict[str, torch.Tensor] = {}
        self._last_ft: Dict[str, torch.Tensor] = {}
        self._tvs = []

    def bind(self, trainer):
        super().bind(trainer)
        self.strategy.bind(trainer)

    def _params(self, model):
        return {n: p for n, p in model._backbone.named_parameters() if p.requires_grad}

    def on_task_start(self, task_id, train_loader, model):
        self.strategy.task = self.task
        self.strategy.on_task_start(task_id, train_loader, model)
        params = self._params(model)
        if not params:
            raise ValueError(
                f"{self!r}: no trainable backbone parameters, so there is nothing to merge and this "
                "would silently be plain head training. Use freeze_backbone=False (CLI: --finetune-backbone) "
                "or wrap a strategy that trains backbone parameters (e.g. LoRAAdapter).")
        for n, p in params.items():
            self._base.setdefault(n, p.detach().cpu().clone())
        src = {"finetuned": self._last_ft, "base": self._base, "merged": {}}[self.restart]
        with torch.no_grad():
            for n, p in params.items():
                if n in src:
                    p.copy_(src[n])

    def wants_training(self, task_id):
        return self.strategy.wants_training(task_id)

    def compute_loss(self, batch, model, task_id):
        return self.strategy.compute_loss(batch, model, task_id)

    def before_update(self, model, optimizer):
        self.strategy.before_update(model, optimizer)

    def on_task_end(self, task_id, train_loader, model):
        self.strategy.on_task_end(task_id, train_loader, model)
        params = self._params(model) or {n: p for n, p in model._backbone.named_parameters() if n in self._base}
        ft = {n: p.detach().cpu().clone() for n, p in params.items() if n in self._base}
        self._last_ft = ft
        self._tvs.append(merging.task_vector(ft, {n: self._base[n] for n in ft}))
        # parameter order, not set order: iteration order must not depend on string hashing
        common = [k for k in self._tvs[0] if all(k in tv for tv in self._tvs[1:])]
        tvs = [{k: tv[k] for k in common} for tv in self._tvs]
        merged = merging.apply({k: self._base[k] for k in common},
                               merging.merge(tvs, self.method, **self.merge_kw))
        with torch.no_grad():
            for n, p in model._backbone.named_parameters():
                if n in merged:
                    p.copy_(merged[n])

    def before_eval(self, model, task_id):
        self.strategy.before_eval(model, task_id)

    def evaluate_task(self, test_loader, model, task_id):
        return self.strategy.evaluate_task(test_loader, model, task_id)

    def extra_params(self):
        return self.strategy.extra_params()

    def __repr__(self):
        return f"Merge({self.method}, restart={self.restart}) + {self.strategy!r}"


class ModelAveraging(Strategy):
    """CoMA / CoFiMA (Marouf et al., "Weighted Ensemble Models Are Strong Continual Learners",
    ECCV 2024): continual model averaging of the trainable backbone parameters.

    Task t trains from the previous averaged model theta*_{t-1}; afterwards

      CoMA   : theta*_t = lam * theta_t + (1 - lam) * theta*_{t-1}
      CoFiMA : theta*_t = (lam F_t theta_t + (1-lam) F*_{t-1} theta*_{t-1}) / (lam F_t + (1-lam) F*_{t-1})

    with diagonal Fishers F_t (theta_t on task t's data) and F*_{t-1} (theta*_{t-1} on task t-1's
    data, computed at the end of task t-1, while that data is available). Where both Fishers
    are zero the CoMA weights are used. Defaults: lam = 0.4 (paper, supervised pre-training).
    The paper builds on SLCA (small backbone lr + classifier alignment); combine with
    ``classifier="slca"`` and ``freeze_backbone=False`` for that setting. The head is not averaged.
    """

    def __init__(self, strategy: Strategy, lam: float = 0.4, fisher: bool = False, fisher_samples: int = 512):
        if not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        self.strategy, self.lam, self.fisher, self.fisher_samples = strategy, lam, fisher, fisher_samples
        self.scenarios, self.train_mask, self.eval_mask = strategy.scenarios, strategy.train_mask, strategy.eval_mask
        self._avg: Dict[str, torch.Tensor] = {}
        self._F_avg: Dict[str, torch.Tensor] = {}

    def bind(self, trainer):
        super().bind(trainer)
        self.strategy.bind(trainer)

    def _params(self, model):
        return {n: p for n, p in model._backbone.named_parameters() if p.requires_grad}

    def on_task_start(self, task_id, train_loader, model):
        self.strategy.task = self.task
        self.strategy.on_task_start(task_id, train_loader, model)
        if not self._params(model):
            raise ValueError(f"{self!r}: no trainable backbone parameters to average. Use freeze_backbone=False "
                             "(CLI: --finetune-backbone).")

    def wants_training(self, task_id):
        return self.strategy.wants_training(task_id)

    def compute_loss(self, batch, model, task_id):
        return self.strategy.compute_loss(batch, model, task_id)

    def before_update(self, model, optimizer):
        self.strategy.before_update(model, optimizer)

    def _fisher(self, loader, model, task_id):
        from cl_toolkit.strategies.regularization.ewc import EWC
        helper = EWC(fisher_samples=self.fisher_samples)
        helper.trainer, helper.scenario = self.trainer, self.scenario
        F = helper.importance(loader, model, task_id)
        return {n: F[p].detach().clone() for n, p in self._params(model).items() if p in F}

    def on_task_end(self, task_id, train_loader, model):
        self.strategy.on_task_end(task_id, train_loader, model)
        params = self._params(model)
        cur = {n: p.detach().clone() for n, p in params.items()}
        if not self._avg:
            new = cur
        elif not self.fisher:
            new = {n: self.lam * cur[n] + (1 - self.lam) * self._avg[n] for n in cur}
        else:
            F_t, eps = self._fisher(train_loader, model, task_id), 1e-12
            new = {}
            for n in cur:
                a, b = self.lam * F_t[n], (1 - self.lam) * self._F_avg[n]
                w = (a + self.lam * eps) / (a + b + eps)          # both zero -> CoMA weight lam
                new[n] = w * cur[n] + (1 - w) * self._avg[n]
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(new[n])
        self._avg = {n: v.clone() for n, v in new.items()}
        if self.fisher:
            self._F_avg = self._fisher(train_loader, model, task_id)   # F*_t: averaged model on task t

    def before_eval(self, model, task_id):
        self.strategy.before_eval(model, task_id)

    def evaluate_task(self, test_loader, model, task_id):
        return self.strategy.evaluate_task(test_loader, model, task_id)

    def extra_params(self):
        return self.strategy.extra_params()

    def __repr__(self):
        return f"{'CoFiMA' if self.fisher else 'CoMA'}(lam={self.lam}) + {self.strategy!r}"

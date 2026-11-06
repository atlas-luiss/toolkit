"""WithHead: run any strategy, and use a classifier head from :mod:`cl_toolkit.heads`."""

from __future__ import annotations

import torch

from cl_toolkit import batch as B

from cl_toolkit.heads import Head
from cl_toolkit.strategies.base import Strategy


class WithHead(Strategy):
    """Wrap ``strategy`` (which trains the backbone/head as usual) with ``head``.

    After each task the head absorbs the task's training features (test-time
    transform, global labels), then may correct the model's head (SLCA, WA).
    If ``head.replaces_inference``, evaluation scores come from the head.

    Examples::

        WithHead(ExperienceReplay(500), FeCAM())
        WithHead(FirstSessionAdaptation("adaptformer"), RanPAC())   # = RanPAC paper
        WithHead(NaiveStrategy(train_mask="current"), SLCA())
    """

    def __init__(self, strategy: Strategy, head: Head):
        from cl_toolkit.strategies.routed import RoutedExperts
        if isinstance(strategy, RoutedExperts):
            # the head would be fitted on (and score) the plain backbone's features while the
            # experts are trained and never used: refuse instead of silently ignoring them
            raise ValueError(f"WithHead({head.name}) cannot wrap routed experts: the head would score the plain "
                             "backbone's features and the experts would be ignored. Use "
                             f"Routed(..., head={head.name.lower()!r}) (CLI/run: classifier={head.name.lower()!r} with "
                             "experts=/router=), which fits one head per expert on that expert's features.")
        self.strategy, self.head = strategy, head
        self.scenarios = tuple(k for k in strategy.scenarios if k in head.scenarios)
        self.train_mask, self.eval_mask = strategy.train_mask, strategy.eval_mask

    # -- delegation -----------------------------------------------------------
    def bind(self, trainer):
        super().bind(trainer)
        self.strategy.bind(trainer)
        self.head.bind(trainer.model, trainer.scenario)

    def on_task_start(self, task_id, train_loader, model):
        self.strategy.task = self.task
        self.strategy.on_task_start(task_id, train_loader, model)

    def wants_training(self, task_id):
        return self.strategy.wants_training(task_id)

    def compute_loss(self, batch, model, task_id):
        return self.strategy.compute_loss(batch, model, task_id)

    def before_update(self, model, optimizer):
        self.strategy.before_update(model, optimizer)

    def extra_params(self):
        return self.strategy.extra_params()

    def before_eval(self, model, task_id):
        self.strategy.before_eval(model, task_id)

    # -- head -----------------------------------------------------------------
    @torch.no_grad()
    def features(self, task_id, model):
        fs, ys = [], []
        for x, y in self.scenario.feature_loader(task_id, global_labels=True):
            with self.trainer.autocast():
                fs.append(model.encode(B.to_device(x, model._device)).float())
            ys.append(y.to(model._device))
        return torch.cat(fs), torch.cat(ys)

    def on_task_end(self, task_id, train_loader, model):
        self.strategy.on_task_end(task_id, train_loader, model)
        model.eval()
        if self.head.needs_features:
            self.head.update(*self.features(task_id, model), task_id)
        self.head.after_task(task_id, self.trainer)

    @torch.no_grad()
    def evaluate_task(self, test_loader, model, task_id):
        if not self.head.replaces_inference:
            return self.strategy.evaluate_task(test_loader, model, task_id)
        if task_id is not None:                          # Task-IL: restrict to the task's classes
            cls = torch.tensor(self.scenario.get_task(task_id).classes, device=model._device)
        mask = model._class_mask
        correct = total = 0
        for x, y in test_loader:
            s = self.head.logits(model.encode(B.to_device(x, model._device)).float())
            if task_id is not None:
                pred = s[:, cls].argmax(1)                    # local index, matches local labels
            else:
                if mask is not None:
                    s = s.masked_fill(~mask[: s.size(1)], float("-inf"))
                pred = s.argmax(1)
            correct += (pred.cpu() == y).sum().item()
            total += len(y)
        return {"accuracy": 100.0 * correct / max(total, 1)}

    def __repr__(self):
        return f"{self.strategy!r} + {self.head!r}"

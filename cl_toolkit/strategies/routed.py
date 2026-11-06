"""RoutedExperts: one expert per task + a router at test time (see :mod:`cl_toolkit.routing`)."""

from __future__ import annotations

import copy

from typing import Optional

import torch

from cl_toolkit import batch as B

from cl_toolkit.routing import ExpertPool, Router, get_experts, get_router
from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.naive import NaiveStrategy


class RoutedExperts(Strategy):
    """Train expert t on task t only (true task id: allowed during training), freeze it,
    and fit the router on the plain PTM's features of task t.

    Evaluation: Task-IL uses the given task's expert; otherwise the router picks one
    per sample from the plain-PTM query. Every evaluation also reports
    ``routing_accuracy``: how often the chosen expert is the sample's true task
    (computed from labels, for diagnosis only; Class-IL and Task-IL).

    ``train_mask`` defaults to "current": each expert is trained to separate its own
    task's classes, the usual choice for task-specific experts.

    ``within_task`` (Class-IL, default True): once the router has picked a task, predict
    among that task's classes only, as in Expert Gate (each expert has its own task
    model). With a shared head and no calibration, the other tasks' class weights were
    never trained on this expert's features, so letting them compete is unprincipled.
    ``within_task=False`` lets all seen classes compete, which can recover from a wrong
    route. (HiDe-Prompt keeps all classes but recalibrates the shared head with its
    task-adaptive prediction step, which is not implemented here.)

    ``head`` (optional): a statistical :mod:`cl_toolkit.heads` head (``"lda"``, ``"ncm"``,
    ``"fecam"``, ...) that replaces each expert's linear head at test time. After task t a
    fresh copy is fitted on task t's training data *as seen through expert t* (its own
    statistics, e.g. LDA with that task's covariance). At test time the router picks a task,
    the sample goes through that task's expert, and that task's head scores it among that
    task's classes only (statistics of different experts live in different feature spaces,
    so they are never compared). The linear head is still trained, as the experts' loss.
    """

    def __init__(self, experts: ExpertPool, router: Router, strategy: Optional[Strategy] = None,
                 train_mask: str = "current", within_task: bool = True, head=None):
        self.experts, self.router, self.within_task = experts, router, within_task
        self.head = _check_head(head)
        self._heads = []                                # one fitted copy of ``head`` per expert
        self.strategy = strategy or NaiveStrategy()
        self.train_mask, self.eval_mask = train_mask, self.strategy.eval_mask
        self.scenarios = tuple(k for k in self.strategy.scenarios if k in router.scenarios or k == "task")

    def bind(self, trainer):
        super().bind(trainer)
        self.strategy.bind(trainer)
        self.router.bind(trainer.scenario)
        self._heads = []

    def on_task_start(self, task_id, train_loader, model):
        model.freeze_backbone()
        self.experts.add(model, task_id)
        self.experts.use_train(model, task_id)
        self.router.begin_task(task_id, model.embed_dim, model._device)
        self.strategy.task = self.task
        self.strategy.on_task_start(task_id, train_loader, model)

    def wants_training(self, task_id):
        return self.strategy.wants_training(task_id)

    def compute_loss(self, batch, model, task_id):
        out = self.strategy.compute_loss(batch, model, task_id)
        if not self.router.trainable:
            return out
        loss, logits = out if isinstance(out, tuple) else (out, None)
        with torch.no_grad():
            self.experts.use_none(model)
            q = model.encode(B.to_device(batch[0], model._device))
            self.experts.use_train(model, task_id)
        return loss + self.router.aux_loss(q, task_id), logits

    def before_update(self, model, optimizer):
        self.strategy.before_update(model, optimizer)

    def extra_params(self):
        return self.strategy.extra_params() + (self.router.params(self.task.task_id) if self.task else [])

    def on_task_end(self, task_id, train_loader, model):
        self.strategy.on_task_end(task_id, train_loader, model)
        self.experts.freeze(model, task_id)
        model.eval()
        if self.head is not None:
            self._fit_expert_head(task_id, model)
        qs, ys = [], []
        self.experts.use_none(model)
        with torch.no_grad():
            for x, y in self.scenario.feature_loader(task_id, global_labels=True):
                with self.trainer.autocast():
                    qs.append(model.encode(B.to_device(x, model._device)).float())
                ys.append(y.to(model._device))
        self.router.fit(task_id, torch.cat(qs), torch.cat(ys))

    def before_eval(self, model, task_id):
        self.strategy.before_eval(model, task_id)

    @torch.no_grad()
    def _fit_expert_head(self, task_id, model):
        """Head statistics of task ``task_id``'s classes, on features from expert ``task_id``."""
        head = copy.deepcopy(self.head)
        head.bind(model, self.scenario)
        fs, ys = [], []
        for x, y in self.scenario.feature_loader(task_id, global_labels=True):
            x = B.to_device(x, model._device)
            ids = torch.full((B.size(x),), task_id, dtype=torch.long, device=model._device)
            with self.trainer.autocast():
                fs.append(self.experts.encode_with(model, x, ids).float())
            ys.append(y.to(model._device))
        head.update(torch.cat(fs), torch.cat(ys), task_id)
        self._heads.append(head)

    def _head_logits(self, f, ids):
        """(B, C) scores; sample i is scored by the head of its expert ids[i] (-inf elsewhere)."""
        out = None
        for e in ids.unique().tolist():
            m = ids == e
            s = self._heads[int(e)].logits(f[m].float())
            if out is None:
                out = s.new_full((len(f), s.size(1)), float("-inf"))
            out[m] = s
        return out

    @torch.no_grad()
    def evaluate_task(self, test_loader, model, task_id):
        scen = self.scenario
        if scen.kind == "class":
            t_of = torch.full((scen.benchmark.total_classes,), -1, dtype=torch.long)
            for t in scen.tasks:
                t_of[t.classes] = t.task_id
        correct = routed_ok = total = 0
        for x, y in test_loader:
            x = B.to_device(x, model._device)
            if task_id is not None:
                ids = torch.full((B.size(x),), task_id, dtype=torch.long, device=model._device)
            else:
                self.experts.use_none(model)
                ids = self.router.route(model.encode(x)).to(model._device)
            f = self.experts.encode_with(model, x, ids)
            if self.head is not None:
                logits = self._head_logits(f, ids)          # global classes; only the routed task's
                if task_id is not None:                     # Task-IL: local labels
                    logits = logits[:, torch.tensor(scen.get_task(task_id).classes, device=logits.device)]
                elif model._class_mask is not None and scen.kind != "class":
                    logits = logits.masked_fill(~model._class_mask[: logits.size(1)], float("-inf"))
            else:
                logits = model.classify(f, task_id)
            if task_id is None and scen.kind == "class" and self.within_task and self.head is None:
                same = t_of[: logits.size(1)].to(logits.device)[None, :] == ids[:, None]
                logits = logits.masked_fill(~same, float("-inf"))
            correct += (logits.argmax(1).cpu() == y).sum().item()
            total += len(y)
            if task_id is not None:
                routed_ok += len(y)
            elif scen.kind == "class":
                routed_ok += (ids.cpu() == t_of[y]).sum().item()
        out = {"accuracy": 100.0 * correct / max(total, 1)}
        if task_id is not None or scen.kind == "class":
            out["routing_accuracy"] = 100.0 * routed_ok / max(total, 1)
        return out

    def __repr__(self):
        head = f", head=per-expert {self.head!r}" if self.head is not None else ""
        return (f"Routed({self.experts!r}, {self.router!r}, train_mask={self.train_mask}, "
                f"within_task={self.within_task}{head}) + {self.strategy!r}")


def _check_head(head):
    if head is None:
        return None
    from cl_toolkit.heads import Head, get_head
    head = head if isinstance(head, Head) else get_head(head)
    if not head.replaces_inference:
        raise ValueError(f"{head.name} corrects one shared head across tasks; with routed experts prediction is "
                         "already restricted to the routed task, so there is nothing for it to correct. Use a "
                         "statistical head (lda, ncm, fecam, klda, ranpac) as the per-expert head.")
    return head


def Routed(experts="lora", router="hide", strategy: Optional[Strategy] = None, train_mask: str = "current",
           experts_kwargs=None, router_kwargs=None, within_task: bool = True, head=None,
           head_kwargs=None) -> RoutedExperts:
    """String-friendly constructor: ``Routed("prompt", "sprompts")``, ``Routed("lora", "hide", head="lda")``."""
    if isinstance(head, str):
        from cl_toolkit.heads import get_head
        head = get_head(head, **(head_kwargs or {}))
    return RoutedExperts(get_experts(experts, **(experts_kwargs or {})),
                         get_router(router, **(router_kwargs or {})), strategy, train_mask, within_task, head)

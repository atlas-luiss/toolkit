"""Strategy: the only thing you write to try a new CL idea.

Lifecycle per task (called by ContinualTrainer)::

    on_task_start(task_id, train_loader, model)
    for each batch (only if wants_training(task_id)):
        compute_loss(batch, model, task_id) -> loss  or  (loss, logits)
        before_update(model, optimizer)          # after backward, before step
    on_task_end(task_id, train_loader, model)
    for each task seen so far:
        before_eval(model, task_id_or_None)
        evaluate_task(loader, model, task_id_or_None) -> dict or None (= default eval)

``task_id_or_None`` is None unless the scenario is Task-IL — evaluation can
never peek at task identity in Class-IL / Domain-IL.

Useful attributes, set by the trainer before training:
  ``self.trainer``, ``self.scenario``, ``self.task`` (current TaskInfo).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cl_toolkit import batch as B
import torch.nn as nn
import torch.nn.functional as F

from cl_toolkit.scenarios import KINDS


class Strategy:
    #: scenarios this strategy is valid for
    scenarios: Tuple[str, ...] = KINDS
    #: shared-head logit masks (ignored in Task-IL, which has per-task heads):
    #:   train_mask: "seen" = classes seen so far, "current" = current task only, "none" = all outputs
    #:   eval_mask:  "seen" = classes seen so far, "none" = all outputs (incl. never-trained classes)
    train_mask: str = "seen"
    eval_mask: str = "seen"

    trainer = None
    scenario = None
    task = None

    # -- lifecycle --------------------------------------------------------------
    def bind(self, trainer) -> None:
        self.trainer, self.scenario = trainer, trainer.scenario

    def on_task_start(self, task_id: int, train_loader, model) -> None: ...

    def wants_training(self, task_id: int) -> bool:
        return True

    def compute_loss(self, batch, model, task_id: int):
        x, y = self.to_device(batch, model)
        return self.ce(model, x, y, task_id)

    def before_update(self, model, optimizer) -> None: ...

    def on_task_end(self, task_id: int, train_loader, model) -> None: ...

    def before_eval(self, model, task_id: Optional[int]) -> None: ...

    def evaluate_task(self, test_loader, model, task_id: Optional[int]) -> Optional[Dict]:
        return None

    def extra_params(self) -> List[nn.Parameter]:
        return []

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def to_device(batch, model):
        return B.to_device(batch[0], model._device), batch[1].to(model._device, non_blocking=True)

    @staticmethod
    def task_ids(n, task_id: int, device) -> torch.Tensor:
        n = n if isinstance(n, int) else B.size(n)
        return torch.full((n,), task_id, dtype=torch.long, device=device)

    def memory_mask(self, model, task_id: int):
        """Context for forwarding *memory* samples: all classes seen so far are allowed,
        even when ``train_mask="current"`` restricts the incoming batch (ER-ACE style).
        Without this, replayed labels outside the current task would get -inf logits."""
        seen = self.scenario.seen_classes(task_id) if self.scenario.shared_head else None
        return model.class_mask(seen)

    def ce(self, model, x, y, task_id) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy + the model's aux loss (prompt key matching etc.)."""
        logits = model(x, task_id)
        return F.cross_entropy(logits, y) + model.aux_loss, logits

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items()
                         if not k.startswith("_") and isinstance(v, (int, float, str, bool)))
        return f"{self.name}({args})"

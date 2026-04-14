"""
HOW-TO: Custom Strategy
========================
Implement your own continual learning algorithm by subclassing Strategy.

You must implement:
  compute_loss(batch, model, task_id) → torch.Tensor

You can optionally override:
  on_task_start(task_id, train_loader, model)  — called before each task
  on_task_end(task_id, train_loader, model)    — called after each task
  before_update(model, optimizer)              — called after backward(), before step()
  extra_params() → List[Parameter]            — extra optimiser params (e.g. task embeddings)
  evaluate_task(test_loader, model, task_id)   — custom eval; return None to use default

Dataset : ImageNet-R  (3 tasks)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ctk.strategies.base import Strategy
from ctk import Trainer
from ctk.scenarios import ClassIncremental

from _setup import make_benchmark, make_model, DEVICE


# ---------------------------------------------------------------------------
# Example: a simple L2-regularisation strategy
#
# After each task we snapshot the parameters.  On subsequent tasks we add
# a penalty that pulls weights back toward the snapshot, weighted by `lam`.
# (This is a simplified, unweighted version of EWC.)
# ---------------------------------------------------------------------------

class L2Regularisation(Strategy):
    """
    Penalise deviation from parameters at the end of the previous task.
    Simpler than EWC (no Fisher weighting) but easy to understand.
    """

    def __init__(self, lam: float = 10.0):
        self.lam = lam
        self._prev_params: Dict[str, torch.Tensor] = {}

    # Required — compute the training loss for one batch
    def compute_loss(self, batch, model, task_id) -> torch.Tensor:
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        # Standard classification loss
        ce_loss = F.cross_entropy(model(x, task_id), y)

        # L2 penalty toward previous task's parameters
        reg_loss = torch.tensor(0.0, device=model._device)
        for name, param in model.named_parameters():
            if name in self._prev_params:
                prev = self._prev_params[name].to(model._device)
                reg_loss = reg_loss + (param - prev).pow(2).sum()

        return ce_loss + self.lam * reg_loss

    # Optional — snapshot parameters after each task
    def on_task_end(self, task_id, train_loader, model):
        self._prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        print(f"  [L2Regularisation] Snapshotted {len(self._prev_params)} "
              f"param tensors after task {task_id}.")


# ---------------------------------------------------------------------------
# Example: gradient clipping inside the strategy (before_update)
# ---------------------------------------------------------------------------

class ClippedNaive(Strategy):
    """NaiveStrategy + gradient norm clipping; shows before_update() usage."""

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        return F.cross_entropy(model(x, task_id), y)

    def before_update(self, model, optimizer):
        nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


# ---------------------------------------------------------------------------
# Run both
# ---------------------------------------------------------------------------

def run(strategy, label):
    benchmark = make_benchmark(num_tasks=3)
    scenario  = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                                 pin_memory=(DEVICE == "cuda"))
    model     = make_model()

    results = Trainer(
        model, scenario, strategy,
        optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
        epochs_per_task=1, device=DEVICE, verbose=False,
    ).fit()

    print(f"  {label:<22}  "
          f"acc={results['avg_accuracy']:5.1f}%  "
          f"forgetting={results['avg_forgetting']:5.1f}%")


print("\nCustom strategies (3 tasks, 1 epoch each):\n")
run(L2Regularisation(lam=10.0), "L2Regularisation")
run(ClippedNaive(max_norm=1.0), "ClippedNaive")

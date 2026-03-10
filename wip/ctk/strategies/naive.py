"""Naive fine-tuning strategy — plain cross-entropy, no forgetting prevention."""

import torch.nn.functional as F

from ctk.strategies.base import Strategy


class NaiveStrategy(Strategy):
    """Plain cross-entropy fine-tuning. No forgetting prevention."""

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        return F.cross_entropy(model(x, task_id), y)

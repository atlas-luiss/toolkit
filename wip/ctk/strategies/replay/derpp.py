"""Dark Experience Replay++ (DER++) — Buzzega et al., NeurIPS 2020."""

import torch
import torch.nn.functional as F

from ctk.strategies.base import Strategy
from ctk.strategies.replay.buffer import ReservoirBuffer


class DERPP(Strategy):
    """DER++: CE + alpha * MSE(logit distillation) + beta * CE(label replay)."""

    def __init__(self, memory_size: int = 500, alpha: float = 0.5, beta: float = 0.5):
        self.memory_size = memory_size
        self.alpha       = alpha
        self.beta        = beta
        self.buffer: ReservoirBuffer | None = None

    def on_task_start(self, task_id, train_loader, model):
        if self.buffer is None:
            self.buffer = ReservoirBuffer(self.memory_size, model._device)

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        logits_new = model(x, task_id)
        loss = F.cross_entropy(logits_new, y)

        x_m, y_m, _, z_m = self.buffer.sample(len(x))
        if x_m is not None:
            x_m = x_m.to(model._device)
            y_m = y_m.to(model._device)
            logits_m = model(x_m)

            if z_m is not None and self.alpha > 0:
                z_m = z_m.to(model._device)
                loss = loss + self.alpha * F.mse_loss(logits_m, z_m)

            if self.beta > 0:
                loss = loss + self.beta * F.cross_entropy(logits_m, y_m)

        self.buffer.add_batch(
            x.detach(), y.detach(), task_id,
            logits=logits_new.detach(),
        )

        return loss

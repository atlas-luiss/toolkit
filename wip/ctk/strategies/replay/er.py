"""Experience Replay (ER) — Rolnick et al., 2019."""

import torch.nn.functional as F

from ctk.strategies.base import Strategy
from ctk.strategies.replay.buffer import ReservoirBuffer


class ExperienceReplay(Strategy):
    """Experience Replay: store to reservoir buffer, replay alongside current batch."""

    def __init__(self, memory_size: int = 500, replay_batch_ratio: float = 0.5):
        self.memory_size        = memory_size
        self.replay_batch_ratio = replay_batch_ratio
        self.buffer: ReservoirBuffer | None = None

    def on_task_start(self, task_id, train_loader, model):
        if self.buffer is None:
            self.buffer = ReservoirBuffer(self.memory_size, model._device)

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        self.buffer.add_batch(x.detach(), y.detach(), task_id)
        loss = F.cross_entropy(model(x, task_id), y)

        n_replay = int(len(x) * self.replay_batch_ratio)
        x_m, y_m, _, _ = self.buffer.sample(n_replay)
        if x_m is not None:
            x_m = x_m.to(model._device)
            y_m = y_m.to(model._device)
            loss = loss + F.cross_entropy(model(x_m), y_m)

        return loss

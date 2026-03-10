"""GDumb — Prabhu et al., ECCV 2020."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctk.strategies.base import Strategy
from ctk.strategies.replay.buffer import ReservoirBuffer


class GDumb(Strategy):
    """GDumb: fill buffer during stream, then retrain the head entirely on buffer at each task end."""

    def __init__(self, memory_size: int = 500, replay_epochs: int = 10, lr: float = 1e-3):
        self.memory_size   = memory_size
        self.replay_epochs = replay_epochs
        self.lr            = lr
        self.buffer: ReservoirBuffer | None = None

    def on_task_start(self, task_id, train_loader, model):
        if self.buffer is None:
            self.buffer = ReservoirBuffer(self.memory_size, model._device)
        model.freeze_backbone()

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)

        self.buffer.add_batch(x.detach(), y.detach(), task_id)

        x_s, y_s, _, _ = self.buffer.sample(len(x))
        if x_s is not None:
            x_s = x_s.to(model._device)
            y_s = y_s.to(model._device)
            return F.cross_entropy(model(x_s, task_id), y_s)

        return F.cross_entropy(model(x, task_id), y)

    def on_task_end(self, task_id, train_loader, model):
        """Re-initialise the head and retrain for replay_epochs epochs on the buffer."""
        device = model._device

        head = model.get_head()
        if hasattr(head, 'weight'):
            nn.init.xavier_uniform_(head.weight)
        if hasattr(head, 'bias') and head.bias is not None:
            nn.init.zeros_(head.bias)

        buf_loader = torch.utils.data.DataLoader(
            self.buffer.as_tensor_dataset(),
            batch_size=64,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(head.parameters(), lr=self.lr)

        model.freeze_backbone()
        head.train()

        for epoch in range(self.replay_epochs):
            for x_b, y_b in buf_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x_b), y_b)
                loss.backward()
                optimizer.step()

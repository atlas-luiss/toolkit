"""A-GEM — Averaged Gradient Episodic Memory (Chaudhry et al., ICLR 2019)."""

import torch
import torch.nn.functional as F

from ctk.strategies.base import Strategy
from ctk.strategies.replay.buffer import ReservoirBuffer


class AGEM(Strategy):
    """A-GEM: project current-task gradients to not conflict with the memory reference gradient."""

    def __init__(self, memory_size: int = 500, ref_sample_size: int = 256):
        self.memory_size     = memory_size
        self.ref_sample_size = ref_sample_size
        self.buffer: ReservoirBuffer | None = None

    def on_task_start(self, task_id, train_loader, model):
        if self.buffer is None:
            self.buffer = ReservoirBuffer(self.memory_size, model._device)

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        self._task_id = task_id   # stored for use in before_update
        self.buffer.add_batch(x.detach(), y.detach(), task_id)
        return F.cross_entropy(model(x, task_id), y)

    def before_update(self, model, optimizer):
        """Project current gradient if it conflicts with the memory reference gradient."""
        if self.buffer is None or len(self.buffer) == 0:
            return

        task_id = getattr(self, '_task_id', 0)
        device  = model._device
        params  = [p for p in model.parameters() if p.requires_grad]

        saved_grads = [
            p.grad.clone() if p.grad is not None else None
            for p in params
        ]

        x_ref, y_ref, _, _ = self.buffer.sample(self.ref_sample_size)
        if x_ref is None:
            return

        x_ref = x_ref.to(device)
        y_ref = y_ref.to(device)

        optimizer.zero_grad()
        loss_ref = F.cross_entropy(model(x_ref, task_id), y_ref)
        loss_ref.backward()

        g_ref = torch.cat([
            p.grad.view(-1).clone()
            for p in params if p.grad is not None
        ])

        for p, saved in zip(params, saved_grads):
            p.grad = saved

        g_cur = torch.cat([
            p.grad.view(-1)
            for p in params if p.grad is not None
        ])

        dot = torch.dot(g_cur, g_ref)
        if dot < 0:
            g_proj = g_cur - (dot / (g_ref.norm() ** 2 + 1e-8)) * g_ref
            offset = 0
            for p in params:
                if p.grad is not None:
                    numel = p.grad.numel()
                    p.grad.copy_(g_proj[offset: offset + numel].view_as(p.grad))
                    offset += numel

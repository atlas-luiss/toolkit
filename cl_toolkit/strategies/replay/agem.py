"""A-GEM (Chaudhry et al., ICLR 2019)."""

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.replay.buffer import ReservoirBuffer


class AGEM(Strategy):
    """Project the gradient so it does not increase the loss on a memory batch.

    The memory only receives samples at the end of each task, so the reference
    gradient always comes from *previous* tasks (as in the paper).
    """

    def __init__(self, memory_size: int = 500, ref_sample_size: int = 256):
        self.memory_size, self.ref_sample_size = memory_size, ref_sample_size
        self.buffer = ReservoirBuffer(memory_size)
        self._pending = None

    def on_task_start(self, task_id, train_loader, model):
        self._pending = ReservoirBuffer(self.memory_size)   # this task's candidates

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        self._pending.add(x, y, task_id)
        return self.ce(model, x, y, task_id)

    def before_update(self, model, optimizer):
        if not len(self.buffer):
            return
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not params:
            return
        g = torch.cat([p.grad.flatten() for p in params])
        xm, ym, tm, _ = self.buffer.sample(self.ref_sample_size, model._device)
        with self.trainer.autocast(), self.memory_mask(model, self.task.task_id):
            ref_loss = F.cross_entropy(model(xm, tm), ym)
        if not ref_loss.requires_grad:          # memory loss uses no trainable parameter: no conflict
            return
        grads = torch.autograd.grad(ref_loss, params, allow_unused=True)
        # parameters the memory loss does not use (e.g. the current task's head in Task-IL)
        # have zero reference gradient; keep them so g and g_ref stay aligned
        g_ref = torch.cat([(torch.zeros_like(p) if gr is None else gr).flatten() for p, gr in zip(params, grads)])
        dot = g @ g_ref
        if dot < 0:
            g = g - dot / (g_ref @ g_ref + 1e-12) * g_ref
            off = 0
            for p in params:
                p.grad.copy_(g[off: off + p.numel()].view_as(p))
                off += p.numel()

    def on_task_end(self, task_id, train_loader, model):
        x, y, t = self._pending.all()
        # Keep an equal share of memory per task seen so far.
        T = task_id + 1
        quota = self.memory_size // T
        if len(self.buffer):
            bx, by, bt = self.buffer.all()
            keep = torch.cat([torch.nonzero(bt == k).flatten()[:quota] for k in bt.unique()])
            bx, by, bt = B.index(bx, keep), by[keep], bt[keep]
        else:
            bx = by = bt = None
        self.buffer = ReservoirBuffer(self.memory_size)
        if bx is not None:
            self.buffer.add(bx, by, bt)
        perm = torch.randperm(B.size(x))[: self.memory_size - len(self.buffer)]
        self.buffer.add(B.index(x, perm), y[perm], t[perm])

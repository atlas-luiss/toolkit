"""GDumb (Prabhu et al., ECCV 2020)."""

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.replay.buffer import ClassBalancedBuffer


class GDumb(Strategy):
    """Greedy class-balanced memory; after each task, re-initialise the head and
    train the trainable parameters *only* on the memory. Never trains on the stream.

    With a frozen pretrained backbone this is "GDumb-head"; unfreeze the backbone
    to retrain more (note it then keeps drifting across tasks, unlike the paper's
    from-scratch retraining).
    """

    def __init__(self, memory_size: int = 2000, epochs: int = 10, lr: float = 1e-3, batch_size: int = 64):
        self.memory_size, self.epochs, self.lr, self.batch_size = memory_size, epochs, lr, batch_size
        self.buffer = ClassBalancedBuffer(memory_size)

    def wants_training(self, task_id):
        return False

    def on_task_end(self, task_id, train_loader, model):
        for batch in self.scenario.feature_loader(task_id, shuffle=True):
            self.buffer.add(batch[0], batch[1], task_id)
        x, y, t = self.buffer.all()
        model.get_head().reset_parameters()
        params = model.trainable_params()
        opt = torch.optim.AdamW(params, lr=self.lr)
        model.train()
        n = B.size(x)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                xb, yb, tb = B.to_device(B.index(x, b), model._device), y[b].to(model._device), t[b].to(model._device)
                opt.zero_grad(set_to_none=True)
                with self.trainer.autocast(), self.memory_mask(model, task_id):
                    loss = F.cross_entropy(model(xb, tb), yb)
                loss.backward()
                opt.step()
        model.eval()

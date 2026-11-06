"""Write a new CL method: subclass Strategy and override only what you need.

Strategy lifecycle (all optional except nothing — the default is plain CE):
  on_task_start -> [compute_loss -> before_update]* -> on_task_end -> before_eval / evaluate_task
Inside a strategy you can use self.scenario, self.task (current TaskInfo), self.trainer.
"""
import os

from copy import deepcopy
import torch
import torch.nn.functional as F
import cl_toolkit as clt
from cl_toolkit.strategies import Strategy, LoRAAdapter, MergeAfterTask
from cl_toolkit.strategies.replay import ReservoirBuffer


class FeatureDistillReplay(Strategy):
    """Example: ER + feature distillation to the previous model on replayed samples."""

    def __init__(self, memory_size=500, beta=1.0):
        self.buffer, self.beta, self.old = ReservoirBuffer(memory_size), beta, None

    def on_task_start(self, task_id, train_loader, model):
        self.old = deepcopy(model).eval() if task_id > 0 else None

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        loss, logits = self.ce(model, x, y, task_id)          # CE (+ model aux loss)
        xm, ym, tm, _ = self.buffer.sample(len(x), x.device)
        if xm is not None:
            f = model.encode(xm)
            loss = loss + F.cross_entropy(model.classify(f, tm), ym)
            if self.old is not None:
                with torch.no_grad():
                    f_old = self.old.encode(xm)
                loss = loss + self.beta * F.mse_loss(f, f_old)
        self.buffer.add(x, y, task_id)
        return loss, logits                                     # logits -> train accuracy for free


if __name__ == "__main__":
    ROOT = os.environ.get("CL_DATA", "./data")
    common = dict(root=f"{ROOT}/cifar100", num_tasks=10, epochs=1, batch_size=128)

    clt.run(data="cifar100", strategy=FeatureDistillReplay(500), **common)
    # Compose with built-in wrappers:
    clt.run(data="cifar100", strategy=LoRAAdapter(FeatureDistillReplay(500), rank=8), **common)
    clt.run(data="cifar100", strategy=MergeAfterTask(FeatureDistillReplay(500), method="ties"),
            freeze_backbone=False, lr=1e-5, **common)

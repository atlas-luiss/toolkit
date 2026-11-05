"""Experience Replay (Chaudhry et al. 2019 / Rolnick et al. 2019)."""

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.replay.buffer import ReservoirBuffer


class ExperienceReplay(Strategy):
    """Reservoir memory; each step trains on [current batch ; memory batch]."""

    def __init__(self, memory_size: int = 500, replay_batch_ratio: float = 1.0):
        self.memory_size, self.replay_batch_ratio = memory_size, replay_batch_ratio
        self.buffer = ReservoirBuffer(memory_size)

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        n = B.size(x)
        t = self.task_ids(n, task_id, y.device)
        xm, ym, tm, _ = self.buffer.sample(int(n * self.replay_batch_ratio), y.device)
        self.buffer.add(x, y, task_id)
        if xm is None:
            logits = model(x, t)
            return F.cross_entropy(logits, y) + model.aux_loss, logits
        # one backbone pass; stream uses the training mask, memory all seen classes
        f = model.encode(B.cat([x, xm]))
        logits = model.classify(f[:n], t)
        with self.memory_mask(model, task_id):
            logits_m = model.classify(f[n:], tm)
        loss = F.cross_entropy(torch.cat([logits, logits_m]), torch.cat([y, ym]))
        return loss + model.aux_loss, logits


def ERACE(memory_size: int = 500, replay_batch_ratio: float = 1.0) -> ExperienceReplay:
    """ER-ACE (Caccia et al., ICLR 2022): asymmetric cross-entropy. Incoming samples
    compete only with the current task's classes; memory samples with all seen classes."""
    s = ExperienceReplay(memory_size, replay_batch_ratio)
    s.train_mask = "current"
    return s

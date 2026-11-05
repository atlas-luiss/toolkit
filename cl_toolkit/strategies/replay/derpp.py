"""DER / DER++ (Buzzega et al., NeurIPS 2020)."""

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.replay.buffer import ReservoirBuffer


class DERPP(Strategy):
    """CE + alpha * MSE(logits, stored logits) + beta * CE(memory labels). beta=0 gives DER."""

    def __init__(self, memory_size: int = 500, alpha: float = 0.3, beta: float = 0.5):
        self.memory_size, self.alpha, self.beta = memory_size, alpha, beta
        self.buffer = ReservoirBuffer(memory_size)

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        logits = model(x, task_id)
        loss = F.cross_entropy(logits, y) + model.aux_loss
        if len(self.buffer):
            if self.alpha > 0:
                xm, _, tm, zm = self.buffer.sample(B.size(x), y.device)
                with self.memory_mask(model, task_id):
                    out = model(xm, tm).float()
                w = min(out.size(1), zm.size(1))
                out, zm = out[:, :w], zm[:, :w]
                ok = torch.isfinite(zm) & torch.isfinite(out)   # only units that existed when stored
                loss = loss + self.alpha * F.mse_loss(out[ok], zm[ok])
            if self.beta > 0:
                xm, ym, tm, _ = self.buffer.sample(B.size(x), y.device)
                with self.memory_mask(model, task_id):
                    loss = loss + self.beta * F.cross_entropy(model(xm, tm), ym)
        self.buffer.add(x, y, task_id, logits=logits)
        return loss, logits


def DER(memory_size: int = 500, alpha: float = 0.3) -> DERPP:
    return DERPP(memory_size, alpha, beta=0.0)

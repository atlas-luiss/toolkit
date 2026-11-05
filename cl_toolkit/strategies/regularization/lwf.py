"""Learning without Forgetting (Li & Hoiem, 2017)."""

from copy import deepcopy

import torch
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy


class LwF(Strategy):
    """CE on new data + KD from the previous model on *old* outputs only.

    Class-/Domain-IL: distil the old model's logits over previously seen classes.
    Task-IL: distil each previous task's head.
    """

    def __init__(self, lam: float = 1.0, temperature: float = 2.0):
        self.lam, self.temperature = lam, temperature
        self._old = None

    def on_task_start(self, task_id, train_loader, model):
        self._old = None
        if task_id > 0:
            self._old = deepcopy(model).eval()
            for p in self._old.parameters():
                p.requires_grad_(False)

    def _kd(self, new, old):
        T = self.temperature
        return F.kl_div(F.log_softmax(new.float() / T, 1), F.softmax(old.float() / T, 1),
                        reduction="batchmean") * T * T

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        feats = model.encode(x)
        logits = model.classify(feats, task_id)
        loss = F.cross_entropy(logits, y) + model.aux_loss
        if self._old is not None:
            with torch.no_grad():
                old_feats = self._old.encode(x)
            if self.scenario.shared_head:
                old_cls = self.scenario.seen_classes(task_id - 1)
                with torch.no_grad():
                    old_logits = self._old.get_head()(old_feats)[:, old_cls]
                loss = loss + self.lam * self._kd(model.get_head()(feats)[:, old_cls], old_logits)
            else:
                for j in range(task_id):
                    with torch.no_grad():
                        o = self._old.classify(old_feats, j)
                    loss = loss + self.lam * self._kd(model.classify(feats, j), o) / task_id
        return loss, logits

    def on_task_end(self, task_id, train_loader, model):
        self._old = None

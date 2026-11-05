"""iCaRL (Rebuffi et al., CVPR 2017)."""

from copy import deepcopy

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.replay.buffer import HerdingBuffer


class iCaRL(Strategy):
    """Herding exemplars + rehearsal + BCE with old-model distillation + nearest-mean-of-exemplars.

    ``memory_size`` is the total exemplar budget, shared across seen classes
    (per-class quota shrinks as classes arrive). Class-IL only.
    """

    scenarios = ("class",)

    def __init__(self, memory_size: int = 2000, replay_batch_ratio: float = 1.0):
        self.memory_size, self.replay_batch_ratio = memory_size, replay_batch_ratio
        self.memory = HerdingBuffer(0)
        self._old = None
        self._old_classes = []
        self._means = None

    def on_task_start(self, task_id, train_loader, model):
        self._old_classes = sorted(self.memory.store)
        if task_id > 0:
            self._old = deepcopy(model).eval()
            for p in self._old.parameters():
                p.requires_grad_(False)

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        n = B.size(x)
        if len(self.memory):
            mx, my = self.memory.all()
            idx = torch.randperm(B.size(mx))[: int(n * self.replay_batch_ratio)]
            x, y = B.cat([x, B.to_device(B.index(mx, idx), y.device)]), torch.cat([y, my[idx].to(y.device)])
        seen = self.scenario.seen_classes(task_id)
        with self.memory_mask(model, task_id):          # iCaRL scores all seen classes
            logits = model(x, task_id)[:, seen]
        target = torch.zeros_like(logits, dtype=torch.float)
        target[torch.arange(len(y)), torch.tensor([seen.index(c) for c in y.tolist()], device=y.device)] = 1
        if self._old is not None and self._old_classes:
            old_cols = [seen.index(c) for c in self._old_classes]
            with torch.no_grad():
                target[:, old_cols] = torch.sigmoid(self._old(x)[:, self._old_classes].float())
        loss = F.binary_cross_entropy_with_logits(logits.float(), target) + model.aux_loss
        return loss, _scatter(logits[:n], seen, model)

    def on_task_end(self, task_id, train_loader, model):
        n_cls = len(self.scenario.seen_classes(task_id))
        k = max(self.memory_size // n_cls, 1)
        self.memory.reduce(k)
        loader = self.scenario.feature_loader(task_id)   # train data, test-time transform, no shuffle
        feats, ys = [], []
        with torch.no_grad():
            for batch in loader:
                with self.trainer.autocast():
                    feats.append(model.encode(B.to_device(batch[0], model._device)).float().cpu())
                ys.append(batch[1])
        feats, ys = torch.cat(feats), torch.cat(ys)
        for c in ys.unique().tolist():
            pos = torch.nonzero(ys == c).flatten()
            chosen = pos[HerdingBuffer.herd(feats[pos], k)]
            self.memory.store[c] = B.stack([loader.dataset[int(i)][0] for i in chosen])
        self._means = self._class_means(model)
        self._old = None

    @torch.no_grad()
    def _class_means(self, model):
        classes = sorted(self.memory.store)
        means = []
        for c in classes:
            ex = B.to_device(self.memory.store[c], model._device)
            with self.trainer.autocast():
                f = torch.cat([F.normalize(model.encode(B.index(ex, torch.arange(i, min(i + 256, B.size(ex))))).float(), dim=1)
                               for i in range(0, B.size(ex), 256)])
            means.append(F.normalize(f.mean(0), dim=0))
        return torch.tensor(classes, device=model._device), torch.stack(means)

    @torch.no_grad()
    def evaluate_task(self, test_loader, model, task_id):
        if self._means is None:
            return None
        cls, means = self._means
        correct = total = 0
        for x, y in test_loader:
            f = F.normalize(model.encode(B.to_device(x, model._device)).float(), dim=1)
            pred = cls[(f @ means.T).argmax(1)]
            correct += (pred.cpu() == y).sum().item()
            total += len(y)
        return {"accuracy": 100.0 * correct / max(total, 1)}


def _scatter(sub_logits, cols, model):
    """Place logits over ``cols`` back into a full-width tensor (for train accuracy)."""
    full = sub_logits.new_full((sub_logits.size(0), model.get_head().num_classes), float("-inf"))
    full[:, cols] = sub_logits
    return full.detach()

"""Quadratic-penalty methods: EWC, Online EWC, MAS."""

from typing import Dict, List, Tuple

import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from cl_toolkit.strategies.base import Strategy


class _QuadraticPenalty(Strategy):
    """loss = CE + lam/2 * sum_k sum_i Omega_k,i (theta_i - theta*_k,i)^2.

    * ``per_task_anchors=True`` (EWC): one term per past task k, anchored at the weights
      after task k (Kirkpatrick et al. 2017, Eq. 3 applied to every previous task).
    * otherwise a single term: Omega accumulated (``decay=None``: summed; float: decayed,
      Online EWC), anchored at the latest weights.

    Importances use per-sample gradients over ``num_samples`` training samples (not
    squared batch-mean gradients, which are a different, smaller quantity).
    Keyed by the Parameter object, so wrapping the backbone later (e.g. LoRA) keeps them.
    """

    decay = None
    per_task_anchors = False

    def __init__(self, lam: float = 1000.0, num_samples: int = 512):
        self.lam, self.num_samples = lam, num_samples
        self._terms: List[Tuple[Dict[torch.nn.Parameter, torch.Tensor], Dict[torch.nn.Parameter, torch.Tensor]]] = []

    def penalty(self) -> torch.Tensor:
        terms = [(w * (p - star[p]).pow(2)).sum() for omega, star in self._terms
                 for p, w in omega.items() if p.requires_grad]
        return (self.lam / 2) * torch.stack(terms).sum() if terms else 0.0

    def compute_loss(self, batch, model, task_id):
        loss, logits = super().compute_loss(batch, model, task_id)
        return loss + self.penalty(), logits

    def importance(self, loader, model, task_id) -> Dict[torch.nn.Parameter, torch.Tensor]:
        raise NotImplementedError

    def on_task_end(self, task_id, train_loader, model):
        new = self.importance(train_loader, model, task_id)
        star = {p: p.detach().clone() for p in new}
        if self.per_task_anchors or not self._terms:
            self._terms.append((new, star))
            return
        old, _ = self._terms[0]
        for p, w in new.items():
            o = old.get(p)
            old[p] = w if o is None else (o + w if self.decay is None else self.decay * o + w)
        self._terms[0] = (old, {p: p.detach().clone() for p in old})

    def _accumulate(self, loader, model, task_id, loss_fn):
        """Mean over samples of per-sample grad^2 (``_squared``) or |grad|."""
        params = [p for p in model.parameters() if p.requires_grad]
        acc = {p: torch.zeros_like(p, dtype=torch.float32) for p in params}
        n = 0
        with torch.enable_grad():
            for batch in loader:
                x, y = self.to_device(batch, model)
                for i in range(B.size(x)):
                    if n >= self.num_samples:
                        break
                    xi, yi = B.index(x, torch.tensor([i], device=y.device)), y[i:i + 1]
                    model.zero_grad(set_to_none=True)
                    with self.trainer.autocast():
                        out = model(xi, task_id).float()
                    loss_fn(out, yi).backward()
                    for p in params:
                        if p.grad is not None:
                            g = p.grad.detach().float()
                            acc[p] += g.pow(2) if self._squared else g.abs()
                    n += 1
                if n >= self.num_samples:
                    break
        model.zero_grad(set_to_none=True)
        return {p: a / max(n, 1) for p, a in acc.items()}


class EWC(_QuadraticPenalty):
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017): diagonal Fisher (labels sampled
    from the model's predictive distribution), one penalty per previous task."""

    _squared = True
    per_task_anchors = True

    def __init__(self, lam: float = 1000.0, fisher_samples: int = 512):
        super().__init__(lam, fisher_samples)

    def importance(self, loader, model, task_id):
        def nll(logits, y):
            logp = F.log_softmax(logits, 1)
            probs = logp.exp().nan_to_num(0)
            return F.nll_loss(logp.nan_to_num(0), torch.multinomial(probs, 1).squeeze(1))
        return self._accumulate(loader, model, task_id, nll)


class EWCOnline(EWC):
    """Online EWC (Schwarz et al. 2018): running Fisher decayed by ``gamma``."""

    per_task_anchors = False

    def __init__(self, lam: float = 1000.0, gamma: float = 0.9, fisher_samples: int = 512):
        super().__init__(lam, fisher_samples)
        self.gamma = self.decay = gamma


class MAS(_QuadraticPenalty):
    """Memory Aware Synapses (Aljundi et al. 2018): importance = mean over samples of
    |d ||f(x)||_2^2 / d theta| (label-free), accumulated over tasks."""

    _squared = False

    def __init__(self, lam: float = 1.0, importance_samples: int = 512):
        super().__init__(lam * 2, importance_samples)   # MAS uses lam * sum, not lam/2

    def importance(self, loader, model, task_id):
        def l2(logits, _):
            logits = logits.masked_fill(~torch.isfinite(logits), 0)
            return logits.pow(2).sum(1).mean()
        return self._accumulate(loader, model, task_id, l2)

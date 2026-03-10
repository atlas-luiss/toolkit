"""Replay memory buffers: ReservoirBuffer (reservoir sampling) and HerdingBuffer (herding selection)."""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List


class ReservoirBuffer:
    """Fixed-capacity memory buffer using reservoir sampling; optionally stores logits (for DER++)."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._n_seen  = 0

        self.xs:     List[torch.Tensor]           = []
        self.ys:     List[torch.Tensor]           = []
        self.tids:   List[int]                    = []
        self.logits: List[Optional[torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.xs)

    # ------------------------------------------------------------------

    def add(
        self,
        x:      torch.Tensor,
        y:      torch.Tensor,
        tid:    int,
        logit:  Optional[torch.Tensor] = None,
    ) -> None:
        self._n_seen += 1
        if len(self.xs) < self.capacity:
            self.xs.append(x)
            self.ys.append(y)
            self.tids.append(tid)
            self.logits.append(logit)
        else:
            idx = int(torch.randint(0, self._n_seen, (1,)).item())
            if idx < self.capacity:
                self.xs[idx]     = x
                self.ys[idx]     = y
                self.tids[idx]   = tid
                self.logits[idx] = logit

    def add_batch(
        self,
        xs:     torch.Tensor,
        ys:     torch.Tensor,
        tid:    int,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        xs = xs.detach().cpu()
        ys = ys.detach().cpu()
        if logits is not None:
            logits = logits.detach().cpu()
        for i in range(xs.size(0)):
            lgt = logits[i] if logits is not None else None
            self.add(xs[i], ys[i], tid, lgt)

    # ------------------------------------------------------------------

    def sample(self, n: int):
        """Random sample of up to n items; returns (xs, ys, tids, logits) on device."""
        if len(self.xs) == 0:
            return None, None, None, None
        n = min(n, len(self.xs))
        idx = torch.randperm(len(self.xs))[:n].tolist()
        xs     = torch.stack([self.xs[i]  for i in idx]).to(self.device)
        ys     = torch.stack([self.ys[i]  for i in idx]).to(self.device)
        tids   = [self.tids[i] for i in idx]
        logits = (
            torch.stack([self.logits[i] for i in idx]).to(self.device)
            if self.logits[0] is not None else None
        )
        return xs, ys, tids, logits

    def as_tensor_dataset(self):
        """Return a TensorDataset of all buffered (x, y) pairs."""
        xs = torch.stack(self.xs)
        ys = torch.stack(self.ys)
        return torch.utils.data.TensorDataset(xs, ys)


class HerdingBuffer:
    """Class-balanced exemplar store using herding selection (greedy class-mean approximation)."""

    def __init__(self, exemplars_per_class: int, device: torch.device):
        self.k      = exemplars_per_class
        self.device = device
        self.store: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(v['inputs'].size(0) for v in self.store.values())

    def update_class(
        self,
        class_id:   int,
        features:   torch.Tensor,   # (N, D) – already extracted, CPU
        raw_inputs: torch.Tensor,   # (N, C, H, W) – original images, CPU
    ) -> None:
        """Select k exemplars for class_id via herding and store them."""
        features = F.normalize(features.float(), dim=1)
        class_mean = features.mean(0)              # (D,)

        selected_idx = []
        running_sum  = torch.zeros_like(class_mean)

        for _ in range(min(self.k, features.size(0))):
            candidate_means = (running_sum.unsqueeze(0) + features) / (len(selected_idx) + 1)
            dists = torch.norm(candidate_means - class_mean.unsqueeze(0), dim=1)
            for j in selected_idx:
                dists[j] = float('inf')
            best = int(dists.argmin().item())
            selected_idx.append(best)
            running_sum += features[best]

        sel = torch.tensor(selected_idx)
        self.store[class_id] = {
            'inputs':   raw_inputs[sel].cpu(),
            'features': features[sel].cpu(),
        }

    def class_mean(self, class_id: int) -> torch.Tensor:
        """Normalised mean feature vector for a class (CPU)."""
        feats = self.store[class_id]['features'].float()
        return F.normalize(feats.mean(0, keepdim=True), dim=1).squeeze(0)

    def all_exemplars(self):
        """Return (xs, ys) across all stored classes (CPU tensors)."""
        xs, ys = [], []
        for cid, data in self.store.items():
            xs.append(data['inputs'])
            ys.append(torch.full((data['inputs'].size(0),), cid, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

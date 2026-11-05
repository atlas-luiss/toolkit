"""Replay memories. Tensor-backed, stored on CPU, sampled to the model's device."""

from __future__ import annotations

from typing import Optional

import torch

from cl_toolkit import batch as B


class ReservoirBuffer:
    """Reservoir sampling over the whole stream. Stores x, y, task id and (optionally) logits."""

    def __init__(self, capacity: int, device=None):
        self.capacity, self.device = capacity, device
        self.n_seen = 0
        self.x = self.y = self.t = self.z = None
        self.size = 0

    def __len__(self):
        return self.size

    def _alloc(self, x, z):
        self.x = B.empty_like_batch(x, self.capacity)
        self.y = torch.empty(self.capacity, dtype=torch.long)
        self.t = torch.empty(self.capacity, dtype=torch.long)
        if z is not None:
            self.z = torch.full((self.capacity, z.size(1)), float("nan"))

    def add(self, x: torch.Tensor, y: torch.Tensor, task_id, logits: Optional[torch.Tensor] = None):
        x, y = B.detach_cpu(x), y.detach().cpu()
        n = B.size(x)
        z = None if logits is None else logits.detach().float().cpu()
        tids = task_id.detach().cpu() if torch.is_tensor(task_id) else torch.full((n,), int(task_id))
        if self.x is None:
            self._alloc(x, z)
        for i in range(n):
            if self.size < self.capacity:
                j = self.size
                self.size += 1
            else:
                j = int(torch.randint(0, self.n_seen + 1, ()).item())
                if j >= self.capacity:
                    self.n_seen += 1
                    continue
            B.assign(self.x, j, x, i)
            self.y[j], self.t[j] = y[i], tids[i]
            if z is not None:
                if z.size(1) > self.z.size(1):   # head grew (task-IL heads of different size)
                    pad = torch.full((self.capacity, z.size(1) - self.z.size(1)), float("nan"))
                    self.z = torch.cat([self.z, pad], 1)
                self.z[j] = float("nan")
                self.z[j, : z.size(1)] = z[i]
            self.n_seen += 1

    add_batch = add  # old name

    def sample(self, n: int, device=None):
        """Returns (x, y, task_ids, logits_or_None) on ``device``; all None if empty."""
        if self.size == 0 or n <= 0:
            return None, None, None, None
        idx = torch.randperm(self.size)[: min(n, self.size)]
        dev = device or self.device
        z = None if self.z is None else self.z[idx].to(dev)
        return B.to_device(B.index(self.x, idx), dev), self.y[idx].to(dev), self.t[idx].to(dev), z

    def all(self):
        keep = torch.arange(self.size)
        return B.index(self.x, keep), self.y[: self.size], self.t[: self.size]


class ClassBalancedBuffer:
    """Greedy class-balanced memory (GDumb sampler): evict from the largest class."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = {}   # class key -> list of (x, y, t)

    def __len__(self):
        return sum(len(v) for v in self.items.values())

    def add(self, x, y, task_id):
        x, n = B.detach_cpu(x), B.size(x)
        tids = task_id if torch.is_tensor(task_id) else torch.full((n,), int(task_id))
        for i, (yi, ti) in enumerate(zip(y.detach().cpu(), tids.cpu())):
            xi = B.index(x, torch.tensor([i]))
            key = (int(ti), int(yi))
            n_cls = len(self.items) + (key not in self.items)
            if len(self) < self.capacity:
                self.items.setdefault(key, []).append((xi, yi, ti))
            elif len(self.items.get(key, [])) < self.capacity // n_cls:
                biggest = max(self.items, key=lambda k: len(self.items[k]))
                self.items[biggest].pop(torch.randint(len(self.items[biggest]), ()).item())
                self.items.setdefault(key, []).append((xi, yi, ti))

    def all(self):
        flat = [it for v in self.items.values() for it in v]
        return (B.cat([a for a, _, _ in flat]), torch.stack([b for _, b, _ in flat]),
                torch.stack([c for _, _, c in flat]))


class HerdingBuffer:
    """iCaRL exemplar store: herding selection per class (keys are global class ids)."""

    def __init__(self, exemplars_per_class: int, device=None):
        self.k = exemplars_per_class
        self.store = {}   # class -> inputs (k, ...)

    def __len__(self):
        return sum(B.size(v) for v in self.store.values())

    @staticmethod
    def herd(features: torch.Tensor, k: int) -> torch.Tensor:
        f = torch.nn.functional.normalize(features.float(), dim=1)
        mu, acc, chosen = f.mean(0), torch.zeros_like(f[0]), []
        avail = torch.ones(len(f), dtype=torch.bool)
        for i in range(min(k, len(f))):
            d = ((mu - (acc + f) / (i + 1)) ** 2).sum(1)
            d[~avail] = float("inf")
            j = int(d.argmin())
            chosen.append(j)
            avail[j] = False
            acc += f[j]
        return torch.tensor(chosen, dtype=torch.long)

    def update_class(self, cls: int, features: torch.Tensor, inputs: torch.Tensor) -> None:
        self.store[cls] = B.detach_cpu(B.index(inputs, self.herd(features, self.k)))

    def reduce(self, k: int) -> None:
        """Keep the first k (herding order = priority order)."""
        self.k = k
        self.store = {c: B.index(v, torch.arange(min(k, B.size(v)))) for c, v in self.store.items()}

    def all(self):
        xs = B.cat(list(self.store.values()))
        ys = torch.cat([torch.full((B.size(v),), c, dtype=torch.long) for c, v in self.store.items()])
        return xs, ys

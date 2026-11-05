"""A built-in toy benchmark for trying things in seconds on a CPU (no download).

    clt.run(data="toy", model="mlp", strategy="er")

Well-separated Gaussian classes as 32-dim vectors. The numbers it gives say nothing about
real data; use it to learn the API, debug a new method, or run tests.
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from .base import CLBenchmark
from .split import split_by_class


def toy_data(num_classes: int = 20, dim: int = 32, train_per_class: int = 100, test_per_class: int = 50,
             seed: int = 0):
    """``(train, test)`` TensorDatasets of ``num_classes`` Gaussian classes in ``dim`` dimensions."""
    means = torch.randn(num_classes, dim, generator=torch.Generator().manual_seed(seed)) * 2.0

    def make(n, s):
        g = torch.Generator().manual_seed(s)
        y = torch.arange(num_classes).repeat_interleave(n)
        return TensorDataset(means[y] + torch.randn(len(y), dim, generator=g), y)
    return make(train_per_class, seed + 1), make(test_per_class, seed + 2)


def toy(root=None, num_tasks: int = 10, seed: int = 42, transforms=None, num_classes: int = 20, dim: int = 32,
        **kw) -> CLBenchmark:
    """The toy data split into ``num_tasks`` class-incremental tasks."""
    bench = split_by_class(*toy_data(num_classes, dim), num_tasks, seed=seed, name="toy", **kw)
    bench.default_num_workers = 0          # in memory: worker processes would only add overhead
    return bench

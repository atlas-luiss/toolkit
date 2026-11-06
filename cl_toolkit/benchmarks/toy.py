"""Built-in toy benchmarks for trying things in seconds (no download).

    clt.run(data="toy", model="mlp", strategy="er")                                   # vectors
    clt.run(data="toy-images", model="vit_tiny_patch16_224.augreg_in21k", strategy="l2p")   # images

``toy``: well-separated Gaussian classes as 32-dim vectors. ``toy-images``: 224x224 RGB images,
one smooth random pattern per class plus noise. The numbers they give say nothing about real
data; use them to learn the API, debug a new method, or run tests.
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from .base import CLBenchmark
from .split import split_by_class


def toy_data(num_classes: int = 20, dim: int = 32, train_per_class: int = 100, test_per_class: int = 50,
             seed: int = 0, image_size=None):
    """``(train, test)`` TensorDatasets of ``num_classes`` Gaussian classes in ``dim`` dimensions,
    or, with ``image_size``, of ``(3, image_size, image_size)`` images (one pattern per class)."""
    g0 = torch.Generator().manual_seed(seed)
    if image_size:
        coarse = torch.rand(num_classes, 3, 4, 4, generator=g0)
        means = torch.nn.functional.interpolate(coarse, size=(image_size, image_size), mode="bilinear")

        def make(n, s):
            g = torch.Generator().manual_seed(s)
            y = torch.arange(num_classes).repeat_interleave(n)
            x = (means[y] + 0.15 * torch.randn(len(y), 3, image_size, image_size, generator=g)).clamp(0, 1)
            return TensorDataset(x, y)
        return make(train_per_class, seed + 1), make(test_per_class, seed + 2)
    means = torch.randn(num_classes, dim, generator=g0) * 2.0

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


def toy_images(root=None, num_tasks: int = 10, seed: int = 42, transforms=None, num_classes: int = 20,
               image_size: int = 224, train_per_class: int = 50, test_per_class: int = 10, **kw) -> CLBenchmark:
    """Small synthetic image benchmark (224x224 by default, for pre-trained ViTs)."""
    data = toy_data(num_classes, train_per_class=train_per_class, test_per_class=test_per_class,
                    image_size=image_size)
    bench = split_by_class(*data, num_tasks, seed=seed, name="toy-images", **kw)
    bench.default_num_workers = 0
    return bench

"""Tiny, learnable fixtures: Gaussian-class benchmark + MLP backbone with a ``qkv`` module (for LoRA)."""

import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cl_toolkit.benchmarks import split_by_class, split_by_domain  # noqa: E402
from cl_toolkit.models.base import CLModel  # noqa: E402

D = 32


def gauss_data(n_classes=20, n_train=80, n_test=40, sep=0.6, seed=0, shift=None):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n_classes, D, generator=g) * sep

    def mk(n):
        y = torch.arange(n_classes).repeat_interleave(n)
        x = mu[y] + torch.randn(len(y), D, generator=g)
        if shift is not None:
            x = x + shift
        return TensorDataset(x, y)
    return mk(n_train), mk(n_test)


def gauss_benchmark(num_tasks=4, n_classes=20, **kw):
    tr, te = gauss_data(n_classes, **kw)
    return split_by_class(tr, te, num_tasks, shuffle_classes=False)


def gauss_domains(num_tasks=3, n_classes=5):
    doms = []
    for t in range(num_tasks):
        shift = torch.randn(D, generator=torch.Generator().manual_seed(100 + t)) * 1.5
        doms.append(gauss_data(n_classes, seed=0, shift=shift, sep=1.0))
    return split_by_domain(doms)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.nn.functional.gelu(self.qkv(x))


class MLP(CLModel):
    def __init__(self, H=64, device="cpu"):
        super().__init__(device)
        self._backbone = nn.Sequential(nn.Linear(D, H), Block(H), Block(H))
        self._H = H

    def _encode(self, x):
        return self._backbone(x)

    @property
    def embed_dim(self):
        return self._H


@pytest.fixture
def bench():
    return gauss_benchmark()


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MLP()

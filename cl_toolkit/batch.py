"""Model-agnostic batch helpers.

Model inputs ``x`` may be a tensor (images), a dict of tensors (tokenised text:
``{"input_ids": ..., "attention_mask": ...}``), or a list/tuple of tensors. All
strategies and buffers go through these helpers, so any model whose ``encode``
accepts that structure works with every method in the toolkit.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


def to_device(x: Any, device, non_blocking: bool = True) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device, non_blocking) for v in x)
    return x


def size(x: Any) -> int:
    """Batch size (first dimension)."""
    if torch.is_tensor(x):
        return x.size(0)
    if isinstance(x, dict):
        return size(next(iter(x.values())))
    return size(x[0])


def index(x: Any, idx) -> Any:
    """``x[idx]`` along the batch dimension, for any structure."""
    if torch.is_tensor(x):
        return x[idx.to(x.device) if torch.is_tensor(idx) else idx]
    if isinstance(x, dict):
        return {k: index(v, idx) for k, v in x.items()}
    return type(x)(index(v, idx) for v in x)


def cat(xs: Sequence[Any]) -> Any:
    first = xs[0]
    if torch.is_tensor(first):
        return torch.cat(list(xs))
    if isinstance(first, dict):
        return {k: cat([x[k] for x in xs]) for k in first}
    return type(first)(cat([x[i] for x in xs]) for i in range(len(first)))


def stack(xs: Sequence[Any]) -> Any:
    """Stack single samples into a batch."""
    first = xs[0]
    if torch.is_tensor(first):
        return torch.stack(list(xs))
    if isinstance(first, dict):
        return {k: stack([x[k] for x in xs]) for k in first}
    return type(first)(stack([x[i] for x in xs]) for i in range(len(first)))


def detach_cpu(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: detach_cpu(v) for k, v in x.items()}
    return type(x)(detach_cpu(v) for v in x)


def empty_like_batch(x: Any, n: int) -> Any:
    """Uninitialised storage for ``n`` samples shaped like batch ``x``."""
    if torch.is_tensor(x):
        return torch.empty((n, *x.shape[1:]), dtype=x.dtype)
    if isinstance(x, dict):
        return {k: empty_like_batch(v, n) for k, v in x.items()}
    return type(x)(empty_like_batch(v, n) for v in x)


def assign(store: Any, j: int, x: Any, i: int) -> None:
    """``store[j] = x[i]`` for any structure."""
    if torch.is_tensor(store):
        store[j] = x[i]
    elif isinstance(store, dict):
        for k in store:
            assign(store[k], j, x[k], i)
    else:
        for s, v in zip(store, x):
            assign(s, j, v, i)

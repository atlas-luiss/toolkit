"""CLModel: encoder + classifier head. Subclass with ``_encode`` and ``embed_dim``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

TaskIds = Union[None, int, torch.Tensor]


class CosineLinear(nn.Module):
    """Cosine classifier: ``scale * cos(w_c, x)`` with a learnable scale (LUCIR, Hou et al., CVPR 2019).

    Removes the weight-norm bias towards recent classes that a plain linear head learns.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 16.0, learn_scale: bool = True,
                 device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.scale = nn.Parameter(torch.tensor(float(scale), device=device), requires_grad=learn_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        return self.scale * nn.functional.normalize(x, dim=-1) @ nn.functional.normalize(self.weight, dim=-1).T


LAYERS = {"linear": nn.Linear, "cosine": CosineLinear}


def _make_layer(layer, d_in: int, d_out: int, device=None) -> nn.Module:
    layer = LAYERS[layer] if isinstance(layer, str) else layer
    return layer(d_in, d_out, device=device)


class IncrementalClassifier(nn.Module):
    """Shared head (Class-IL / Domain-IL). Can grow, preserving old weights.

    ``layer``: "linear" (default), "cosine" (:class:`CosineLinear`) or any
    ``(in_features, out_features, device=) -> nn.Module`` with ``.weight`` rows per class.
    """

    def __init__(self, embed_dim: int, initial_classes: int = 0, layer="linear"):
        super().__init__()
        self.embed_dim, self.layer = embed_dim, layer
        self._fc: Optional[nn.Module] = _make_layer(layer, embed_dim, initial_classes) if initial_classes else None

    def grow(self, n_new: int) -> None:
        if self._fc is None:
            self._fc = _make_layer(self.layer, self.embed_dim, n_new)
            return
        old = self._fc
        new = _make_layer(self.layer, self.embed_dim, old.out_features + n_new, device=old.weight.device)
        with torch.no_grad():
            new.weight[: old.out_features].copy_(old.weight)
            if getattr(old, "bias", None) is not None and getattr(new, "bias", None) is not None:
                new.bias[: old.out_features].copy_(old.bias)
            if hasattr(old, "scale") and hasattr(new, "scale"):
                new.scale.copy_(old.scale)
        self._fc = new

    @property
    def num_classes(self) -> int:
        return 0 if self._fc is None else self._fc.out_features

    def reset_parameters(self) -> None:
        if self._fc is not None:
            self._fc.reset_parameters()

    def forward(self, x):
        if self._fc is None:
            raise RuntimeError("IncrementalClassifier has no classes; call grow() first")
        return self._fc(x)


class MultiHeadClassifier(nn.Module):
    """One head per task (Task-IL). ``layer`` as in :class:`IncrementalClassifier`."""

    def __init__(self, embed_dim: int, layer="linear"):
        super().__init__()
        self.embed_dim, self.layer = embed_dim, layer
        self._heads = nn.ModuleDict()

    def add_head(self, task_id: int, num_classes: int) -> None:
        self._heads[str(task_id)] = _make_layer(self.layer, self.embed_dim, num_classes)

    def reset_parameters(self) -> None:
        for h in self._heads.values():
            h.reset_parameters()

    def forward(self, x: torch.Tensor, task_id: TaskIds) -> torch.Tensor:
        if task_id is None:
            raise ValueError("MultiHeadClassifier needs a task id (Task-IL scenario)")
        if not torch.is_tensor(task_id):
            return self._heads[str(int(task_id))](x)
        # Per-sample task ids (e.g. replay batches mixing tasks): pad with -inf.
        tids = task_id.to(x.device)
        width = max(self._heads[str(int(t))].out_features for t in tids.unique().tolist())
        out = x.new_full((x.size(0), width), float("-inf"))
        for t in tids.unique().tolist():
            m = tids == t
            o = self._heads[str(int(t))](x[m])
            out[m, : o.size(1)] = o.to(out.dtype)
        return out


class CLModel(nn.Module, ABC):
    """Abstract CL model.

    Contract for subclasses:
      * store the feature extractor as ``self._backbone``
      * implement ``_encode(x) -> (B, embed_dim)`` and ``embed_dim``

    Optional hooks: ``begin_task(task_id, num_tasks)``, ``end_task(task_id)``,
    and ``aux_loss`` (e.g. prompt-key matching) which built-in strategies add.
    """

    def __init__(self, device: Union[None, str, torch.device] = None):
        super().__init__()
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._head: Optional[nn.Module] = None
        self._class_mask: Optional[torch.Tensor] = None   # allowed output units (shared head)
        #: default LoRA placement for every LoRA-based method (see cl_toolkit.adapters)
        self.lora_targets = None      # e.g. ["qkv", "fc1"], a regex, or "all-linear"
        self.lora_layers = None       # e.g. [9, 10, 11]: only these blocks

    # -- subclass contract ----------------------------------------------------
    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    def _default_lora_targets(self) -> List[str]:
        return ["qkv"]

    # -- optional hooks ---------------------------------------------------------
    def begin_task(self, task_id: int, num_tasks: int) -> None: ...
    def end_task(self, task_id: int) -> None: ...

    @property
    def aux_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self._device)

    # -- forward ------------------------------------------------------------------
    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def classify(self, features: torch.Tensor, task_id: TaskIds = None) -> torch.Tensor:
        if self._head is None:
            raise RuntimeError("No head; call scenario.setup(model) first")
        if isinstance(self._head, MultiHeadClassifier):
            return self._head(features, task_id)
        logits = self._head(features)
        if self._class_mask is not None:
            logits = logits.masked_fill(~self._class_mask.to(logits.device), float("-inf"))
        return logits

    def forward(self, x: torch.Tensor, task_id: TaskIds = None) -> torch.Tensor:
        return self.classify(self.encode(x), task_id)

    def set_class_mask(self, classes: Optional[Sequence[int]]) -> None:
        """Restrict shared-head logits to ``classes`` (None = no restriction)."""
        if classes is None or self._head is None or isinstance(self._head, MultiHeadClassifier):
            self._class_mask = None
            return
        m = torch.zeros(self._head.num_classes, dtype=torch.bool, device=self._device)
        m[list(classes)] = True
        self._class_mask = m

    @contextmanager
    def class_mask(self, classes: Optional[Sequence[int]]):
        """Temporarily use a different logit mask: ``with model.class_mask(seen): ...``"""
        old = self._class_mask
        self.set_class_mask(classes)
        try:
            yield
        finally:
            self._class_mask = old

    # -- head management ------------------------------------------------------------
    def set_head(self, head: nn.Module) -> None:
        self._head = head.to(self._device)

    def get_head(self) -> nn.Module:
        return self._head

    def grow_head(self, n: int) -> None:
        self._head.grow(n)
        self._head.to(self._device)

    def add_task_head(self, task_id: int, num_classes: int) -> None:
        self._head.add_head(task_id, num_classes)
        self._head.to(self._device)

    # -- params ---------------------------------------------------------------------
    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_backbone(self) -> None:
        """Freeze backbone weights; keeps LoRA / adapter / prompt / head params as they are."""
        for name, p in self._backbone.named_parameters():
            if "lora_" not in name and "ctk_adapter" not in name:
                p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for p in self._backbone.parameters():
            p.requires_grad_(True)

    def num_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev = kwargs.get("device", args[0] if args and isinstance(args[0], (str, torch.device)) else None)
        if dev is not None:
            self._device = torch.device(dev)
        return out

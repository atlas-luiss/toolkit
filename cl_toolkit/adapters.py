r"""Parameter-efficient modules for CLModel backbones: LoRA (via peft) and AdaptFormer.

All functions take a :class:`~cl_toolkit.models.base.CLModel` and mutate ``model._backbone``.

LoRA::

    lora_candidates(model)                           # which layer names exist
    add_lora(model, "task_0", rank=8)                # default targets (ViT: "qkv")
    add_lora(model, "t", targets=["qkv", "fc1"], layers=[9, 10, 11])   # choose layers + blocks
    add_lora(model, "t", targets=r".*attn\.(qkv|proj)")               # regex on full names
    add_lora(model, "t", targets="all-linear")
    activate_lora(model, ["task_0", "task_1"], trainable=["task_1"])  # outputs add up
    merge_lora(model, "task_0")                                       # fold into base weights

AdaptFormer (parallel bottleneck MLP adapter, Chen et al. 2022)::

    add_adaptformer(model, bottleneck=64)
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------------

def _is_lora(model) -> bool:
    from peft.tuners.tuners_utils import BaseTuner
    return isinstance(model._backbone, BaseTuner)


def lora_names(model) -> List[str]:
    return list(model._backbone.peft_config) if _is_lora(model) else []


_LORA_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)
_BLOCK = re.compile(r"(?:^|\.)(?:blocks|layers?|h|resblocks|block)\.(\d+)\.")
_ANY_INDEX = re.compile(r"(?:^|\.)(\d+)\.")


def _block_index(name: str) -> Optional[int]:
    """Block number of a module path: 'blocks.3.attn.qkv' -> 3, 'encoder.layer.5.x' -> 5, '2.qkv' -> 2."""
    m = _BLOCK.search(name) or _ANY_INDEX.search(name)
    return int(m.group(1)) if m else None


def _base(model):
    bb = model._backbone
    return bb.model if _is_lora(model) else bb


def _targetable(model):
    """(full name, module) of every layer LoRA can wrap. Layers already wrapped by peft
    appear as ``<name>.base_layer``; they are reported under ``<name>``."""
    out = {}
    for n, m in _base(model).named_modules():
        if not n or not isinstance(m, _LORA_TYPES) or "lora_" in n or "ctk_adapter" in n:
            continue
        out[n[: -len(".base_layer")] if n.endswith(".base_layer") else n] = m
    return list(out.items())


def lora_candidates(model) -> List[str]:
    """Distinct short names of layers LoRA can target (e.g. ``['fc1', 'fc2', 'proj', 'qkv']``)."""
    return sorted({n.split(".")[-1] for n, _ in _targetable(model)})


def resolve_lora_targets(model, targets: Union[None, str, Sequence[str]] = None,
                         layers: Optional[Sequence[int]] = None) -> List[str]:
    """Full module names LoRA will be applied to.

    ``targets``: None (model default: ``model.lora_targets`` or the model's built-in
    choice), a list of short names (``["qkv", "fc1"]``), a regex on full module
    names, or ``"all-linear"``. ``layers``: keep only modules inside these block
    indices (``blocks.<i>.`` for timm ViTs, ``layer.<i>.``/``layers.<i>.`` for HF).
    """
    if targets is None:
        targets = getattr(model, "lora_targets", None) or model._default_lora_targets()
    if layers is None:
        layers = getattr(model, "lora_layers", None)
    mods = _targetable(model)
    if targets == "all-linear":
        chosen = [n for n, m in mods if isinstance(m, nn.Linear)]
    elif isinstance(targets, str):
        rx = re.compile(targets)
        chosen = [n for n, _ in mods if rx.fullmatch(n)]
    else:
        short = set(targets)
        chosen = [n for n, _ in mods if n.split(".")[-1] in short]
    if layers is not None:
        keep = set(int(i) for i in layers)
        chosen = [n for n in chosen if _block_index(n) in keep]
    if not chosen:
        raise ValueError(f"LoRA targets {targets!r} (layers={layers}) match no module. "
                         f"Available layer names: {lora_candidates(model)}")
    return chosen


def add_lora(model, name: str, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
             targets: Union[None, str, Sequence[str]] = None, layers: Optional[Sequence[int]] = None,
             trainable_only: bool = True, target_modules=None) -> List[str]:
    """Attach a new LoRA adapter ``name`` (the only trainable one). Returns the adapted module names.

    See :func:`resolve_lora_targets` for ``targets`` / ``layers``
    (``target_modules`` is the old name of ``targets``).
    """
    from peft import LoraConfig, LoraModel
    modules = resolve_lora_targets(model, targets if targets is not None else target_modules, layers)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=modules)
    if _is_lora(model):
        model._backbone.peft_config[name] = cfg
        model._backbone.inject_adapter(model._backbone.model, name)
    else:
        model._backbone = LoraModel(model._backbone, cfg, adapter_name=name)
    model._backbone.to(model._device)
    activate_lora(model, lora_names(model), trainable=[name] if trainable_only else None)
    return modules


def activate_lora(model, names: Iterable[str], trainable: Optional[Iterable[str]] = None) -> None:
    """Make ``names`` active (their deltas are summed). Only ``trainable`` get gradients
    (default: keep current requires_grad flags)."""
    names = [n for n in names if n in lora_names(model)]
    if not names:
        return
    flags = {n: p.requires_grad for n, p in model._backbone.named_parameters() if "lora_" in n}
    model._backbone.set_adapter(names)  # NB: peft marks all active adapters trainable
    for n, p in model._backbone.named_parameters():
        if "lora_" not in n:
            continue
        if trainable is not None:
            p.requires_grad_(any(f".{t}." in n for t in trainable))
        else:
            p.requires_grad_(flags.get(n, False))


def freeze_lora(model, name: str) -> None:
    for n, p in model._backbone.named_parameters():
        if f".{name}." in n:
            p.requires_grad_(False)


def lora_state(model, name: str) -> dict:
    """Copy of the A/B weights of adapter ``name``."""
    return {n: p.detach().clone() for n, p in model._backbone.named_parameters() if f".{name}." in n}


def merge_lora(model, name: str) -> None:
    """Fold adapter ``name`` into the base weights and delete it."""
    if name not in lora_names(model):
        return
    bb = model._backbone
    active = [a for a in (bb.active_adapters if isinstance(bb.active_adapters, list) else [bb.active_adapter])
              if a != name]
    bb.merge_adapter(adapter_names=[name])
    # merged weights are now part of base; unmerge bookkeeping by deleting adapter
    for module in bb.modules():
        if hasattr(module, "merged_adapters") and name in module.merged_adapters:
            module.merged_adapters.remove(name)
    bb.delete_adapter(name)
    if not bb.peft_config:
        model._backbone = bb.model
        if hasattr(model._backbone, "peft_config"):
            del model._backbone.peft_config
    elif active:
        activate_lora(model, active)


# ----------------------------------------------------------------------------
# AdaptFormer
# ----------------------------------------------------------------------------

class _ParallelAdapter(nn.Module):
    """AdaptFormer's parallel AdaptMLP (Chen et al., NeurIPS 2022; the APER / RanPAC code):
    ``mlp(norm2(x)) + s * sum_{active} up(relu(down(x)))``. The adapter reads the block's
    input *before* ``norm2``, as in the authors' code, so the wrapper takes over ``norm2``
    (the block's own ``norm2`` becomes Identity; nothing else changes). Holds several
    *named* adapters (one per task for routing); ``up`` is zero-init so a new adapter
    starts at the PTM."""

    def __init__(self, mlp: nn.Module, norm: nn.Module, scale: float):
        super().__init__()
        self.mlp, self.norm, self.scale = mlp, norm, scale
        self.ctk_adapter = nn.ModuleDict()
        self.active: List[str] = []

    def add(self, name: str, dim: int, bottleneck: int, device=None) -> nn.Module:
        m = nn.Sequential(nn.Linear(dim, bottleneck), nn.ReLU(), nn.Linear(bottleneck, dim)).to(device)
        nn.init.zeros_(m[2].weight)
        nn.init.zeros_(m[2].bias)
        self.ctk_adapter[name] = m
        self.active.append(name)
        return m

    def forward(self, x):
        out = self.mlp(self.norm(x))
        for n in self.active:
            out = out + self.scale * self.ctk_adapter[n](x)
        return out


def _vit(model):
    bb = model._backbone
    bb = getattr(bb, "model", bb)  # unwrap peft
    if not hasattr(bb, "blocks"):
        raise TypeError("expected a timm VisionTransformer backbone (with .blocks)")
    return bb


def add_adaptformer(model, bottleneck: int = 64, scale: float = 0.1,
                    layers: Optional[Iterable[int]] = None, name: str = "default") -> List[nn.Parameter]:
    """Insert an AdaptFormer adapter ``name`` in the MLP of each (or selected) ViT block.

    It becomes active (summed with other active adapters). Returns its parameters.
    """
    vit = _vit(model)
    params = []
    for i, blk in enumerate(vit.blocks):
        if layers is not None and i not in set(layers):
            continue
        if not isinstance(blk.mlp, _ParallelAdapter):
            blk.mlp = _ParallelAdapter(blk.mlp, blk.norm2, scale).to(model._device)
            blk.norm2 = nn.Identity()
        if name in blk.mlp.ctk_adapter:
            continue
        params += list(blk.mlp.add(name, vit.embed_dim, bottleneck, model._device).parameters())
    return params


def adaptformer_names(model) -> List[str]:
    for blk in _vit(model).blocks:
        if isinstance(blk.mlp, _ParallelAdapter):
            return list(blk.mlp.ctk_adapter)
    return []


def set_adaptformer_active(model, names: Iterable[str]) -> None:
    """Which named AdaptFormer adapters are applied (``[]`` = none: the plain PTM)."""
    names = list(names)
    for blk in _vit(model).blocks:
        if isinstance(blk.mlp, _ParallelAdapter):
            blk.mlp.active = [n for n in names if n in blk.mlp.ctk_adapter]


def adaptformer_params(model, name: str) -> List[nn.Parameter]:
    return [p for blk in _vit(model).blocks if isinstance(blk.mlp, _ParallelAdapter)
            and name in blk.mlp.ctk_adapter for p in blk.mlp.ctk_adapter[name].parameters()]


def adapter_params(model) -> List[nn.Parameter]:
    """All AdaptFormer + LoRA parameters."""
    return [p for n, p in model._backbone.named_parameters() if "ctk_adapter" in n or "lora_" in n]


def set_adapters_trainable(model, flag: bool) -> None:
    for p in adapter_params(model):
        p.requires_grad_(flag)

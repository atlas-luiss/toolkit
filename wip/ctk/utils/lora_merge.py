"""
LoRA adapter merging utilities.

Reusable operations for building, concatenating, and hierarchically merging
LoRA adapters.  Used by HAM and available to any custom strategy.

Functions
---------
build_adapter_from_weights(base_model, weights, lora_config)
    Reconstruct a PEFT model from an ejected adapter weight dict.

concatenate_adapters(task_adapters, task_alphas, rank, base_model,
                     target_modules, lora_alpha, lora_dropout, top_k_ratio,
                     pad_to_size)
    Concatenate multiple LoRA adapters column-wise (with optional zero padding).

hierarchical_merge(group_adapters, group_weights, base_model,
                   combination_type, density)
    Merge a list of group-level PEFT models via PEFT add_weighted_adapter.
"""

from __future__ import annotations

import os
import shutil
import logging
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    raise ImportError("peft is required: pip install peft")

from ctk.utils.adapter_ops import prune_adapter_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build a PEFT model from ejected weights
# ---------------------------------------------------------------------------

def build_adapter_from_weights(
    base_model,
    weights:     Dict[str, torch.Tensor],
    lora_config: "LoraConfig",
) -> nn.Module:
    """
    Reconstruct a frozen PEFT model from an adapter weight dict.

    Typically used to rebuild an adapter that was ejected from a CLModel
    so it can be stored, compared, or merged independently.

    Args:
        base_model:  The underlying backbone (CLModel or nn.Module).
        weights:     Dict mapping ``"<module>.lora_A"`` /
                     ``"<module>.lora_B"`` keys to weight tensors,
                     as returned by ``CLModel.eject_lora()``.
        lora_config: PEFT LoraConfig specifying rank / alpha / modules.

    Returns:
        A PEFT model with the given weights loaded and all parameters frozen.
    """
    peft_model = get_peft_model(deepcopy(base_model), lora_config)

    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key not in weights or b_key not in weights:
                continue
            if isinstance(module.lora_A, nn.ModuleDict):
                key = list(module.lora_A.keys())[0]
                module.lora_A[key].weight.data.copy_(weights[a_key])
                module.lora_B[key].weight.data.copy_(weights[b_key])
            else:
                module.lora_A.weight.data.copy_(weights[a_key])
                module.lora_B.weight.data.copy_(weights[b_key])

    for p in peft_model.parameters():
        p.requires_grad = False

    return peft_model


# ---------------------------------------------------------------------------
# Concatenate adapters column-wise (intra-group merge)
# ---------------------------------------------------------------------------

def concat_lora_adapters(
    task_adapters:  List[nn.Module],
    task_alphas:    List[nn.Parameter],
    rank:           int,
    base_model,
    target_modules: List[str],
    lora_alpha:     float = 1.0,
    lora_dropout:   float = 0.0,
    top_k_ratio:    float = 1.0,
    pad_to_size:    Optional[int] = None,
) -> nn.Module:
    """
    Concatenate multiple LoRA adapters into a single wider adapter.

    For each LoRA module the A matrices are stacked row-wise and B matrices
    are stacked column-wise, producing a rank-``(rank * n_adapters)`` adapter.
    Weights are optionally magnitude-pruned before concatenation.
    Zero-padding is inserted for groups smaller than ``pad_to_size``.

    Args:
        task_adapters:  List of PEFT models (one per task in the group).
        task_alphas:    Learned alpha scalars — used to weight the B matrices.
        rank:           Per-task LoRA rank.
        base_model:     Backbone used to build the combined PEFT model.
        target_modules: Modules to apply LoRA to.
        lora_alpha:     Alpha for the combined adapter.
        lora_dropout:   Dropout for the combined adapter.
        top_k_ratio:    Fraction of weights to keep (magnitude pruning).
                        1.0 = no pruning.
        pad_to_size:    If set, zero-pad to this number of adapters total.
                        Useful to equalise ranks across groups.

    Returns:
        A new PEFT model whose LoRA matrices are the concatenation of inputs.
    """
    n_tasks     = len(task_adapters)
    n_slots     = pad_to_size if (pad_to_size and pad_to_size > n_tasks) else n_tasks
    total_rank  = n_slots * rank
    device      = next(iter(base_model.parameters())).device

    lora_config = LoraConfig(
        r              = total_rank,
        lora_alpha     = lora_alpha,
        target_modules = target_modules,
        lora_dropout   = lora_dropout,
        bias           = "none",
    )
    combined = get_peft_model(deepcopy(base_model), lora_config)

    with torch.no_grad():
        for mod_name, module in combined.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            A_parts, B_parts = [], []

            for i, adapter in enumerate(task_adapters):
                alpha_val = task_alphas[i].item() if i < len(task_alphas) else 1.0

                pruned = (
                    prune_adapter_weights(adapter, k_ratio=top_k_ratio, method="magnitude")
                    if top_k_ratio < 1.0 else adapter
                )

                for src_name, src_mod in pruned.named_modules():
                    if src_name != mod_name or not hasattr(src_mod, "lora_A"):
                        continue
                    if isinstance(src_mod.lora_A, nn.ModuleDict):
                        key = list(src_mod.lora_A.keys())[0]
                        A_parts.append(src_mod.lora_A[key].weight.data.clone())
                        B_parts.append(src_mod.lora_B[key].weight.data.clone() * alpha_val)
                    else:
                        A_parts.append(src_mod.lora_A.weight.data.clone())
                        B_parts.append(src_mod.lora_B.weight.data.clone() * alpha_val)
                    break

            # Zero-pad to pad_to_size
            while A_parts and len(A_parts) < n_slots:
                A_parts.append(torch.zeros(rank, A_parts[0].shape[1], device=device))
                B_parts.append(torch.zeros(B_parts[0].shape[0], rank, device=device))

            if not A_parts:
                continue

            A_cat = torch.cat(A_parts, dim=0)
            B_cat = torch.cat(B_parts, dim=1)

            if isinstance(module.lora_A, nn.ModuleDict):
                key = list(module.lora_A.keys())[0]
                module.lora_A[key].weight.data = A_cat
                module.lora_B[key].weight.data = B_cat
            else:
                module.lora_A.weight.data = A_cat
                module.lora_B.weight.data = B_cat

    return combined


# ---------------------------------------------------------------------------
# Hierarchical merge across groups (inter-group merge)
# ---------------------------------------------------------------------------

def merge_lora_groups(
    group_adapters:   List[nn.Module],
    group_weights:    List[float],
    base_model,
    combination_type: str   = "dare_ties",
    density:          float = 0.2,
) -> nn.Module:
    """
    Merge a list of group-level PEFT adapters into a single model.

    Uses PEFT's ``add_weighted_adapter`` (DARE-TIES by default) to combine
    group adapters with normalised weights.

    Args:
        group_adapters:   One PEFT model per group.
        group_weights:    Unnormalised importance weight for each group.
        base_model:       Backbone used if only one group is present.
        combination_type: PEFT combination method (``"dare_ties"``,
                          ``"ties"``, ``"linear"``, etc.).
        density:          Density parameter passed to DARE-TIES.

    Returns:
        A single PEFT model with adapter name ``"merged"``.
    """
    valid = [(a, w) for a, w in zip(group_adapters, group_weights) if a is not None]
    if not valid:
        raise ValueError("No valid group adapters to merge.")

    if len(valid) == 1:
        return valid[0][0]

    adapters, weights = zip(*valid)
    total = sum(weights)
    norm_weights = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

    temp_dir = "/tmp/ctk_ham_merge"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        names = []
        for i, adapter in enumerate(adapters):
            name = f"group_{i}"
            adapter.save_pretrained(f"{temp_dir}/{name}")
            names.append(name)

        merged = get_peft_model(
            deepcopy(base_model),
            adapters[0].peft_config["default"],
        )
        for name in names:
            merged.load_adapter(f"{temp_dir}/{name}", adapter_name=name)

        merged.add_weighted_adapter(
            names,
            list(norm_weights),
            "merged",
            combination_type=combination_type,
            density=density,
        )
        merged.set_adapter("merged")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(
        f"Hierarchical merge: {len(names)} groups → 'merged' "
        f"(weights: {[f'{w:.3f}' for w in norm_weights]})"
    )
    return merged

"""Inspection utilities: trainable params, adapter names, LoRA weights, sparsity."""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def check_trainable_params(
    model: nn.Module,
    verbose: bool = True
) -> Dict[str, Any]:
    """Print and return trainable vs total parameter counts."""
    trainable_params = 0
    all_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0

    stats = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
        "trainable_names": trainable_names
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Trainable Parameter Analysis")
        print(f"{'='*70}")
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable %: {trainable_percent:.4f}%")
        print(f"{'='*70}\n")

        if trainable_names and len(trainable_names) < 50:
            print("Trainable parameters:")
            for name in trainable_names:
                print(f"  - {name}")
            print()

    return stats


def get_adapter_names(model: nn.Module) -> List[str]:
    """Return adapter names from a PEFT model."""
    adapter_names = []

    if hasattr(model, 'peft_config'):
        adapter_names = list(model.peft_config.keys())
    elif hasattr(model, 'active_adapter'):
        if isinstance(model.active_adapter, str):
            adapter_names = [model.active_adapter]
        elif isinstance(model.active_adapter, list):
            adapter_names = model.active_adapter
    else:
        for name, module in model.named_modules():
            if 'lora' in name.lower():
                adapter_name = name.split('.')[0]
                if adapter_name not in adapter_names:
                    adapter_names.append(adapter_name)

    return adapter_names


def print_parameter_count(
    model: nn.Module,
    detailed: bool = False
) -> None:
    """Print total/trainable/frozen parameter counts, optionally per layer."""
    total = 0
    trainable = 0
    frozen = 0

    layer_info = []

    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count

        if param.requires_grad:
            trainable += param_count
            status = "trainable"
        else:
            frozen += param_count
            status = "frozen"

        layer_info.append((name, param_count, status))

    print(f"\n{'='*70}")
    print(f"Model Parameter Count")
    print(f"{'='*70}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Frozen parameters:    {frozen:,} ({100*frozen/total:.2f}%)")
    print(f"{'='*70}\n")

    if detailed:
        print(f"\n{'='*70}")
        print(f"Per-Layer Breakdown")
        print(f"{'='*70}")
        print(f"{'Layer Name':<50} {'Parameters':>12} {'Status':>10}")
        print(f"{'-'*70}")
        for name, count, status in layer_info:
            print(f"{name:<50} {count:>12,} {status:>10}")
        print(f"{'='*70}\n")


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return all LoRA parameters as a name→tensor dict."""
    lora_params = {}

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params[name] = param

    return lora_params


def count_nonzero_params(model: nn.Module) -> Dict[str, int]:
    """Count non-zero LoRA parameters; useful for measuring sparsity after pruning."""
    total_params = 0
    nonzero_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                if isinstance(module.lora_A, nn.ModuleDict):
                    adapter_key = list(module.lora_A.keys())[0]
                    A_weight = module.lora_A[adapter_key].weight.data
                    B_weight = module.lora_B[adapter_key].weight.data
                else:
                    A_weight = module.lora_A.weight.data
                    B_weight = module.lora_B.weight.data

                total_params += A_weight.numel() + B_weight.numel()
                nonzero_params += (A_weight != 0).sum().item() + (B_weight != 0).sum().item()
            except (AttributeError, IndexError):
                continue

    sparsity = 1 - (nonzero_params / total_params) if total_params > 0 else 0

    return {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'zero_params': total_params - nonzero_params,
        'sparsity': sparsity
    }

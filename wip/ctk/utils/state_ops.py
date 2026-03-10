"""State dict operations for PEFT adapters: merge, concat, save/load, freeze."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def get_adapter_state_dict(
    model: nn.Module,
    adapter_name: Optional[str] = None
) -> OrderedDict:
    """Extract adapter parameters (lora/adapter keys) into an OrderedDict."""
    adapter_state = OrderedDict()

    for name, param in model.named_parameters():
        if adapter_name is None:
            if 'lora' in name.lower() or 'adapter' in name.lower():
                adapter_state[name] = param.data.clone()
        else:
            if adapter_name in name:
                adapter_state[name] = param.data.clone()

    return adapter_state


def merge_adapters(
    adapter_states: List[OrderedDict],
    merge_weights: Optional[List[float]] = None,
    method: str = "average"
) -> OrderedDict:
    """Merge adapter state dicts by weighted average or task arithmetic."""
    if not adapter_states:
        raise ValueError("No adapter states provided")

    if len(adapter_states) == 1:
        return adapter_states[0]

    merged_state = OrderedDict()
    param_names = list(adapter_states[0].keys())

    for state in adapter_states:
        if set(state.keys()) != set(param_names):
            raise ValueError("Adapter states have different parameter names")

    if merge_weights is None:
        merge_weights = [1.0 / len(adapter_states)] * len(adapter_states)
    else:
        if len(merge_weights) != len(adapter_states):
            raise ValueError("Number of merge weights must match number of adapters")
        total = sum(merge_weights)
        merge_weights = [w / total for w in merge_weights]

    if method == "average" or method == "weighted":
        for param_name in param_names:
            merged_param = torch.zeros_like(adapter_states[0][param_name])
            for state, weight in zip(adapter_states, merge_weights):
                merged_param += weight * state[param_name]
            merged_state[param_name] = merged_param

    elif method == "task_arithmetic":
        base_state = adapter_states[0]
        for param_name in param_names:
            task_vector = torch.zeros_like(base_state[param_name])
            for state, weight in zip(adapter_states[1:], merge_weights[1:]):
                task_vector += weight * (state[param_name] - base_state[param_name])
            merged_state[param_name] = base_state[param_name] + task_vector

    else:
        raise ValueError(f"Unknown merge method: {method}")

    return merged_state


def concatenate_adapter_states(
    adapter_states: List[OrderedDict],
    dim: int = 0
) -> OrderedDict:
    """Concatenate adapter state dicts along a tensor dimension."""
    if not adapter_states:
        raise ValueError("No adapter states provided")

    concatenated_state = OrderedDict()
    param_names = list(adapter_states[0].keys())

    for param_name in param_names:
        params = [state[param_name] for state in adapter_states]
        concatenated_state[param_name] = torch.cat(params, dim=dim)

    return concatenated_state


def hierarchical_merge(
    adapter_states: List[OrderedDict],
    hierarchy: List[List[int]],
    merge_method: str = "average"
) -> OrderedDict:
    """Merge adapters hierarchically: group → merge within groups → merge groups."""
    current_states = adapter_states.copy()

    for level_groups in hierarchy:
        next_level_states = []
        for group_indices in level_groups:
            if isinstance(group_indices, int):
                group_indices = [group_indices]

            group_states = [current_states[i] for i in group_indices]
            merged = merge_adapters(group_states, method=merge_method)
            next_level_states.append(merged)

        current_states = next_level_states

    if len(current_states) > 1:
        return merge_adapters(current_states, method=merge_method)
    else:
        return current_states[0]


def compute_adapter_similarity(
    adapter_state_1: OrderedDict,
    adapter_state_2: OrderedDict,
    metric: str = "cosine"
) -> float:
    """Compute cosine/l1/l2 similarity between two adapter state dicts."""
    params_1 = torch.cat([p.flatten() for p in adapter_state_1.values()])
    params_2 = torch.cat([p.flatten() for p in adapter_state_2.values()])

    if metric == "cosine":
        similarity = F.cosine_similarity(
            params_1.unsqueeze(0),
            params_2.unsqueeze(0)
        ).item()
    elif metric == "l2":
        similarity = -torch.norm(params_1 - params_2, p=2).item()
    elif metric == "l1":
        similarity = -torch.norm(params_1 - params_2, p=1).item()
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    return similarity


def freeze_base_model(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze non-adapter (base model) parameters."""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower() and 'adapter' not in name.lower():
            param.requires_grad = not freeze


def freeze_adapters(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze adapter parameters."""
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            param.requires_grad = not freeze


def save_adapter_only(
    model: nn.Module,
    save_path: Union[str, Path],
    adapter_name: Optional[str] = None,
    save_metadata: bool = True
) -> None:
    """Save adapter parameters (and optional metadata) to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_state = get_adapter_state_dict(model, adapter_name)
    torch.save(adapter_state, save_path)

    if save_metadata:
        metadata = {
            "adapter_name": adapter_name,
            "num_parameters": sum(p.numel() for p in adapter_state.values()),
            "parameter_names": list(adapter_state.keys()),
        }

        if hasattr(model, 'peft_config'):
            if adapter_name and adapter_name in model.peft_config:
                metadata["peft_config"] = str(model.peft_config[adapter_name])

        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_adapter_weights(
    model: nn.Module,
    load_path: Union[str, Path],
    adapter_name: Optional[str] = None,
    strict: bool = False
) -> None:
    """Load adapter weights from disk; non-strict by default (shape-matched only)."""
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {load_path}")

    adapter_state = torch.load(load_path, map_location='cpu')

    if strict:
        model.load_state_dict(adapter_state, strict=True)
    else:
        model_dict = model.state_dict()
        filtered_state = {
            k: v for k, v in adapter_state.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_state)
        model.load_state_dict(model_dict, strict=False)


def copy_adapter(
    model: nn.Module,
    source_adapter: str,
    target_adapter: str
) -> None:
    """Copy weights from source_adapter to target_adapter within the same model."""
    if not hasattr(model, 'peft_config'):
        raise ValueError("Model does not appear to be a PEFT model")

    source_state = get_adapter_state_dict(model, source_adapter)

    if target_adapter not in model.peft_config:
        if hasattr(model, 'add_adapter'):
            model.add_adapter(target_adapter, model.peft_config[source_adapter])

    for name, param in model.named_parameters():
        if source_adapter in name:
            target_name = name.replace(source_adapter, target_adapter)
            if target_name in dict(model.named_parameters()):
                target_param = dict(model.named_parameters())[target_name]
                target_param.data.copy_(param.data)


def reset_adapter(
    model: nn.Module,
    adapter_name: Optional[str] = None,
    init_method: str = "default"
) -> None:
    """Re-initialize adapter weights (default: lora_A kaiming, lora_B zeros)."""
    for name, param in model.named_parameters():
        if adapter_name is None or adapter_name in name:
            if 'lora' in name.lower() or 'adapter' in name.lower():
                if init_method == "zeros":
                    nn.init.zeros_(param)
                elif init_method == "xavier":
                    nn.init.xavier_uniform_(param)
                elif init_method == "kaiming":
                    nn.init.kaiming_uniform_(param)
                elif init_method == "default":
                    if 'lora_A' in name or 'lora_a' in name:
                        nn.init.kaiming_uniform_(param)
                    elif 'lora_B' in name or 'lora_b' in name:
                        nn.init.zeros_(param)
                else:
                    raise ValueError(f"Unknown initialization method: {init_method}")

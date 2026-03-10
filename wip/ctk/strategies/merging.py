"""Adapter merging utilities: weighted average, task arithmetic, TIES, and DARE-TIES."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def merge_adapters_weighted(
    adapter_states: List[OrderedDict],
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True
) -> OrderedDict:
    """Merge adapters by weighted averaging of parameters."""
    if not adapter_states:
        raise ValueError("No adapter states provided")

    if len(adapter_states) == 1:
        return adapter_states[0]

    if weights is None:
        weights = [1.0 / len(adapter_states)] * len(adapter_states)

    if len(weights) != len(adapter_states):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of adapters ({len(adapter_states)})")

    if normalize_weights:
        total = sum(weights)
        weights = [w / total for w in weights]

    merged_state = OrderedDict()
    param_names = list(adapter_states[0].keys())

    for state in adapter_states:
        if set(state.keys()) != set(param_names):
            raise ValueError("All adapter states must have the same parameter names")

    for param_name in param_names:
        merged_param = torch.zeros_like(adapter_states[0][param_name])
        for state, weight in zip(adapter_states, weights):
            merged_param += weight * state[param_name]
        merged_state[param_name] = merged_param

    return merged_state


def task_arithmetic_merge(
    base_adapter: OrderedDict,
    task_adapters: List[OrderedDict],
    weights: Optional[List[float]] = None
) -> OrderedDict:
    """Merge: base + sum(weight_i * (adapter_i - base))."""
    if weights is None:
        weights = [1.0] * len(task_adapters)

    if len(weights) != len(task_adapters):
        raise ValueError("Number of weights must match number of task adapters")

    merged_state = OrderedDict()
    param_names = list(base_adapter.keys())

    for param_name in param_names:
        merged_param = base_adapter[param_name].clone()
        for task_adapter, weight in zip(task_adapters, weights):
            task_vector = task_adapter[param_name] - base_adapter[param_name]
            merged_param += weight * task_vector
        merged_state[param_name] = merged_param

    return merged_state


def ties_merging(
    adapter_states: List[OrderedDict],
    weights: Optional[List[float]] = None,
    density: float = 0.2,
    majority_sign_method: str = "total"
) -> OrderedDict:
    """TIES merging: trim top-k%, elect majority sign, merge consistent-sign parameters."""
    if not adapter_states:
        raise ValueError("No adapter states provided")

    if weights is None:
        weights = [1.0] * len(adapter_states)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    merged_state = OrderedDict()
    param_names = list(adapter_states[0].keys())

    for param_name in param_names:
        params = [state[param_name] for state in adapter_states]

        # Step 1: Trim
        trimmed_params = []
        for param in params:
            flat_param = param.flatten()
            k = max(1, int(density * flat_param.numel()))
            _, topk_indices = torch.topk(flat_param.abs(), k)
            mask = torch.zeros_like(flat_param, dtype=torch.bool)
            mask[topk_indices] = True
            mask = mask.view(param.shape)
            trimmed_params.append(param * mask)

        # Step 2: Elect sign
        stacked = torch.stack(trimmed_params)
        if majority_sign_method == "total":
            total_sum = (stacked * torch.tensor(weights).view(-1, *([1] * (stacked.dim() - 1))).to(stacked.device)).sum(dim=0)
            majority_sign = torch.sign(total_sum)
        elif majority_sign_method == "frequency":
            signs = torch.sign(stacked)
            majority_sign = torch.mode(signs, dim=0).values
        else:
            raise ValueError(f"Unknown majority sign method: {majority_sign_method}")

        # Step 3: Merge
        merged_param = torch.zeros_like(params[0])
        for trimmed, weight in zip(trimmed_params, weights):
            mask = (torch.sign(trimmed) == majority_sign) | (trimmed == 0)
            merged_param += weight * (trimmed * mask)

        merged_state[param_name] = merged_param

    return merged_state


def dare_ties_merging(
    adapter_states: List[OrderedDict],
    weights: Optional[List[float]] = None,
    density: float = 0.2,
    drop_rate: float = 0.9,
    rescale: bool = True
) -> OrderedDict:
    """DARE + TIES: randomly drop and rescale parameters, then apply TIES merging."""
    dare_states = []
    for state in adapter_states:
        dare_state = OrderedDict()
        for param_name, param in state.items():
            mask = torch.rand_like(param) > drop_rate
            dropped = param * mask
            if rescale:
                dropped = dropped * (1.0 / (1.0 - drop_rate))
            dare_state[param_name] = dropped
        dare_states.append(dare_state)

    return ties_merging(dare_states, weights=weights, density=density)


def hierarchical_merge(
    adapter_states: List[OrderedDict],
    hierarchy: List[List[int]],
    merge_method: str = "weighted",
    **merge_kwargs
) -> OrderedDict:
    """Merge adapters hierarchically according to specified groupings."""
    current_states = adapter_states.copy()

    for level_idx, level_groups in enumerate(hierarchy):
        next_level_states = []

        if isinstance(level_groups[0], (list, tuple)):
            groups = level_groups
        else:
            groups = [level_groups]

        for group_indices in groups:
            if isinstance(group_indices, int):
                group_indices = [group_indices]

            group_states = [current_states[i] for i in group_indices]

            if merge_method == "weighted":
                merged = merge_adapters_weighted(group_states, **merge_kwargs)
            elif merge_method == "task_arithmetic":
                merged = task_arithmetic_merge(
                    group_states[0],
                    group_states[1:] if len(group_states) > 1 else [],
                    **merge_kwargs
                )
            elif merge_method == "ties":
                merged = ties_merging(group_states, **merge_kwargs)
            else:
                raise ValueError(f"Unknown merge method: {merge_method}")

            next_level_states.append(merged)

        current_states = next_level_states

    if len(current_states) > 1:
        if merge_method == "weighted":
            return merge_adapters_weighted(current_states, **merge_kwargs)
        elif merge_method == "ties":
            return ties_merging(current_states, **merge_kwargs)
        else:
            return current_states[0]
    else:
        return current_states[0]


def get_merge_statistics(
    original_states: List[OrderedDict],
    merged_state: OrderedDict
) -> Dict[str, float]:
    """Compute sparsity and distance statistics for the merged adapter."""
    stats = {}

    total_params = 0
    nonzero_params = 0
    for param_name, param in merged_state.items():
        total_params += param.numel()
        nonzero_params += (param != 0).sum().item()

    stats['total_params'] = total_params
    stats['nonzero_params'] = nonzero_params
    stats['sparsity'] = 1 - (nonzero_params / total_params) if total_params > 0 else 0

    distances = []
    for original_state in original_states:
        dist = 0
        for param_name in merged_state.keys():
            if param_name in original_state:
                dist += torch.norm(merged_state[param_name] - original_state[param_name], p=2).item()
        distances.append(dist)

    stats['avg_distance_to_original'] = sum(distances) / len(distances) if distances else 0
    stats['min_distance_to_original'] = min(distances) if distances else 0
    stats['max_distance_to_original'] = max(distances) if distances else 0

    return stats


def apply_merged_state_to_model(
    model: nn.Module,
    merged_state: OrderedDict,
    strict: bool = False
) -> None:
    """Apply merged adapter state to a model, loading only matching parameters when strict=False."""
    if strict:
        model.load_state_dict(merged_state, strict=True)
    else:
        model_dict = model.state_dict()
        filtered_state = {
            k: v for k, v in merged_state.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_state)
        model.load_state_dict(model_dict, strict=False)
        logger.info(f"Loaded {len(filtered_state)}/{len(merged_state)} parameters from merged state")

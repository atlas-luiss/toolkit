"""Consolidated PEFT utilities: inspection, similarity, merge, concat, prune, save/load."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
from collections import OrderedDict
from copy import deepcopy
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

    # Check if model has peft_config (from HuggingFace PEFT)
    if hasattr(model, 'peft_config'):
        adapter_names = list(model.peft_config.keys())
    # Check for active_adapter attribute
    elif hasattr(model, 'active_adapter'):
        if isinstance(model.active_adapter, str):
            adapter_names = [model.active_adapter]
        elif isinstance(model.active_adapter, list):
            adapter_names = model.active_adapter
    # Fallback: search for LoRA layers in the model
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


def get_adapter_state_dict(
    model: nn.Module,
    adapter_name: Optional[str] = None
) -> OrderedDict:
    """Extract adapter parameters (lora/adapter keys) into an OrderedDict."""
    adapter_state = OrderedDict()

    for name, param in model.named_parameters():
        # Include parameter if it's an adapter parameter
        if adapter_name is None:
            if 'lora' in name.lower() or 'adapter' in name.lower():
                adapter_state[name] = param.data.clone()
        else:
            if adapter_name in name:
                adapter_state[name] = param.data.clone()

    return adapter_state


def count_nonzero_params(model: nn.Module) -> Dict[str, int]:
    """Count non-zero LoRA parameters; useful for measuring sparsity after pruning."""
    total_params = 0
    nonzero_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                # Handle both old and new PEFT versions
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


def calculate_adapter_similarity(
    adapter1: nn.Module,
    adapter2: nn.Module,
    method: str = "cosine",
    layer_filter: Optional[str] = None
) -> float:
    """Compute mean pairwise LoRA weight similarity between two adapters."""
    similarity_scores = []

    adapter1_modules = {
        name: m for name, m in adapter1.named_modules()
        if hasattr(m, 'lora_A')
    }
    adapter2_modules = {
        name: m for name, m in adapter2.named_modules()
        if hasattr(m, 'lora_A')
    }

    common_names = set(adapter1_modules.keys()) & set(adapter2_modules.keys())

    if layer_filter:
        common_names = [name for name in common_names if layer_filter in name]

    if not common_names:
        logger.warning("No common LoRA modules found between adapters")
        return 0.0

    for name in common_names:
        try:
            module1 = adapter1_modules[name]
            module2 = adapter2_modules[name]

            if isinstance(module1.lora_A, nn.ModuleDict):
                adapter_key1 = list(module1.lora_A.keys())[0]
                adapter_key2 = list(module2.lora_A.keys())[0]
                A1 = module1.lora_A[adapter_key1].weight.data.view(-1)
                A2 = module2.lora_A[adapter_key2].weight.data.view(-1)
                B1 = module1.lora_B[adapter_key1].weight.data.view(-1)
                B2 = module2.lora_B[adapter_key2].weight.data.view(-1)
            else:
                A1 = module1.lora_A.weight.data.view(-1)
                A2 = module2.lora_A.weight.data.view(-1)
                B1 = module1.lora_B.weight.data.view(-1)
                B2 = module2.lora_B.weight.data.view(-1)

            if method == "cosine":
                sim_A = F.cosine_similarity(A1.unsqueeze(0), A2.unsqueeze(0))
                sim_B = F.cosine_similarity(B1.unsqueeze(0), B2.unsqueeze(0))
                similarity_scores.append((sim_A + sim_B) / 2)

            elif method == "l2":
                dist_A = torch.norm(A1 - A2, p=2)
                dist_B = torch.norm(B1 - B2, p=2)
                similarity_scores.append(-((dist_A + dist_B) / 2))

            elif method == "l1":
                dist_A = torch.norm(A1 - A2, p=1)
                dist_B = torch.norm(B1 - B2, p=1)
                similarity_scores.append(-((dist_A + dist_B) / 2))

            else:
                raise ValueError(f"Unknown similarity method: {method}")

        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not process module {name}: {e}")
            continue

    if not similarity_scores:
        return 0.0

    return torch.mean(torch.stack(similarity_scores)).item()


def compute_pairwise_similarities(
    adapters: List[nn.Module],
    method: str = "cosine",
    layer_filter: Optional[str] = None
) -> torch.Tensor:
    """Return (n, n) pairwise similarity matrix for a list of adapters."""
    n_adapters = len(adapters)
    sim_matrix = torch.zeros(n_adapters, n_adapters)

    for i in range(n_adapters):
        for j in range(i, n_adapters):
            if i == j:
                sim_matrix[i, j] = 1.0 if method == "cosine" else 0.0
            else:
                sim = calculate_adapter_similarity(
                    adapters[i], adapters[j],
                    method=method,
                    layer_filter=layer_filter
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    return sim_matrix


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


def compare_adapters(
    adapter1: nn.Module,
    adapter2: nn.Module,
    verbose: bool = True
) -> Dict[str, Any]:
    """Return similarity and parameter count statistics for two adapters."""
    similarity = calculate_adapter_similarity(adapter1, adapter2, method="cosine")
    counts1 = count_nonzero_params(adapter1)
    counts2 = count_nonzero_params(adapter2)
    state1 = get_adapter_state_dict(adapter1)
    state2 = get_adapter_state_dict(adapter2)

    stats = {
        'similarity': similarity,
        'adapter1_params': counts1,
        'adapter2_params': counts2,
        'adapter1_state_keys': list(state1.keys()),
        'adapter2_state_keys': list(state2.keys()),
        'common_keys': list(set(state1.keys()) & set(state2.keys()))
    }

    if verbose:
        print(f"Adapter Comparison:")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print(f"  Adapter 1: {counts1['nonzero_params']:,} / {counts1['total_params']:,} params ({counts1['sparsity']:.2%} sparse)")
        print(f"  Adapter 2: {counts2['nonzero_params']:,} / {counts2['total_params']:,} params ({counts2['sparsity']:.2%} sparse)")
        print(f"  Common parameter keys: {len(stats['common_keys'])}")

    return stats


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


def concatenate_adapters(
    adapters: List[nn.Module],
    base_model: nn.Module,
    alphas: Optional[List[float]] = None,
    dim: int = 0,
    lora_config: Optional[Any] = None
) -> nn.Module:
    """Concatenate LoRA adapters along rank dim; result rank = sum of input ranks."""
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError("PEFT library required for adapter concatenation")

    if not adapters:
        raise ValueError("No adapters provided for concatenation")

    if alphas is None:
        alphas = [1.0] * len(adapters)

    if len(alphas) != len(adapters):
        raise ValueError("Number of alphas must match number of adapters")

    ranks = []
    for adapter in adapters:
        for name, module in adapter.named_modules():
            if hasattr(module, 'lora_A'):
                if isinstance(module.lora_A, nn.ModuleDict):
                    adapter_key = list(module.lora_A.keys())[0]
                    r = module.lora_A[adapter_key].weight.shape[0]
                else:
                    r = module.lora_A.weight.shape[0]
                ranks.append(r)
                break

    total_rank = sum(ranks)

    if lora_config is None:
        lora_config = LoraConfig(
            r=total_rank,
            lora_alpha=1,
            target_modules=["qkv"],
            lora_dropout=0.01,
            bias="none"
        )

    concat_adapter = get_peft_model(deepcopy(base_model), lora_config)

    with torch.no_grad():
        for name, module in concat_adapter.named_modules():
            if hasattr(module, 'lora_A'):
                A_list = []
                B_list = []

                for adapter, alpha in zip(adapters, alphas):
                    for adapter_name, adapter_module in adapter.named_modules():
                        if adapter_name == name and hasattr(adapter_module, 'lora_A'):
                            if isinstance(adapter_module.lora_A, nn.ModuleDict):
                                adapter_key = list(adapter_module.lora_A.keys())[0]
                                A_weight = adapter_module.lora_A[adapter_key].weight.data
                                B_weight = adapter_module.lora_B[adapter_key].weight.data
                            else:
                                A_weight = adapter_module.lora_A.weight.data
                                B_weight = adapter_module.lora_B.weight.data

                            A_list.append(A_weight)
                            B_list.append(B_weight * alpha)
                            break

                if A_list and B_list:
                    if isinstance(module.lora_A, nn.ModuleDict):
                        adapter_key = list(module.lora_A.keys())[0]
                        module.lora_A[adapter_key].weight.data = torch.cat(A_list, dim=dim)
                        module.lora_B[adapter_key].weight.data = torch.cat(B_list, dim=1 if dim == 0 else 0)
                    else:
                        module.lora_A.weight.data = torch.cat(A_list, dim=dim)
                        module.lora_B.weight.data = torch.cat(B_list, dim=1 if dim == 0 else 0)

    return concat_adapter


def prune_adapter_weights(
    adapter: nn.Module,
    k_ratio: float = 0.5,
    method: str = "magnitude"
) -> nn.Module:
    """Prune adapter weights, keeping top k_ratio fraction by magnitude or randomly."""
    pruned_adapter = deepcopy(adapter)

    with torch.no_grad():
        for name, module in pruned_adapter.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    if isinstance(module.lora_A, nn.ModuleDict):
                        adapter_key = list(module.lora_A.keys())[0]
                        A_weight = module.lora_A[adapter_key].weight.data
                        B_weight = module.lora_B[adapter_key].weight.data
                    else:
                        A_weight = module.lora_A.weight.data
                        B_weight = module.lora_B.weight.data
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Skipping module {name}: {e}")
                    continue

                if method == "magnitude":
                    A_flat = A_weight.flatten()
                    k = int(k_ratio * A_flat.numel())

                    if k > 0:
                        topk_values, _ = torch.topk(A_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = A_weight.abs() >= threshold
                        A_weight.mul_(mask)

                    B_flat = B_weight.flatten()
                    k = int(k_ratio * B_flat.numel())

                    if k > 0:
                        topk_values, _ = torch.topk(B_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = B_weight.abs() >= threshold
                        B_weight.mul_(mask)

                elif method == "random":
                    A_mask = torch.rand_like(A_weight) < k_ratio
                    B_mask = torch.rand_like(B_weight) < k_ratio
                    A_weight.mul_(A_mask)
                    B_weight.mul_(B_mask)

                else:
                    raise ValueError(f"Unknown pruning method: {method}")

    return pruned_adapter


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

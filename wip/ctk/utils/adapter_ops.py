"""Adapter operations at the nn.Module level: similarity, concatenation, pruning, comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


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
                # Get weight matrices - handle both old and new PEFT versions
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
                    # Prune LoRA A weights
                    A_flat = A_weight.flatten()
                    k = int(k_ratio * A_flat.numel())

                    if k > 0:
                        topk_values, _ = torch.topk(A_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = A_weight.abs() >= threshold
                        A_weight.mul_(mask)

                    # Prune LoRA B weights
                    B_flat = B_weight.flatten()
                    k = int(k_ratio * B_flat.numel())

                    if k > 0:
                        topk_values, _ = torch.topk(B_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = B_weight.abs() >= threshold
                        B_weight.mul_(mask)

                elif method == "random":
                    # Random pruning
                    A_mask = torch.rand_like(A_weight) < k_ratio
                    B_mask = torch.rand_like(B_weight) < k_ratio
                    A_weight.mul_(A_mask)
                    B_weight.mul_(B_mask)

                else:
                    raise ValueError(f"Unknown pruning method: {method}")

    return pruned_adapter


def compare_adapters(
    adapter1: nn.Module,
    adapter2: nn.Module,
    verbose: bool = True
) -> Dict[str, Any]:
    """Return similarity and parameter count statistics for two adapters."""
    from .inspection import count_nonzero_params
    from .state_ops import get_adapter_state_dict

    # Calculate similarity
    similarity = calculate_adapter_similarity(adapter1, adapter2, method="cosine")

    # Count parameters
    counts1 = count_nonzero_params(adapter1)
    counts2 = count_nonzero_params(adapter2)

    # Get state dicts
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

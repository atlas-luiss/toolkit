"""CTK utility functions: seeds, logging, adapter inspection, merging, and timing."""

# General utility functions (inline to avoid circular imports)
import os
import random
import logging
import numpy as np
import torch
from typing import Optional, Union


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the torch device to use for training."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device) if isinstance(device, str) else device


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

from .inspection import (
    check_trainable_params,
    get_adapter_names,
    print_parameter_count,
    get_lora_parameters,
    count_nonzero_params,
)

from .adapter_ops import (
    calculate_adapter_similarity,
    compute_pairwise_similarities,
    concatenate_adapters,
    prune_adapter_weights,
    compare_adapters,
)

from .state_ops import (
    get_adapter_state_dict,
    merge_adapters,
    concatenate_adapter_states,
    hierarchical_merge,
    compute_adapter_similarity,
    freeze_base_model,
    freeze_adapters,
    save_adapter_only,
    load_adapter_weights,
    copy_adapter,
    reset_adapter,
)

# LoRA merging utilities (PEFT model level)
from .lora_merge import (
    build_adapter_from_weights,
    concat_lora_adapters,
    merge_lora_groups,
)

# Timing utilities
from .timing import (
    Timer,
    TimingStats,
    format_time,
)

__all__ = [
    # General utilities
    'set_seed',
    'get_device',
    'setup_logging',

    # Inspection
    'check_trainable_params',
    'get_adapter_names',
    'print_parameter_count',
    'get_lora_parameters',
    'count_nonzero_params',

    # Adapter operations (module level)
    'calculate_adapter_similarity',
    'compute_pairwise_similarities',
    'concatenate_adapters',
    'prune_adapter_weights',
    'compare_adapters',

    # State operations (state dict level)
    'get_adapter_state_dict',
    'merge_adapters',
    'concatenate_adapter_states',
    'hierarchical_merge',
    'compute_adapter_similarity',
    'freeze_base_model',
    'freeze_adapters',
    'save_adapter_only',
    'load_adapter_weights',
    'copy_adapter',
    'reset_adapter',

    # LoRA merging (PEFT model level)
    'build_adapter_from_weights',
    'concat_lora_adapters',
    'merge_lora_groups',

    # Timing utilities
    'Timer',
    'TimingStats',
    'format_time',
]

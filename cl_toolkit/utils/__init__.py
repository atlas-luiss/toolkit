"""Small utilities: seeds, devices, logging."""

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

__all__ = ['set_seed', 'get_device', 'setup_logging']

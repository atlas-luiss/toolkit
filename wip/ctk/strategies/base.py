"""Base strategy classes for continual learning."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from abc import ABC, abstractmethod


class Strategy(ABC):
    """Base class for all continual learning strategies (new interface)."""

    def on_task_start(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        model: 'CLModel',
    ) -> None:
        """Called before training on task_id begins."""

    @abstractmethod
    def compute_loss(
        self,
        batch: tuple,
        model: 'CLModel',
        task_id: int,
    ) -> torch.Tensor:
        """Compute the training loss for one batch."""

    def before_update(
        self,
        model: 'CLModel',
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Called after loss.backward(), before optimizer.step()."""

    def on_task_end(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        model: 'CLModel',
    ) -> None:
        """Called after training on task_id ends."""

    def extra_params(self) -> List[nn.Parameter]:
        """Extra parameters this strategy needs in the optimizer."""
        return []

    def evaluate_task(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: 'CLModel',
        task_id: int,
    ) -> Optional[Dict]:
        """Custom evaluation; return None to use Trainer's default."""
        return None

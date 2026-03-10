"""Base strategy classes for continual learning."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
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


class ContinualLearningStrategy(ABC, nn.Module):
    """Legacy base class for continual learning strategies."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.current_task    = 0
        self.num_seen_tasks  = 0
        self.task_info: Dict = {}

    @abstractmethod
    def train_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        task_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate_task(
        self,
        test_loader: torch.utils.data.DataLoader,
        task_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    def before_train_task(self, task_id: int, **kwargs) -> None:
        self.current_task = task_id

    def after_train_task(self, task_id: int, **kwargs) -> None:
        self.num_seen_tasks = max(self.num_seen_tasks, task_id + 1)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task
        return self.model(x)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str, task_id: int) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'current_task':     self.current_task,
            'num_seen_tasks':   self.num_seen_tasks,
            'task_info':        self.task_info,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_task   = checkpoint['current_task']
        self.num_seen_tasks = checkpoint['num_seen_tasks']
        self.task_info      = checkpoint['task_info']


class ReplayStrategy(ContinualLearningStrategy):
    """Legacy base class for replay-based strategies."""

    def __init__(
        self,
        model:       nn.Module,
        memory_size: int                    = 500,
        device:      Optional[torch.device] = None
    ):
        super().__init__(model, device)
        self.memory_size   = memory_size
        self.memory_buffer = {'inputs': [], 'targets': [], 'task_ids': []}

    def update_memory(
        self,
        inputs:             torch.Tensor,
        targets:            torch.Tensor,
        task_id:            int,
        selection_strategy: str = "random"
    ) -> None:
        for i in range(inputs.size(0)):
            if len(self.memory_buffer['inputs']) < self.memory_size:
                self.memory_buffer['inputs'].append(inputs[i].cpu())
                self.memory_buffer['targets'].append(targets[i].cpu())
                self.memory_buffer['task_ids'].append(task_id)
            else:
                idx = torch.randint(
                    0, len(self.memory_buffer['inputs']) + i + 1, (1,)
                ).item()
                if idx < self.memory_size:
                    self.memory_buffer['inputs'][idx]   = inputs[i].cpu()
                    self.memory_buffer['targets'][idx]  = targets[i].cpu()
                    self.memory_buffer['task_ids'][idx] = task_id

    def get_memory_batch(self, batch_size: int) -> tuple:
        if len(self.memory_buffer['inputs']) == 0:
            return None, None, None
        indices = torch.randint(
            0,
            len(self.memory_buffer['inputs']),
            (min(batch_size, len(self.memory_buffer['inputs'])),),
        )
        inputs   = torch.stack([self.memory_buffer['inputs'][i]  for i in indices]).to(self.device)
        targets  = torch.tensor([self.memory_buffer['targets'][i] for i in indices]).to(self.device)
        task_ids = [self.memory_buffer['task_ids'][i] for i in indices]
        return inputs, targets, task_ids


class RegularizationStrategy(ContinualLearningStrategy):
    """Legacy base class for regularization-based strategies."""

    def __init__(
        self,
        model:                   nn.Module,
        regularization_strength: float                  = 1000.0,
        device:                  Optional[torch.device] = None
    ):
        super().__init__(model, device)
        self.regularization_strength = regularization_strength
        self.importance_weights: Dict[str, torch.Tensor] = {}
        self.previous_params:    Dict[str, torch.Tensor] = {}

    def compute_importance(
        self,
        train_loader: torch.utils.data.DataLoader,
        task_id: int,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement compute_importance()")

    def compute_regularization_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.importance_weights and name in self.previous_params:
                importance = self.importance_weights[name]
                prev_param = self.previous_params[name]
                loss = loss + (importance * (param - prev_param).pow(2)).sum()
        return self.regularization_strength * loss

    def store_parameters(self) -> None:
        self.previous_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

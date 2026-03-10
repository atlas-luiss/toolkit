"""Base classes for CL benchmarks: TaskSplit and CLBenchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


@dataclass
class TaskSplit:
    """Represents one task in a continual learning scenario."""
    task_id: int
    classes: List[int]       # global class indices for this task
    train_dataset: Dataset
    test_dataset: Dataset


class CLBenchmark(ABC):
    """Abstract base class producing a sequence of disjoint-class TaskSplits."""

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        """Total number of tasks in this benchmark."""
        ...

    @property
    @abstractmethod
    def total_classes(self) -> int:
        """Total number of classes across all tasks."""
        ...

    @property
    @abstractmethod
    def classes_per_task(self) -> int:
        """Number of classes per task (assumes uniform split)."""
        ...

    @abstractmethod
    def get_splits(self) -> List[TaskSplit]:
        """Return one TaskSplit per task, in task order."""
        ...

    def class_name(self, class_id: int) -> str:
        """Human-readable name for a global class index; override for dataset-specific names."""
        return str(class_id)

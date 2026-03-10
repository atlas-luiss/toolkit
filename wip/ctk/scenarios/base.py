"""Base scenario classes: TaskInfo and CLScenario."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

from torch.utils.data import DataLoader

from ctk.benchmarks.base import CLBenchmark, TaskSplit
from ctk.models.base import CLModel


@dataclass
class TaskInfo:
    """All information the Trainer needs for one task."""

    task_id: int
    classes: List[int]          # global class indices for this task
    train_loader: DataLoader
    test_loader: DataLoader

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_offset(self) -> int:
        """Lowest global class index in this task."""
        return min(self.classes)


class CLScenario(ABC):
    """Wraps a CLBenchmark, creates DataLoaders per task, and configures the model head."""

    def __init__(
        self,
        benchmark: CLBenchmark,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._task_infos: Optional[List[TaskInfo]] = None

    @abstractmethod
    def setup(self, model: CLModel) -> None:
        """Configure model output head for this scenario."""

    def _build_loaders(self) -> List[TaskInfo]:
        """Build a TaskInfo per benchmark split with global labels; override to remap."""
        task_infos: List[TaskInfo] = []
        for split in self.benchmark.get_splits():
            train_loader = DataLoader(
                split.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            test_loader = DataLoader(
                split.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            task_infos.append(
                TaskInfo(
                    task_id=split.task_id,
                    classes=list(split.classes),
                    train_loader=train_loader,
                    test_loader=test_loader,
                )
            )
        return task_infos

    # ------------------------------------------------------------------
    # Iteration interface used by Trainer
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TaskInfo]:
        if self._task_infos is None:
            self._task_infos = self._build_loaders()
        return iter(self._task_infos)

    def __len__(self) -> int:
        return self.benchmark.num_tasks

    def get_task(self, task_id: int) -> TaskInfo:
        if self._task_infos is None:
            self._task_infos = self._build_loaders()
        return self._task_infos[task_id]

    @property
    def num_tasks(self) -> int:
        return self.benchmark.num_tasks

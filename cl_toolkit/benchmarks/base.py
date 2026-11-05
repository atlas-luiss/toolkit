"""Benchmark = an ordered list of TaskSplits. That is the whole contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class TaskSplit:
    """One task: its global class ids and its train/test datasets.

    Datasets must yield ``(x, y)`` with ``y`` a *global* class id.
    """
    task_id: int
    classes: List[int]
    train_dataset: Dataset
    test_dataset: Dataset


class CLBenchmark:
    """A sequence of tasks.

    You rarely subclass this: build one with :func:`cl_toolkit.benchmarks.split_by_class`,
    :func:`cl_toolkit.benchmarks.split_by_domain`, or pass your own ``splits``.

    Subclasses only need to implement :meth:`get_splits` (plus ``total_classes``
    if labels are not ``0..max``).
    """

    def __init__(
        self,
        splits: Optional[List[TaskSplit]] = None,
        total_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        name: str = "custom",
    ):
        self._splits = splits
        self._total_classes = total_classes
        self._class_names = class_names or {}
        self.name = name

    def get_splits(self) -> List[TaskSplit]:
        if self._splits is None:
            raise NotImplementedError("Pass splits= or override get_splits()")
        return self._splits

    @property
    def num_tasks(self) -> int:
        return len(self.get_splits())

    @property
    def total_classes(self) -> int:
        if self._total_classes is not None:
            return self._total_classes
        return max(max(s.classes) for s in self.get_splits()) + 1

    @property
    def classes_per_task(self) -> int:
        return len(self.get_splits()[0].classes)

    def class_name(self, class_id: int) -> str:
        return self._class_names.get(class_id, str(class_id))

    def map_splits(self, fn: Callable[[TaskSplit], TaskSplit], name: Optional[str] = None) -> "CLBenchmark":
        """Return a new benchmark with ``fn`` applied to every split."""
        return CLBenchmark([fn(s) for s in self.get_splits()], self._total_classes,
                           self._class_names, name or self.name)

    def __repr__(self) -> str:
        sizes = [len(s.train_dataset) for s in self.get_splits()]
        return (f"CLBenchmark({self.name!r}, tasks={self.num_tasks}, "
                f"classes={self.total_classes}, train_sizes={sizes})")

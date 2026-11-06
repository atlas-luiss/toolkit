"""One generic Scenario: how a benchmark is presented to the learner.

``kind`` controls three things, and nothing else:

============  =================  ============  =========================
kind          head               labels        task id at evaluation
============  =================  ============  =========================
``"class"``   one shared head    global        no  (Class-IL)
``"task"``    one head per task  task-local    yes (Task-IL)
``"domain"``  one shared head    global        no  (Domain-IL)
============  =================  ============  =========================

In shared-head scenarios, logits are restricted to classes seen so far
(standard Class-IL practice), so unseen-class neurons cannot win argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from cl_toolkit.benchmarks.base import CLBenchmark

KINDS = ("class", "task", "domain")


@dataclass
class TaskInfo:
    task_id: int
    classes: List[int]          # global class ids in this task
    train_loader: DataLoader
    test_loader: DataLoader

    @property
    def num_classes(self) -> int:
        return len(self.classes)


class _LocalLabels(Dataset):
    """Maps global class ids to 0..C-1 within a task (Task-IL)."""

    def __init__(self, ds: Dataset, classes: List[int]):
        self.ds, self.map = ds, {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i][:2]
        return x, self.map[int(y)]


def _find_transform(ds):
    from torch.utils.data import ConcatDataset, Subset
    while True:
        if isinstance(ds, Subset):
            ds = ds.dataset
        elif isinstance(ds, ConcatDataset):
            ds = ds.datasets[0]
        elif isinstance(ds, _LocalLabels):
            ds = ds.ds
        else:
            return getattr(ds, "transform", None)


def _with_transform(ds, tf):
    import copy
    from torch.utils.data import ConcatDataset, Subset
    if isinstance(ds, Subset):
        return Subset(_with_transform(ds.dataset, tf), ds.indices)
    if isinstance(ds, ConcatDataset):
        return ConcatDataset([_with_transform(d, tf) for d in ds.datasets])
    if hasattr(ds, "transform"):
        ds = copy.copy(ds)
        ds.transform = tf
    return ds


class Scenario:
    def __init__(
        self,
        benchmark: CLBenchmark,
        kind: str = "class",
        batch_size: int = 64,
        eval_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        drop_last: bool = False,
        head="linear",
    ):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.benchmark = benchmark
        self.kind = kind
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        # default: 4, or 0 for in-memory benchmarks (e.g. toy), where workers only add overhead
        self.num_workers = num_workers if num_workers is not None else getattr(benchmark, "default_num_workers", 4)
        # None = decided by the trainer from the run's device (pinned only for CUDA runs)
        self._pin_auto = pin_memory is None
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.drop_last = drop_last
        self.head = head          # classifier layer: "linear" | "cosine" | callable(in, out, device=)
        self._tasks: Optional[List[TaskInfo]] = None

    # -- what the learner is allowed to know --------------------------------
    @property
    def task_aware(self) -> bool:
        """True iff the task id may be used at test time."""
        return self.kind == "task"

    @property
    def shared_head(self) -> bool:
        return self.kind != "task"

    @property
    def num_tasks(self) -> int:
        return self.benchmark.num_tasks

    def __len__(self):
        return self.num_tasks

    # -- model setup ---------------------------------------------------------
    def setup(self, model) -> None:
        from cl_toolkit.models.base import IncrementalClassifier, MultiHeadClassifier
        if self.shared_head:
            model.set_head(IncrementalClassifier(model.embed_dim, self.benchmark.total_classes, self.head))
        else:
            model.set_head(MultiHeadClassifier(model.embed_dim, self.head))
            for s in self.benchmark.get_splits():
                model.add_task_head(s.task_id, len(s.classes))

    def seen_classes(self, task_id: int) -> List[int]:
        """Output units the model may predict after training on ``task_id``."""
        if not self.shared_head:
            return list(range(len(self.get_task(task_id).classes)))
        return sorted({c for t in self.tasks[: task_id + 1] for c in t.classes})

    # -- data ----------------------------------------------------------------
    def use_device(self, device) -> None:
        """Pin host memory only when training on CUDA (unless ``pin_memory`` was given)."""
        if not self._pin_auto:
            return
        self.pin_memory = torch.device(device).type == "cuda"
        for t in self._tasks or []:
            for dl in (t.train_loader, t.test_loader):
                dl.pin_memory = self.pin_memory

    def wrap_dataset(self, ds: Dataset, classes: List[int]) -> Dataset:
        """Label transform applied to every task dataset (and to replay data)."""
        return ds if self.shared_head else _LocalLabels(ds, classes)

    def _loader(self, ds, train: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size if train else self.eval_batch_size,
                          shuffle=train, num_workers=self.num_workers, pin_memory=self.pin_memory,
                          drop_last=self.drop_last and train)

    def _build(self) -> List[TaskInfo]:
        out = []
        for s in self.benchmark.get_splits():
            out.append(TaskInfo(
                s.task_id, list(s.classes),
                self._loader(self.wrap_dataset(s.train_dataset, s.classes), True),
                self._loader(self.wrap_dataset(s.test_dataset, s.classes), False),
            ))
        return out

    def feature_loader(self, task_id: int, shuffle: bool = False, global_labels: bool = False) -> DataLoader:
        """Task ``task_id``'s *training* data with the *test* transform (no augmentation).

        Used by prototype / closed-form methods (SimpleCIL, RanPAC, iCaRL herding).
        Falls back to the training transform if it cannot be swapped.
        """
        s = self.benchmark.get_splits()[task_id]
        tf = _find_transform(s.test_dataset)
        ds = _with_transform(s.train_dataset, tf) if tf is not None else s.train_dataset
        ds = ds if global_labels else self.wrap_dataset(ds, s.classes)
        return DataLoader(ds, batch_size=self.eval_batch_size,
                          shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    @property
    def tasks(self) -> List[TaskInfo]:
        if self._tasks is None:
            self._tasks = self._build()
        return self._tasks

    def get_task(self, task_id: int) -> TaskInfo:
        return self.tasks[task_id]

    def __iter__(self) -> Iterator[TaskInfo]:
        return iter(self.tasks)

    def __repr__(self):
        return f"Scenario(kind={self.kind!r}, {self.benchmark!r})"


# Convenience aliases — same class, preset kind.
def ClassIncremental(benchmark, **kw):  return Scenario(benchmark, "class", **kw)
def TaskIncremental(benchmark, **kw):   return Scenario(benchmark, "task", **kw)
def DomainIncremental(benchmark, **kw): return Scenario(benchmark, "domain", **kw)

CLScenario = Scenario  # backwards-compatible name

__all__ = ["Scenario", "TaskInfo", "ClassIncremental", "TaskIncremental", "DomainIncremental",
           "CLScenario", "KINDS"]

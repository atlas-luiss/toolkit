"""CustomStream — build a CL benchmark from any PyTorch dataset via class or domain splits."""

from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset, Subset

from .base import CLBenchmark, TaskSplit
from .split import make_class_splits


class CustomStream(CLBenchmark):
    """CL benchmark built from any PyTorch dataset with explicit task-to-classes mapping."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        task_to_classes: Dict[int, List[int]],
    ):
        """Build Subset-based TaskSplits from the task_to_classes mapping."""
        self._train_dataset   = train_dataset
        self._test_dataset    = test_dataset
        self._task_to_classes = task_to_classes

        train_class_to_indices: Dict[int, List[int]] = {}
        for idx, target in enumerate(train_dataset.targets):
            c = int(target)
            train_class_to_indices.setdefault(c, []).append(idx)

        test_class_to_indices: Dict[int, List[int]] = {}
        for idx, target in enumerate(test_dataset.targets):
            c = int(target)
            test_class_to_indices.setdefault(c, []).append(idx)

        self._splits: List[TaskSplit] = []
        for task_id in sorted(task_to_classes.keys()):
            classes = task_to_classes[task_id]

            train_indices: List[int] = []
            for c in classes:
                train_indices.extend(train_class_to_indices.get(c, []))

            test_indices: List[int] = []
            for c in classes:
                test_indices.extend(test_class_to_indices.get(c, []))

            self._splits.append(TaskSplit(
                task_id=task_id,
                classes=list(classes),
                train_dataset=Subset(train_dataset, train_indices),
                test_dataset=Subset(test_dataset, test_indices),
            ))

        all_classes = sorted(
            {c for classes in task_to_classes.values() for c in classes}
        )
        self._total_classes = len(all_classes)

        task_sizes = [len(v) for v in task_to_classes.values()]
        self._classes_per_task = task_sizes[0] if len(set(task_sizes)) == 1 else -1

    # ------------------------------------------------------------------
    # CLBenchmark interface
    # ------------------------------------------------------------------

    @property
    def num_tasks(self) -> int:
        return len(self._splits)

    @property
    def total_classes(self) -> int:
        return self._total_classes

    @property
    def classes_per_task(self) -> int:
        """Uniform number of classes per task, or -1 if tasks differ in size."""
        return self._classes_per_task

    def get_splits(self) -> List[TaskSplit]:
        return self._splits

    # ------------------------------------------------------------------
    # Factory: class-based split
    # ------------------------------------------------------------------

    @classmethod
    def from_class_split(
        cls,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_tasks: int,
        classes_per_task: Optional[int] = None,
        class_order: Optional[List[int]] = None,
        seed: int = 42,
    ) -> 'CustomStream':
        """Factory: split classes evenly into num_tasks tasks."""
        splits = make_class_splits(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            class_order=class_order,
            seed=seed,
        )

        task_to_classes = {s.task_id: s.classes for s in splits}

        instance = cls.__new__(cls)
        instance._train_dataset   = train_dataset
        instance._test_dataset    = test_dataset
        instance._task_to_classes = task_to_classes
        instance._splits          = splits

        all_classes = sorted({c for cs in task_to_classes.values() for c in cs})
        instance._total_classes = len(all_classes)

        task_sizes = [len(v) for v in task_to_classes.values()]
        instance._classes_per_task = (
            task_sizes[0] if len(set(task_sizes)) == 1 else -1
        )
        return instance

    # ------------------------------------------------------------------
    # Factory: domain-based split
    # ------------------------------------------------------------------

    @classmethod
    def from_domain_split(
        cls,
        task_datasets: List[Tuple[Dataset, Dataset]],
        shared_classes: Optional[List[int]] = None,
    ) -> 'CustomStream':
        """
        Factory: each (train_ds, test_ds) pair is one domain task.

        Classes are inferred from .targets unless shared_classes is given.
        TaskSplits point directly to the per-domain datasets (not Subsets).
        """
        from torch.utils.data import ConcatDataset

        all_train: List[Dataset] = []
        all_test:  List[Dataset] = []
        splits:    List[TaskSplit] = []
        task_to_classes: Dict[int, List[int]] = {}

        cumulative_train_offset = 0
        cumulative_test_offset  = 0

        for task_id, (train_ds, test_ds) in enumerate(task_datasets):
            if shared_classes is not None:
                task_classes = list(shared_classes)
            else:
                task_classes = sorted(set(int(t) for t in train_ds.targets))

            task_to_classes[task_id] = task_classes
            all_train.append(train_ds)
            all_test.append(test_ds)

            splits.append(TaskSplit(
                task_id=task_id,
                classes=task_classes,
                train_dataset=train_ds,
                test_dataset=test_ds,
            ))

        combined_train = ConcatDataset(all_train)
        combined_test  = ConcatDataset(all_test)

        instance = cls.__new__(cls)
        instance._train_dataset   = combined_train
        instance._test_dataset    = combined_test
        instance._task_to_classes = task_to_classes
        instance._splits          = splits

        all_classes = sorted({c for cs in task_to_classes.values() for c in cs})
        instance._total_classes = len(all_classes)

        task_sizes = [len(v) for v in task_to_classes.values()]
        instance._classes_per_task = (
            task_sizes[0] if len(set(task_sizes)) == 1 else -1
        )
        return instance

"""Task-Incremental Learning scenario: per-task heads, task identity known at inference."""

from __future__ import annotations

from typing import List

from torch.utils.data import DataLoader, Dataset

from ctk.models.base import CLModel

from .base import CLScenario, TaskInfo


class _LabelRemap(Dataset):
    """Remaps global class IDs to task-local (0-based) indices.

    Uses an explicit dict mapping so non-consecutive and shuffled class
    orderings are handled correctly.
    """

    def __init__(self, dataset: Dataset, class_map: dict) -> None:
        self.dataset   = dataset
        self.class_map = class_map

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x, self.class_map[int(y)]


class TaskIncremental(CLScenario):
    """Task-IL: MultiHeadClassifier with local (0-based) labels per task."""

    def setup(self, model: CLModel) -> None:
        """
        Attach a MultiHeadClassifier with one head pre-built per task.
        """
        from ctk.models.base import MultiHeadClassifier

        head = MultiHeadClassifier(model.embed_dim)
        model.set_head(head)

        for split in self.benchmark.get_splits():
            model.add_task_head(split.task_id, len(split.classes))

    def _build_loaders(self) -> List[TaskInfo]:
        """Like the base but wraps each dataset with _LabelRemap for local labels."""
        task_infos: List[TaskInfo] = []
        for split in self.benchmark.get_splits():
            # Map each original class ID to its local 0-based index within the task.
            # Works correctly for consecutive, non-consecutive, and shuffled class orderings.
            class_map = {cls: local for local, cls in enumerate(split.classes)}

            remapped_train = _LabelRemap(split.train_dataset, class_map)
            remapped_test  = _LabelRemap(split.test_dataset,  class_map)

            train_loader = DataLoader(
                remapped_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            test_loader = DataLoader(
                remapped_test,
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

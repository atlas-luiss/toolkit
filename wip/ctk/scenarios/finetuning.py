"""Finetuning (lower bound) and JointTraining (upper bound) scenarios."""

from __future__ import annotations

from typing import List

from torch.utils.data import ConcatDataset, DataLoader

from ctk.models.base import CLModel

from .base import CLScenario, TaskInfo


class Finetuning(CLScenario):
    """Sequential fine-tuning without forgetting prevention; same head layout as ClassIncremental."""

    def setup(self, model: CLModel) -> None:
        """Attach an IncrementalClassifier pre-grown to cover all benchmark classes."""
        from ctk.models.base import IncrementalClassifier

        head = IncrementalClassifier(model.embed_dim, initial_classes=0)
        model.set_head(head)
        model.grow_head(self.benchmark.total_classes)


class JointTraining(CLScenario):
    """All task data concatenated into a single training pass (offline upper bound)."""

    def setup(self, model: CLModel) -> None:
        """
        Attach a fixed IncrementalClassifier covering all classes at once.
        """
        from ctk.models.base import IncrementalClassifier

        head = IncrementalClassifier(
            model.embed_dim,
            initial_classes=0,
        )
        model.set_head(head)
        model.grow_head(self.benchmark.total_classes)

    def _build_loaders(self) -> List[TaskInfo]:
        """Concatenate all task datasets into a single TaskInfo (task_id=0)."""
        splits = self.benchmark.get_splits()

        all_train = ConcatDataset([s.train_dataset for s in splits])
        all_test  = ConcatDataset([s.test_dataset  for s in splits])

        all_classes: List[int] = []
        for split in splits:
            all_classes.extend(split.classes)

        train_loader = DataLoader(
            all_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = DataLoader(
            all_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return [
            TaskInfo(
                task_id=0,
                classes=all_classes,
                train_loader=train_loader,
                test_loader=test_loader,
            )
        ]

    @property
    def num_tasks(self) -> int:
        return 1

    def __len__(self) -> int:
        return 1

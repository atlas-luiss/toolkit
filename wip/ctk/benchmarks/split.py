"""Generic class-split utility: partitions a dataset by class into TaskSplits."""

import random
from typing import List, Optional

from torch.utils.data import Subset

from .base import TaskSplit


def make_class_splits(
    train_dataset,
    test_dataset,
    num_tasks: int,
    classes_per_task: Optional[int] = None,
    class_order: Optional[List[int]] = None,
    shuffle_classes: bool = True,
    seed: int = 42,
) -> List[TaskSplit]:
    """Split train/test datasets into num_tasks TaskSplits by class.

    shuffle_classes=True randomises the class-to-task assignment (reproducible via seed).
    Pass class_order to use a fully custom ordering (overrides shuffle_classes).
    """
    all_classes = sorted(set(int(t) for t in train_dataset.targets))

    if class_order is not None:
        unknown = set(class_order) - set(all_classes)
        if unknown:
            raise ValueError(
                f"class_order contains class indices not found in "
                f"train_dataset.targets: {sorted(unknown)}"
            )
        ordered_classes = list(class_order)
    elif shuffle_classes:
        rng = random.Random(seed)
        ordered_classes = all_classes[:]
        rng.shuffle(ordered_classes)
    else:
        ordered_classes = all_classes

    total_classes = len(ordered_classes)

    if classes_per_task is None:
        classes_per_task = total_classes // num_tasks

    if classes_per_task * num_tasks > total_classes:
        raise ValueError(
            f"classes_per_task ({classes_per_task}) * num_tasks ({num_tasks}) = "
            f"{classes_per_task * num_tasks} exceeds total available classes "
            f"({total_classes})."
        )

    train_class_to_indices: dict = {}
    for idx, target in enumerate(train_dataset.targets):
        c = int(target)
        train_class_to_indices.setdefault(c, []).append(idx)

    test_class_to_indices: dict = {}
    for idx, target in enumerate(test_dataset.targets):
        c = int(target)
        test_class_to_indices.setdefault(c, []).append(idx)

    splits: List[TaskSplit] = []

    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        end = start + classes_per_task
        task_classes = ordered_classes[start:end]

        train_indices = []
        for c in task_classes:
            train_indices.extend(train_class_to_indices.get(c, []))

        test_indices = []
        for c in task_classes:
            test_indices.extend(test_class_to_indices.get(c, []))

        splits.append(TaskSplit(
            task_id=task_id,
            classes=task_classes,
            train_dataset=Subset(train_dataset, train_indices),
            test_dataset=Subset(test_dataset, test_indices),
        ))

    return splits

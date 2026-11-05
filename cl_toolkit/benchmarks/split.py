"""Turn any dataset(s) into a CL benchmark."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from torch.utils.data import ConcatDataset, Dataset, Subset

from .base import CLBenchmark, TaskSplit


def get_targets(dataset: Dataset) -> List[int]:
    """Labels of a dataset. Uses ``.targets`` / ``.labels`` when present, else iterates."""
    if isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    for attr in ("targets", "labels", "_labels", "y"):
        t = getattr(dataset, attr, None)
        if t is not None:
            return [int(v) for v in (t.tolist() if hasattr(t, "tolist") else t)]
    if hasattr(dataset, "tensors"):  # TensorDataset
        return [int(v) for v in dataset.tensors[1].tolist()]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def _by_class(targets: Sequence[int]):
    idx = defaultdict(list)
    for i, t in enumerate(targets):
        idx[t].append(i)
    return idx


def split_by_class(
    train: Dataset,
    test: Dataset,
    num_tasks: int,
    class_order: Optional[List[int]] = None,
    shuffle_classes: bool = True,
    seed: int = 42,
    initial_classes: Optional[int] = None,
    class_names=None,
    name: str = "custom",
) -> CLBenchmark:
    """Class-incremental stream: classes are partitioned into ``num_tasks`` groups.

    Args:
        class_order: explicit class ordering (overrides ``shuffle_classes``).
        initial_classes: size of the first task (e.g. 50 for the "B50" protocol);
            the remaining classes are split evenly over ``num_tasks - 1`` tasks.
    """
    tr_t, te_t = get_targets(train), get_targets(test)
    classes = sorted(set(tr_t))
    if class_order is not None:
        if set(class_order) - set(classes):
            raise ValueError(f"class_order has unknown classes: {sorted(set(class_order) - set(classes))}")
        classes = list(class_order)
    elif shuffle_classes:
        random.Random(seed).shuffle(classes)

    if initial_classes:
        rest = len(classes) - initial_classes
        per = rest // max(num_tasks - 1, 1)
        sizes = [initial_classes] + [per] * (num_tasks - 1)
    else:
        per = len(classes) // num_tasks
        sizes = [per] * num_tasks
    if per == 0:
        raise ValueError(f"{len(classes)} classes cannot be split into {num_tasks} tasks")

    tr_idx, te_idx = _by_class(tr_t), _by_class(te_t)
    splits, start = [], 0
    for t, n in enumerate(sizes):
        cls = classes[start:start + n]
        start += n
        splits.append(TaskSplit(
            t, cls,
            Subset(train, [i for c in cls for i in tr_idx[c]]),
            Subset(test, [i for c in cls for i in te_idx[c]]),
        ))
    return CLBenchmark(splits, total_classes=max(tr_t) + 1, class_names=class_names, name=name)


def split_by_domain(
    domains: Sequence[Tuple[Dataset, Dataset]],
    num_classes: Optional[int] = None,
    name: str = "custom-domains",
) -> CLBenchmark:
    """Domain-incremental stream: one task per ``(train, test)`` pair, shared label space."""
    splits = []
    for t, (tr, te) in enumerate(domains):
        cls = sorted(set(get_targets(tr)))
        splits.append(TaskSplit(t, cls, tr, te))
    n = num_classes or max(max(s.classes) for s in splits) + 1
    return CLBenchmark(splits, total_classes=n, name=name)


def few_shot(benchmark: CLBenchmark, k: int, skip_first: bool = False, seed: int = 42) -> CLBenchmark:
    """Keep at most ``k`` training samples per class.

    ``skip_first=True`` keeps task 0 full: the standard FSCIL protocol.
    Test sets are untouched.
    """
    def fn(split: TaskSplit) -> TaskSplit:
        if skip_first and split.task_id == 0:
            return split
        rng = random.Random(seed + split.task_id)
        idx = _by_class(get_targets(split.train_dataset))
        keep = []
        for c in split.classes:
            ids = idx.get(c, [])[:]
            rng.shuffle(ids)
            keep += ids[:k]
        return TaskSplit(split.task_id, split.classes, Subset(split.train_dataset, keep), split.test_dataset)
    return benchmark.map_splits(fn, name=f"{benchmark.name}-{k}shot")


def joint(benchmark: CLBenchmark) -> CLBenchmark:
    """Collapse all tasks into one: the offline upper bound."""
    s = benchmark.get_splits()
    merged = TaskSplit(0, sorted({c for x in s for c in x.classes}),
                       ConcatDataset([x.train_dataset for x in s]),
                       ConcatDataset([x.test_dataset for x in s]))
    return CLBenchmark([merged], benchmark.total_classes, benchmark._class_names, f"{benchmark.name}-joint")


# Backwards-compatible name
def make_class_splits(train_dataset, test_dataset, num_tasks, classes_per_task=None,
                      class_order=None, shuffle_classes=True, seed=42) -> List[TaskSplit]:
    return split_by_class(train_dataset, test_dataset, num_tasks, class_order,
                          shuffle_classes, seed).get_splits()

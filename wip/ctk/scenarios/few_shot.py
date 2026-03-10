"""FewShot scenario — limits each task's training set to k_shots examples per class."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional, Type

from torch.utils.data import DataLoader, Subset

from ctk.benchmarks.base import CLBenchmark, TaskSplit
from ctk.models.base import CLModel
from ctk.scenarios.base import CLScenario, TaskInfo
from ctk.scenarios.class_incremental import ClassIncremental


def _k_shot_subset(dataset, classes: List[int], k: int, seed: int = 42) -> Subset:
    """Return a Subset with at most k examples per class."""
    rng = random.Random(seed)
    class_set = set(classes)

    if hasattr(dataset, "targets"):
        raw = dataset.targets
        targets = [int(t.item() if hasattr(t, "item") else t) for t in raw]
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "targets"):
        raw = dataset.dataset.targets
        targets = [
            int(raw[i].item() if hasattr(raw[i], "item") else raw[i])
            for i in dataset.indices
        ]
    else:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]

    by_class: dict = defaultdict(list)
    for i, t in enumerate(targets):
        if t in class_set:
            by_class[t].append(i)

    selected: List[int] = []
    for cls in classes:
        idxs = by_class.get(cls, [])
        rng.shuffle(idxs)
        selected.extend(idxs[:k])

    return Subset(dataset, selected)


class FewShot(CLScenario):
    """
    Few-shot continual learning scenario.

    Limits each task's training split to k_shots examples per class;
    inherits head setup from base_scenario (default: ClassIncremental).
    """

    def __init__(
        self,
        benchmark:     CLBenchmark,
        k_shots:       Optional[int]       = 5,
        base_scenario: Type[CLScenario]    = ClassIncremental,
        seed:          int                 = 42,
        batch_size:    int                 = 64,
        num_workers:   int                 = 4,
        pin_memory:    bool                = True,
    ) -> None:
        super().__init__(benchmark, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)
        self.k_shots       = k_shots
        self._seed         = seed
        self._base_cls     = base_scenario

        self._base: CLScenario = base_scenario(
            benchmark,
            batch_size  = batch_size,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

    # ------------------------------------------------------------------
    # Head setup — delegate to the chosen base scenario
    # ------------------------------------------------------------------

    def setup(self, model: CLModel) -> None:
        self._base.setup(model)

    # ------------------------------------------------------------------
    # DataLoaders — subsample training, keep full test
    # ------------------------------------------------------------------

    def _build_loaders(self) -> List[TaskInfo]:
        base_infos = self._base._build_loaders()

        few_shot_infos: List[TaskInfo] = []
        for info, split in zip(base_infos, self.benchmark.get_splits()):
            if self.k_shots is not None:
                few_train_ds = _k_shot_subset(
                    split.train_dataset,
                    split.classes,
                    self.k_shots,
                    seed=self._seed + info.task_id,
                )
                few_train_ds = self._apply_remap(few_train_ds, info)

                train_loader = DataLoader(
                    few_train_ds,
                    batch_size  = self.batch_size,
                    shuffle     = True,
                    num_workers = self.num_workers,
                    pin_memory  = self.pin_memory,
                )
            else:
                train_loader = info.train_loader

            few_shot_infos.append(TaskInfo(
                task_id     = info.task_id,
                classes     = info.classes,
                train_loader = train_loader,
                test_loader  = info.test_loader,
            ))

        return few_shot_infos

    # ------------------------------------------------------------------
    # Iteration — FewShot controls num_tasks, not the base
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.benchmark.num_tasks

    @property
    def num_tasks(self) -> int:
        return self.benchmark.num_tasks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_remap(self, subset: Subset, task_info: TaskInfo) -> Subset:
        """Re-apply TaskIncremental's label remap to the k-shot subset, if needed."""
        from ctk.scenarios.task_incremental import TaskIncremental
        if not isinstance(self._base, TaskIncremental):
            return subset
        from ctk.scenarios.task_incremental import _LabelRemap
        return _LabelRemap(subset, task_info.class_offset)

    @property
    def total_train_samples(self) -> int:
        """Total number of training samples across all tasks."""
        n_classes_total = self.benchmark.total_classes
        effective_k     = self.k_shots or float("inf")
        return int(n_classes_total * min(effective_k,
                   self.benchmark.classes_per_task * 500))   # rough upper bound

"""
FSCIL scenario: task 0 uses the full training split; tasks 1+ use k_shots per class.
"""

from __future__ import annotations

from typing import List, Optional, Type

from ctk.benchmarks.base import CLBenchmark
from ctk.models.base import CLModel
from ctk.scenarios.base import CLScenario, TaskInfo
from ctk.scenarios.class_incremental import ClassIncremental
from ctk.scenarios.few_shot import FewShot, _k_shot_subset
from torch.utils.data import DataLoader


class FSCIL(CLScenario):
    """FSCIL: full training for task 0, k_shots per class for tasks 1+."""

    def __init__(
        self,
        benchmark:  CLBenchmark,
        k_shots:    int  = 5,
        seed:       int  = 42,
        batch_size: int  = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(benchmark, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)
        self.k_shots = k_shots
        self._seed   = seed

    # ------------------------------------------------------------------
    # Head setup — Class-IL (growing output, no task oracle at test time)
    # ------------------------------------------------------------------

    def setup(self, model: CLModel) -> None:
        from ctk.models.base import IncrementalClassifier
        head = IncrementalClassifier(model.embed_dim, initial_classes=0)
        model.set_head(head)
        for split in self.benchmark.get_splits():
            model.grow_head(len(split.classes))

    # ------------------------------------------------------------------
    # DataLoaders — full base session, K-shot for incremental sessions
    # ------------------------------------------------------------------

    def _build_loaders(self) -> List[TaskInfo]:
        task_infos: List[TaskInfo] = []

        for split in self.benchmark.get_splits():
            if split.task_id == 0:
                train_ds = split.train_dataset
            else:
                train_ds = _k_shot_subset(
                    split.train_dataset,
                    split.classes,
                    self.k_shots,
                    seed=self._seed + split.task_id,
                )

            train_loader = DataLoader(
                train_ds,
                batch_size  = self.batch_size,
                shuffle     = True,
                num_workers = self.num_workers,
                pin_memory  = self.pin_memory,
            )
            test_loader = DataLoader(
                split.test_dataset,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = self.pin_memory,
            )
            task_infos.append(TaskInfo(
                task_id      = split.task_id,
                classes      = list(split.classes),
                train_loader = train_loader,
                test_loader  = test_loader,
            ))

        return task_infos

    @property
    def num_tasks(self) -> int:
        return self.benchmark.num_tasks

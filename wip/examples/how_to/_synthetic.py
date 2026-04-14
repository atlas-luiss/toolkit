"""
Shared synthetic helpers used by all how-to examples.

Keeps every example file self-contained — no real datasets or internet
access required.  Swap these out for your own data when ready.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from typing import List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ctk.benchmarks.base import CLBenchmark, TaskSplit
from ctk.models.base import CLModel, IncrementalClassifier


# ---------------------------------------------------------------------------
# Synthetic CLBenchmark
# ---------------------------------------------------------------------------

class SyntheticBenchmark(CLBenchmark):
    """
    Generates random (x, y) tensors split into disjoint-class tasks.

    Args:
        num_tasks:       how many sequential tasks
        classes_per_task: how many distinct classes per task
        n_train:         training samples per task
        n_test:          test samples per task
        input_dim:       feature dimension of each sample
    """

    def __init__(
        self,
        num_tasks:        int = 10,
        classes_per_task: int = 10,
        n_train:          int = 500,
        n_test:           int = 150,
        input_dim:        int = 32,
        seed:             int = 0,
    ):
        torch.manual_seed(seed)
        self._num_tasks        = num_tasks
        self._classes_per_task = classes_per_task
        self._total_classes    = num_tasks * classes_per_task

        self._splits: List[TaskSplit] = []
        for t in range(num_tasks):
            global_classes = list(range(t * classes_per_task,
                                        (t + 1) * classes_per_task))
            # Labels drawn uniformly from this task's global class indices
            y_train = torch.randint(0, classes_per_task, (n_train,)) \
                      + t * classes_per_task
            y_test  = torch.randint(0, classes_per_task, (n_test,)) \
                      + t * classes_per_task

            self._splits.append(TaskSplit(
                task_id       = t,
                classes       = global_classes,
                train_dataset = TensorDataset(
                    torch.randn(n_train, input_dim), y_train),
                test_dataset  = TensorDataset(
                    torch.randn(n_test, input_dim), y_test),
            ))

    @property
    def num_tasks(self) -> int:         return self._num_tasks
    @property
    def total_classes(self) -> int:     return self._total_classes
    @property
    def classes_per_task(self) -> int:  return self._classes_per_task

    def get_splits(self) -> List[TaskSplit]:
        return self._splits


# ---------------------------------------------------------------------------
# Tiny CLModel (MLP backbone)
# ---------------------------------------------------------------------------

class TinyModel(CLModel):
    """
    Two-layer MLP wrapped as a CLModel.
    Drop-in replacement for ViTModel / CLIPModel in how-to examples.
    """

    def __init__(self, input_dim: int = 32, embed_dim: int = 64,
                 device=None):
        super().__init__(device=device)
        self._backbone = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self._device)
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def _default_lora_targets(self) -> list:
        return ["0", "2"]   # indices of Linear layers inside the Sequential

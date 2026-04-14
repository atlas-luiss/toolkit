"""
HOW-TO: Custom Benchmark
=========================
Wrap any dataset you already have into a CLBenchmark so it works with
all CTK scenarios, strategies, and trainers out of the box.

You need to implement four things:
  1. num_tasks        (int property)
  2. total_classes    (int property)
  3. classes_per_task (int property)
  4. get_splits()     (returns List[TaskSplit])

TaskSplit holds:
  task_id        — int index starting from 0
  classes        — list of global class indices for this task
  train_dataset  — any torch.utils.data.Dataset
  test_dataset   — any torch.utils.data.Dataset

This example shows how to wrap an ImageFolder-style directory.
Replace IMAGENET_R_ROOT with the path to your own image dataset.
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tvds, transforms
from typing import List

from ctk.benchmarks.base import CLBenchmark, TaskSplit
from ctk.benchmarks.split import make_class_splits
from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import NaiveStrategy

from _setup import make_model, IMAGENET_R_ROOT, DEVICE


# ---------------------------------------------------------------------------
# Step 1 — define your CLBenchmark
# ---------------------------------------------------------------------------

class MyImageFolderBenchmark(CLBenchmark):
    """
    Wraps any ImageFolder-style directory into a CL benchmark.

    The directory should have one subdirectory per class:
        root/
          class_a/  img1.jpg  img2.jpg  ...
          class_b/  ...

    Classes are shuffled and split evenly across tasks.
    Replace `root` with the path to your own dataset.
    """

    def __init__(
        self,
        root: str,
        num_tasks: int = 5,
        test_split: float = 0.2,
        image_size: int = 224,
        seed: int = 42,
    ):
        self._num_tasks = num_tasks

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        full_ds = tvds.ImageFolder(root)
        self._total_classes   = len(full_ds.classes)
        self._classes_per_task = self._total_classes // num_tasks
        self._idx_to_classname = {v: k for k, v in full_ds.class_to_idx.items()}

        # Stratified train/test split
        from sklearn.model_selection import train_test_split as sk_split
        samples = full_ds.samples
        targets = [s[1] for s in samples]
        train_s, test_s = sk_split(samples, test_size=test_split,
                                   stratify=targets, random_state=seed)

        # Datasets with the right transforms
        class _ListDS(Dataset):
            def __init__(self, items, transform):
                from torchvision.datasets.folder import default_loader
                self.items     = items
                self.targets   = [int(s[1]) for s in items]
                self.transform = transform
                self.loader    = default_loader
            def __len__(self):  return len(self.items)
            def __getitem__(self, i):
                path, y = self.items[i]
                return self.transform(self.loader(path)), y

        train_ds = _ListDS(train_s, train_tf)
        test_ds  = _ListDS(test_s,  test_tf)

        self._splits = make_class_splits(
            train_dataset=train_ds,
            test_dataset=test_ds,
            num_tasks=num_tasks,
            classes_per_task=self._classes_per_task,
            shuffle_classes=True,
            seed=seed,
        )

    # --- required properties ---
    @property
    def num_tasks(self) -> int:         return self._num_tasks
    @property
    def total_classes(self) -> int:     return self._total_classes
    @property
    def classes_per_task(self) -> int:  return self._classes_per_task

    def get_splits(self) -> List[TaskSplit]:
        return self._splits

    def class_name(self, class_id: int) -> str:
        return self._idx_to_classname.get(class_id, str(class_id))


# ---------------------------------------------------------------------------
# Step 2 — plug into CTK as normal
# ---------------------------------------------------------------------------
benchmark = MyImageFolderBenchmark(
    root=IMAGENET_R_ROOT,   # swap with your own image folder
    num_tasks=3,
)
print(f"Benchmark  →  {benchmark.num_tasks} tasks  "
      f"|  {benchmark.classes_per_task} classes/task  "
      f"|  {benchmark.total_classes} total")

scenario = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
model    = make_model()

results = Trainer(
    model, scenario, NaiveStrategy(),
    optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
    epochs_per_task=1, device=DEVICE, verbose=True,
).fit()

print(f"\nCustom benchmark — avg_accuracy: {results['avg_accuracy']:.1f}%")

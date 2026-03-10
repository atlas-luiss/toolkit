"""
Split-CUB-200 benchmark: 200 bird classes divided into num_tasks groups.

Supports ImageFolder layout (root/train/ and root/test/) or raw CUB_200_2011
layout (auto-detected).
"""

import os
from typing import List

from torchvision import transforms

from .base import CLBenchmark, TaskSplit
from .split import make_class_splits


def _build_cub200_transforms(image_size: int = 224):
    """Standard ImageNet-style transforms used for CUB-200."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, test_transform


def _load_imagefolder(root: str, train_tf, test_tf):
    """Load CUB-200 from ImageFolder layout (root/train/ and root/test/)."""
    from torchvision import datasets as tvds

    train_path = os.path.join(root, 'train')
    test_path  = os.path.join(root, 'test')

    train_ds = tvds.ImageFolder(train_path, transform=train_tf)
    test_ds  = tvds.ImageFolder(test_path,  transform=test_tf)

    train_ds.targets = [int(t) for t in train_ds.targets]
    test_ds.targets  = [int(t) for t in test_ds.targets]

    idx_to_classname = {v: k for k, v in train_ds.class_to_idx.items()}
    return train_ds, test_ds, idx_to_classname


def _load_cub200dataset(root: str, train_tf, test_tf):
    """Load CUB-200 from raw CUB_200_2011 layout via ctk.datasets.cub200.CUB200Dataset."""
    from ctk.datasets.cub200 import CUB200Dataset

    train_ds = CUB200Dataset(root=root, train=True,  transform=train_tf)
    test_ds  = CUB200Dataset(root=root, train=False, transform=test_tf)

    num_classes = max(train_ds.targets) + 1
    idx_to_classname = {i: str(i) for i in range(num_classes)}
    return train_ds, test_ds, idx_to_classname


class SplitCUB200(CLBenchmark):
    """Class-incremental CUB-200 benchmark; auto-detects ImageFolder vs raw layout."""

    def __init__(
        self,
        root: str,
        num_tasks: int = 10,
        image_size: int = 224,
        seed: int = 42,
        shuffle_classes: bool = True,
    ):
        self._num_tasks = num_tasks
        self._total_classes = 200
        self._classes_per_task = self._total_classes // num_tasks

        train_tf, test_tf = _build_cub200_transforms(image_size)

        train_path = os.path.join(root, 'train')
        test_path  = os.path.join(root, 'test')

        if os.path.isdir(train_path) and os.path.isdir(test_path):
            train_ds, test_ds, idx_to_classname = _load_imagefolder(
                root, train_tf, test_tf
            )
        else:
            train_ds, test_ds, idx_to_classname = _load_cub200dataset(
                root, train_tf, test_tf
            )

        self._idx_to_classname = idx_to_classname

        self._splits: List[TaskSplit] = make_class_splits(
            train_dataset=train_ds,
            test_dataset=test_ds,
            num_tasks=num_tasks,
            classes_per_task=self._classes_per_task,
            shuffle_classes=shuffle_classes,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # CLBenchmark interface
    # ------------------------------------------------------------------

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    @property
    def total_classes(self) -> int:
        return self._total_classes

    @property
    def classes_per_task(self) -> int:
        return self._classes_per_task

    def get_splits(self) -> List[TaskSplit]:
        return self._splits

    def class_name(self, class_id: int) -> str:
        return self._idx_to_classname.get(class_id, str(class_id))

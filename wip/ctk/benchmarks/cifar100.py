"""Split-CIFAR-100 benchmark: 100 classes divided into num_tasks disjoint groups."""

from typing import List, Optional

from torchvision import datasets, transforms

from .base import CLBenchmark, TaskSplit
from .split import make_class_splits


def _build_cifar100_transforms(image_size: int = 224):
    """Return (train_transform, test_transform) for CIFAR-100."""
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    resize_size = int(image_size * 256 / 224)
    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


class SplitCIFAR100(CLBenchmark):
    """Class-incremental CIFAR-100 split into num_tasks disjoint class groups."""

    def __init__(
        self,
        root: str = './data',
        num_tasks: int = 5,
        image_size: int = 224,
        seed: int = 42,
        download: bool = True,
        shuffle_classes: bool = True,
    ):
        self._num_tasks = num_tasks
        self._total_classes = 100
        self._classes_per_task = self._total_classes // num_tasks

        train_tf, test_tf = _build_cifar100_transforms(image_size)

        train_ds = datasets.CIFAR100(
            root=root, train=True, download=download, transform=train_tf
        )
        test_ds = datasets.CIFAR100(
            root=root, train=False, download=download, transform=test_tf
        )

        self._idx_to_classname = {
            v: k for k, v in train_ds.class_to_idx.items()
        }

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

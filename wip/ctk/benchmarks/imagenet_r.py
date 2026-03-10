"""ImageNet-R benchmark: stratified train/test split from a single ImageFolder, divided into tasks."""

import os
from typing import List

from torchvision import datasets as tvds, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from .base import CLBenchmark, TaskSplit
from .split import make_class_splits


def _build_imagenet_transforms(image_size: int = 224):
    """Standard ImageNet transforms."""
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


class _SampleListDataset(Dataset):
    """Dataset backed by an explicit (path, target) list; exposes .targets for make_class_splits."""

    def __init__(self, samples, transform=None):
        self.samples  = list(samples)
        self.targets  = [int(s[1]) for s in self.samples]
        self.transform = transform
        self.loader   = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _train_test_split_stratified(samples, targets, test_split: float, seed: int):
    """Stratified train/test split using sklearn if available, else deterministic random split."""
    try:
        from sklearn.model_selection import train_test_split
        train_s, test_s = train_test_split(
            samples,
            test_size=test_split,
            stratify=targets,
            random_state=seed,
        )
    except ImportError:
        import random
        rng = random.Random(seed)
        combined = list(zip(samples, targets))
        rng.shuffle(combined)
        n_test = max(1, int(len(combined) * test_split))
        test_s  = [s for s, _ in combined[:n_test]]
        train_s = [s for s, _ in combined[n_test:]]
    return train_s, test_s


class ImageNetR(CLBenchmark):
    """Class-incremental ImageNet-R benchmark with automatic stratified train/test split."""

    def __init__(
        self,
        root: str,
        num_tasks: int = 10,
        image_size: int = 224,
        test_split: float = 0.2,
        seed: int = 42,
        shuffle_classes: bool = True,
    ):
        self._num_tasks = num_tasks

        train_tf, test_tf = _build_imagenet_transforms(image_size)

        full_ds = tvds.ImageFolder(root)
        all_samples = full_ds.samples
        all_targets = [s[1] for s in all_samples]

        self._total_classes   = len(full_ds.classes)
        self._classes_per_task = self._total_classes // num_tasks
        self._idx_to_classname = {v: k for k, v in full_ds.class_to_idx.items()}

        train_samples, test_samples = _train_test_split_stratified(
            all_samples, all_targets, test_split, seed
        )

        train_ds = _SampleListDataset(train_samples, transform=train_tf)
        test_ds  = _SampleListDataset(test_samples,  transform=test_tf)

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

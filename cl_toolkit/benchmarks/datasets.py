"""Named benchmarks + generic folder loader.

Every loader takes ``transforms=(train_tf, test_tf)``. When omitted, standard
ImageNet-style transforms are used; :func:`cl_toolkit.run` passes the *model's own*
preprocessing (mean/std/resolution from timm), which matters a lot for
pretrained backbones (e.g. CLIP and in21k ViTs do not use ImageNet stats).
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, Optional, Tuple

from torch.utils.data import Dataset

from .base import CLBenchmark
from .split import split_by_class

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def default_transforms(image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T
    train = T.Compose([T.RandomResizedCrop(image_size), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(mean, std)])
    test = T.Compose([T.Resize(int(image_size * 256 / 224)), T.CenterCrop(image_size),
                      T.ToTensor(), T.Normalize(mean, std)])
    return train, test


class _Samples(Dataset):
    """(path, label) list dataset exposing ``.targets``."""

    def __init__(self, samples, transform=None):
        from torchvision.datasets.folder import default_loader
        self.samples, self.transform, self.loader = list(samples), transform, default_loader
        self.targets = [int(s[1]) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path)
        return (self.transform(img) if self.transform else img), y


def _stratified(samples, test_split: float, seed: int):
    by = {}
    for s in samples:
        by.setdefault(s[1], []).append(s)
    rng, tr, te = random.Random(seed), [], []
    for _, items in sorted(by.items()):
        rng.shuffle(items)
        n = max(1, round(len(items) * test_split))
        te += items[:n]
        tr += items[n:]
    return tr, te


def from_folder(
    root: str,
    num_tasks: int = 10,
    transforms: Optional[Tuple[Callable, Callable]] = None,
    test_split: float = 0.2,
    seed: int = 42,
    name: Optional[str] = None,
    **split_kw,
) -> CLBenchmark:
    """Any ImageFolder-layout dataset.

    * ``root/train/<class>/*`` + ``root/test/<class>/*`` → used as-is.
    * ``root/<class>/*`` → stratified ``test_split`` hold-out per class (seeded).
    """
    from torchvision.datasets import ImageFolder
    tr_tf, te_tf = transforms or default_transforms()
    if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "test")):
        tr = ImageFolder(os.path.join(root, "train"))
        te = ImageFolder(os.path.join(root, "test"))
        tr_s, te_s, c2i = tr.samples, te.samples, tr.class_to_idx
    else:
        full = ImageFolder(root)
        tr_s, te_s = _stratified(full.samples, test_split, seed)
        c2i = full.class_to_idx
    names = {v: k for k, v in c2i.items()}
    return split_by_class(_Samples(tr_s, tr_tf), _Samples(te_s, te_tf), num_tasks, seed=seed,
                          class_names=names, name=name or os.path.basename(root.rstrip("/")), **split_kw)


def cifar100(root: str, num_tasks: int = 10, transforms=None, seed: int = 42, download: bool = True, **kw):
    from torchvision.datasets import CIFAR100
    tr_tf, te_tf = transforms or default_transforms(224, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    tr = CIFAR100(root, train=True, download=download, transform=tr_tf)
    te = CIFAR100(root, train=False, download=download, transform=te_tf)
    return split_by_class(tr, te, num_tasks, seed=seed, name="cifar100",
                          class_names={v: k for k, v in tr.class_to_idx.items()}, **kw)


def cifar10(root: str, num_tasks: int = 5, transforms=None, seed: int = 42, download: bool = True, **kw):
    from torchvision.datasets import CIFAR10
    tr_tf, te_tf = transforms or default_transforms()
    tr = CIFAR10(root, train=True, download=download, transform=tr_tf)
    te = CIFAR10(root, train=False, download=download, transform=te_tf)
    return split_by_class(tr, te, num_tasks, seed=seed, name="cifar10", **kw)


# name -> loader(root, num_tasks, transforms, seed, **kw)
DATASETS: Dict[str, Callable[..., CLBenchmark]] = {
    "cifar100": cifar100,
    "cifar10": cifar10,
    "imagenet-r": lambda root, num_tasks=10, **kw: from_folder(root, num_tasks, name="imagenet-r", **kw),
    "imagenet-a": lambda root, num_tasks=10, **kw: from_folder(root, num_tasks, name="imagenet-a", **kw),
    "cub200": lambda root, num_tasks=10, **kw: from_folder(root, num_tasks, name="cub200", **kw),
    "folder": from_folder,
    "toy": lambda root=None, num_tasks=10, **kw: __import__("cl_toolkit.benchmarks.toy", fromlist=["toy"]).toy(
        root, num_tasks, **kw),
}


def get_benchmark(name: str, root: Optional[str], num_tasks: int = 10, **kw) -> CLBenchmark:
    """``get_benchmark("cifar100", root, 10)``; ``"hf:fancyzhx/ag_news"`` loads a Hugging Face text
    dataset; unknown names that are directories load as image folders."""
    if name.startswith("hf:"):
        from .text import hf_text
        return hf_text(name[3:], num_tasks, root=root, **kw)
    key = name.lower().replace("_", "-")
    if root is None and key in DATASETS and key not in ("folder", "toy"):
        root = os.path.join("data", key)
        print(f"[cl_toolkit] no root given: using ./{root}", flush=True)
    if key in DATASETS:
        return DATASETS[key](root, num_tasks=num_tasks, **kw)
    root = root or name
    if os.path.isdir(name):
        return from_folder(name, num_tasks, **kw)
    if os.path.isdir(root or ""):
        return from_folder(root, num_tasks, **kw)
    raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)} (or pass a folder path)")


# --- backwards-compatible class-style constructors --------------------------
def SplitCIFAR100(root="./data", num_tasks=5, image_size=224, seed=42, download=True, shuffle_classes=True):
    return cifar100(root, num_tasks, seed=seed, download=download, shuffle_classes=shuffle_classes,
                    transforms=default_transforms(image_size, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))


def ImageNetR(root, num_tasks=10, image_size=224, test_split=0.2, seed=42, shuffle_classes=True):
    return from_folder(root, num_tasks, default_transforms(image_size), test_split, seed,
                       "imagenet-r", shuffle_classes=shuffle_classes)


def ImageNetA(root, num_tasks=10, image_size=224, test_split=0.2, seed=42, shuffle_classes=True):
    return from_folder(root, num_tasks, default_transforms(image_size), test_split, seed,
                       "imagenet-a", shuffle_classes=shuffle_classes)


def SplitCUB200(root, num_tasks=10, image_size=224, seed=42, shuffle_classes=True):
    return from_folder(root, num_tasks, default_transforms(image_size), seed=seed,
                       name="cub200", shuffle_classes=shuffle_classes)

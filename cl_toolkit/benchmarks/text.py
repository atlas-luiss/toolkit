"""Text classification streams. The model's tokenizer is the transform (like image transforms)."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from .base import CLBenchmark
from .split import split_by_class


class TextDataset(Dataset):
    """(text, label) pairs; ``transform`` (e.g. a tokenizer fn) is applied per item."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], transform: Optional[Callable] = None):
        self.texts, self.targets, self.transform = list(texts), [int(y) for y in labels], transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = self.texts[i]
        return (self.transform(t) if self.transform else t), self.targets[i]


def from_texts(train: Tuple[Sequence[str], Sequence[int]], test: Tuple[Sequence[str], Sequence[int]],
               num_tasks: int, transforms=None, **split_kw) -> CLBenchmark:
    """Class-incremental text stream from ``(texts, labels)`` lists."""
    tr_tf, te_tf = transforms or (None, None)
    return split_by_class(TextDataset(*train, tr_tf), TextDataset(*test, te_tf), num_tasks, **split_kw)


def _subsample(texts, labels, per_class: Optional[int], seed: int):
    if not per_class:
        return list(texts), list(labels)
    rng, by = random.Random(seed), {}
    for t, y in zip(texts, labels):
        by.setdefault(int(y), []).append(t)
    out_t, out_y = [], []
    for y, ts in sorted(by.items()):
        rng.shuffle(ts)
        out_t += ts[:per_class]
        out_y += [y] * len(ts[:per_class])
    return out_t, out_y


def hf_text(name: str, num_tasks: int = 2, transforms=None, config: Optional[str] = None,
            text_field: Optional[str] = None, label_field: Optional[str] = None,
            train_split: str = "train", test_split: str = "test",
            max_train_per_class: Optional[int] = None, max_test_per_class: Optional[int] = None,
            seed: int = 42, root: Optional[str] = None, **split_kw) -> CLBenchmark:
    """Any Hugging Face text-classification dataset, e.g. ``hf_text("fancyzhx/ag_news", 2)``.

    Text/label columns are auto-detected. ``max_*_per_class`` subsample for quick runs.
    """
    from datasets import load_dataset
    ds = load_dataset(name, config, cache_dir=root)
    cols = ds[train_split].column_names
    text_field = text_field or next(c for c in ("text", "sentence", "content", "question", "title") if c in cols)
    label_field = label_field or next(c for c in ("label", "labels", "coarse_label", "topic") if c in cols)
    names = None
    feat = ds[train_split].features.get(label_field)
    if hasattr(feat, "names"):
        names = dict(enumerate(feat.names))
    tr = _subsample(ds[train_split][text_field], ds[train_split][label_field], max_train_per_class, seed)
    te = _subsample(ds[test_split][text_field], ds[test_split][label_field], max_test_per_class, seed)
    tr_tf, te_tf = transforms or (None, None)
    return split_by_class(TextDataset(*tr, tr_tf), TextDataset(*te, te_tf), num_tasks, seed=seed,
                          class_names=names, name=name, **split_kw)

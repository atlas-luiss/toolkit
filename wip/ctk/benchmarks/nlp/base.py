"""Base class and shared utilities for NLP fine-tuning benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class NLPBenchmark(ABC):
    """Base class for NLP fine-tuning datasets.

    Subclasses load a HF dataset, tokenize it, and expose train/eval DataLoaders.
    The tokenizer is passed in so the benchmark is model-agnostic.
    """

    def __init__(
        self,
        tokenizer,
        max_len: int           = 512,
        cache_dir: Optional[str] = None,
        num_workers: int       = 4,
        pin_memory: bool       = True,
    ):
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.cache_dir   = cache_dir
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

    @abstractmethod
    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        """DataLoader for the training split."""

    @abstractmethod
    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        """DataLoader for the evaluation split."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """One of: 'causal_lm', 'seq2seq_lm', 'classification'."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _to_loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_pad_collate,
        )


def _pad_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a list of tokenized dicts to the same length."""
    keys = batch[0].keys()
    out  = {}
    for key in keys:
        tensors = [item[key] for item in batch]
        if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() > 0:
            out[key] = torch.nn.utils.rnn.pad_sequence(
                tensors,
                batch_first=True,
                padding_value=-100 if key == "labels" else 0,
            )
        else:
            out[key] = torch.tensor(tensors)
    return out

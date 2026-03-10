"""Text classification benchmarks: SST-2, AG News, MNLI."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .base import NLPBenchmark, _pad_collate


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

SST2_LABELS   = {0: "negative", 1: "positive"}
AGNEWS_LABELS = {0: "world", 1: "sports", 2: "business", 3: "science"}
MNLI_LABELS   = {0: "entailment", 1: "neutral", 2: "contradiction"}


# ---------------------------------------------------------------------------
# Tokenized dataset wrapper
# ---------------------------------------------------------------------------

class _ClassificationDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ---------------------------------------------------------------------------
# Shared tokenizer
# ---------------------------------------------------------------------------

def _tokenize_classification(
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_len: int,
    task_type: str = "classification",
) -> List[Dict]:
    """Tokenize text for classification (seq cls) or causal LM with label appended."""
    records = []
    if task_type == "classification":
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        for i, label in enumerate(labels):
            records.append({
                "input_ids":      torch.tensor(encodings["input_ids"][i]),
                "attention_mask": torch.tensor(encodings["attention_mask"][i]),
                "labels":         torch.tensor(label),
            })
    else:
        # Causal LM: append label word to text and compute CLM loss on full sequence
        for text, label in zip(texts, labels):
            label_word = str(label)
            full_text  = f"{text} {label_word}"
            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            records.append({
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         input_ids.clone(),
            })
    return records


# ---------------------------------------------------------------------------
# SST-2
# ---------------------------------------------------------------------------

class SST2(NLPBenchmark):
    """Stanford Sentiment Treebank (binary); from glue/sst2."""

    def __init__(
        self,
        tokenizer,
        max_len: int                = 128,
        cache_dir: Optional[str]    = None,
        model_task_type: str        = "classification",   # "classification" or "causal_lm"
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._model_task_type = model_task_type
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds     = load_dataset("glue", "sst2", cache_dir=self.cache_dir)
        train_ = self._process(ds["train"])
        eval_  = self._process(ds["validation"])
        return train_, eval_

    def _process(self, split) -> _ClassificationDataset:
        texts  = [ex["sentence"] for ex in split]
        labels = [ex["label"]    for ex in split]
        return _ClassificationDataset(
            _tokenize_classification(
                self.tokenizer, texts, labels, self.max_len, self._model_task_type
            )
        )

    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return self._model_task_type

    @property
    def num_labels(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# AG News
# ---------------------------------------------------------------------------

class AGNews(NLPBenchmark):
    """AG News topic classification (4 classes)."""

    def __init__(
        self,
        tokenizer,
        max_len: int                = 128,
        cache_dir: Optional[str]    = None,
        model_task_type: str        = "classification",
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._model_task_type = model_task_type
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("ag_news", cache_dir=self.cache_dir)
        return (
            self._process(ds["train"]),
            self._process(ds["test"]),
        )

    def _process(self, split) -> _ClassificationDataset:
        texts  = [ex["text"]  for ex in split]
        labels = [ex["label"] for ex in split]
        return _ClassificationDataset(
            _tokenize_classification(
                self.tokenizer, texts, labels, self.max_len, self._model_task_type
            )
        )

    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return self._model_task_type

    @property
    def num_labels(self) -> int:
        return 4


# ---------------------------------------------------------------------------
# MNLI
# ---------------------------------------------------------------------------

class MNLI(NLPBenchmark):
    """Multi-Genre Natural Language Inference (3 classes); from glue/mnli."""

    def __init__(
        self,
        tokenizer,
        max_len: int                = 256,
        cache_dir: Optional[str]    = None,
        model_task_type: str        = "classification",
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._model_task_type = model_task_type
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds    = load_dataset("glue", "mnli", cache_dir=self.cache_dir)
        train_ = self._process(ds["train"])
        eval_  = self._process(ds["validation_matched"])
        return train_, eval_

    def _process(self, split) -> _ClassificationDataset:
        records = []
        for ex in split:
            text = f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}"
            records.append((text, ex["label"]))
        texts, labels = zip(*records)
        return _ClassificationDataset(
            _tokenize_classification(
                self.tokenizer, list(texts), list(labels),
                self.max_len, self._model_task_type
            )
        )

    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return self._model_task_type

    @property
    def num_labels(self) -> int:
        return 3

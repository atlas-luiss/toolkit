"""Summarization benchmarks: SAMSum (dialogue) and CNN/DailyMail."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .base import NLPBenchmark, _pad_collate


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _SummarizationDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ---------------------------------------------------------------------------
# Tokenizers per model type
# ---------------------------------------------------------------------------

def _tokenize_causal(tokenizer, source: str, target: str, max_len: int, max_target: int):
    """Causal LM format: source\n\ntarget (loss on target only)."""
    prompt = f"{source}\n\nSummary:"
    full   = f"{prompt} {target}"

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len - max_target,
        add_special_tokens=True,
    )["input_ids"]

    full_enc = tokenizer(
        full,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    full_ids = full_enc["input_ids"]
    n_prompt = len(prompt_ids)
    labels   = [-100] * n_prompt + full_ids[n_prompt:]

    return {
        "input_ids":      torch.tensor(full_ids),
        "attention_mask": torch.tensor(full_enc["attention_mask"]),
        "labels":         torch.tensor(labels),
    }


def _tokenize_seq2seq(tokenizer, source: str, target: str, max_src: int, max_tgt: int):
    """Seq2seq format: separate source and target sequences."""
    src_enc = tokenizer(
        source,
        truncation=True,
        max_length=max_src,
        padding=False,
        add_special_tokens=True,
    )
    tgt_enc = tokenizer(
        target,
        truncation=True,
        max_length=max_tgt,
        padding=False,
        add_special_tokens=True,
    )
    tgt_ids = tgt_enc["input_ids"]
    # Replace pad token id with -100 for loss masking
    labels = [t if t != tokenizer.pad_token_id else -100 for t in tgt_ids]

    return {
        "input_ids":      torch.tensor(src_enc["input_ids"]),
        "attention_mask": torch.tensor(src_enc["attention_mask"]),
        "labels":         torch.tensor(labels),
    }


# ---------------------------------------------------------------------------
# SAMSum
# ---------------------------------------------------------------------------

class SAMSum(NLPBenchmark):
    """SAMSum dialogue summarization (~16k examples; much smaller than CNN/DM).

    Supports both causal_lm (prompt-based) and seq2seq_lm model types.
    """

    def __init__(
        self,
        tokenizer,
        max_len: int                = 512,
        max_target_len: int         = 128,
        cache_dir: Optional[str]    = None,
        model_task_type: str        = "seq2seq_lm",   # "seq2seq_lm" or "causal_lm"
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._model_task_type = model_task_type
        self.max_target_len   = max_target_len
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("knkarthick/samsum", cache_dir=self.cache_dir)
        return self._process(ds["train"]), self._process(ds["validation"])

    def _process(self, split) -> _SummarizationDataset:
        records = []
        for ex in split:
            source = f"Summarize the following dialogue:\n{ex['dialogue']}"
            target = ex["summary"]
            if self._model_task_type == "causal_lm":
                rec = _tokenize_causal(
                    self.tokenizer, source, target, self.max_len, self.max_target_len
                )
            else:
                rec = _tokenize_seq2seq(
                    self.tokenizer, source, target, self.max_len, self.max_target_len
                )
            records.append(rec)
        return _SummarizationDataset(records)

    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return self._model_task_type


# ---------------------------------------------------------------------------
# CNN / DailyMail
# ---------------------------------------------------------------------------

class CNNDailyMail(NLPBenchmark):
    """CNN/DailyMail news summarization (300k examples; use a subset for quick experiments)."""

    def __init__(
        self,
        tokenizer,
        max_len: int                = 1024,
        max_target_len: int         = 128,
        cache_dir: Optional[str]    = None,
        model_task_type: str        = "seq2seq_lm",
        max_train_samples: Optional[int] = None,   # subsample for speed
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._model_task_type  = model_task_type
        self.max_target_len    = max_target_len
        self.max_train_samples = max_train_samples
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", cache_dir=self.cache_dir)
        train_split = ds["train"]
        if self.max_train_samples:
            train_split = train_split.select(range(self.max_train_samples))
        return self._process(train_split), self._process(ds["validation"])

    def _process(self, split) -> _SummarizationDataset:
        records = []
        for ex in split:
            source = f"Summarize the following article:\n{ex['article']}"
            target = ex["highlights"]
            if self._model_task_type == "causal_lm":
                rec = _tokenize_causal(
                    self.tokenizer, source, target, self.max_len, self.max_target_len
                )
            else:
                rec = _tokenize_seq2seq(
                    self.tokenizer, source, target, self.max_len, self.max_target_len
                )
            records.append(rec)
        return _SummarizationDataset(records)

    def get_train_loader(self, batch_size: int = 4) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return self._model_task_type

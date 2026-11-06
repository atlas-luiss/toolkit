"""Question-answering benchmark: SQuAD v2 (extractive + unanswerable)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .base import NLPBenchmark, _pad_collate


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _squad_prompt(context: str, question: str, answer: str = "") -> str:
    """Format a SQuAD example as a single prompt string."""
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _QADataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ---------------------------------------------------------------------------
# SQuAD v2
# ---------------------------------------------------------------------------

class SQuAD2(NLPBenchmark):
    """SQuAD 2.0 formatted as a causal-LM task (context + question → answer generation).

    Unanswerable questions are mapped to "unanswerable".
    Loss is computed on the answer tokens only (prompt tokens masked with -100).
    """

    def __init__(
        self,
        tokenizer,
        max_len: int                = 512,
        max_answer_len: int         = 64,
        cache_dir: Optional[str]    = None,
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self.max_answer_len = max_answer_len
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=self.cache_dir)
        return self._process(ds["train"]), self._process(ds["validation"])

    def _process(self, split) -> _QADataset:
        records = []
        for ex in split:
            answers = ex.get("answers", {}).get("text", [])
            answer  = answers[0] if answers else "unanswerable"

            prompt = _squad_prompt(ex["context"], ex["question"])
            full   = _squad_prompt(ex["context"], ex["question"], answer)

            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_len - self.max_answer_len,
                add_special_tokens=True,
            )["input_ids"]

            full_enc = self.tokenizer(
                full,
                truncation=True,
                max_length=self.max_len,
                add_special_tokens=True,
            )
            full_ids = full_enc["input_ids"]
            attn     = full_enc["attention_mask"]

            # Mask prompt tokens with -100 so loss is computed on answer only
            n_prompt = len(prompt_ids)
            labels   = [-100] * n_prompt + full_ids[n_prompt:]

            records.append({
                "input_ids":      torch.tensor(full_ids),
                "attention_mask": torch.tensor(attn),
                "labels":         torch.tensor(labels),
            })
        return _QADataset(records)

    def get_train_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 16) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return "causal_lm"

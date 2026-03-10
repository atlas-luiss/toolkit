"""Instruction-tuning benchmarks: Alpaca and GSM8K (math reasoning)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .base import NLPBenchmark


# ---------------------------------------------------------------------------
# Alpaca prompt template
# ---------------------------------------------------------------------------

_ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def _format_alpaca(instruction: str, inp: str, output: str) -> tuple[str, str]:
    """Return (prompt, full_text) for an Alpaca example."""
    if inp.strip():
        prompt = _ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    else:
        prompt = _ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)
    return prompt, prompt + output


# ---------------------------------------------------------------------------
# Shared tokenizer (causal LM, loss on response only)
# ---------------------------------------------------------------------------

def _tokenize_instruct(tokenizer, prompt: str, full: str, max_len: int) -> Optional[Dict]:
    """Tokenize instruction example; mask prompt tokens with -100."""
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )["input_ids"]

    full_enc = tokenizer(
        full,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    full_ids = full_enc["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        return None  # response was truncated away

    n_prompt = len(prompt_ids)
    labels   = [-100] * n_prompt + full_ids[n_prompt:]

    return {
        "input_ids":      torch.tensor(full_ids),
        "attention_mask": torch.tensor(full_enc["attention_mask"]),
        "labels":         torch.tensor(labels),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _InstructDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

class Alpaca(NLPBenchmark):
    """Stanford Alpaca instruction-following dataset (~52k examples).

    Loss is computed on the response tokens only.
    Requires a causal LM.
    """

    def __init__(
        self,
        tokenizer,
        max_len: int                = 512,
        cache_dir: Optional[str]    = None,
        max_train_samples: Optional[int] = None,
        eval_split: float           = 0.05,
        seed: int                   = 42,
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self.max_train_samples = max_train_samples
        self.eval_split        = eval_split
        self.seed              = seed
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir, split="train")
        ds = ds.shuffle(seed=self.seed)

        n_eval = int(len(ds) * self.eval_split)
        eval_  = ds.select(range(n_eval))
        train_ = ds.select(range(n_eval, len(ds)))

        if self.max_train_samples:
            train_ = train_.select(range(min(self.max_train_samples, len(train_))))

        return self._process(train_), self._process(eval_)

    def _process(self, split) -> _InstructDataset:
        records = []
        for ex in split:
            prompt, full = _format_alpaca(ex["instruction"], ex.get("input", ""), ex["output"])
            rec = _tokenize_instruct(self.tokenizer, prompt, full, self.max_len)
            if rec is not None:
                records.append(rec)
        return _InstructDataset(records)

    def get_train_loader(self, batch_size: int = 4) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return "causal_lm"


# ---------------------------------------------------------------------------
# GSM8K (math reasoning)
# ---------------------------------------------------------------------------

_GSM8K_PROMPT = (
    "Solve the following math problem step by step.\n\n"
    "Problem: {question}\n\n"
    "Solution:\n"
)


class GSM8K(NLPBenchmark):
    """GSM8K grade-school math (~8.5k chain-of-thought examples).

    Loss is computed on the solution tokens only.
    Requires a causal LM.
    """

    def __init__(
        self,
        tokenizer,
        max_len: int                = 512,
        cache_dir: Optional[str]    = None,
        num_workers: int            = 4,
        pin_memory: bool            = True,
    ):
        super().__init__(tokenizer, max_len, cache_dir, num_workers, pin_memory)
        self._train, self._eval = self._load()

    def _load(self):
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", cache_dir=self.cache_dir)
        return self._process(ds["train"]), self._process(ds["test"])

    def _process(self, split) -> _InstructDataset:
        records = []
        for ex in split:
            prompt = _GSM8K_PROMPT.format(question=ex["question"])
            full   = prompt + ex["answer"]
            rec    = _tokenize_instruct(self.tokenizer, prompt, full, self.max_len)
            if rec is not None:
                records.append(rec)
        return _InstructDataset(records)

    def get_train_loader(self, batch_size: int = 4) -> DataLoader:
        return self._to_loader(self._train, batch_size, shuffle=True)

    def get_eval_loader(self, batch_size: int = 8) -> DataLoader:
        return self._to_loader(self._eval, batch_size, shuffle=False)

    @property
    def task_type(self) -> str:
        return "causal_lm"

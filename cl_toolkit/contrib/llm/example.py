"""
HOW-TO: LLMTrainer
===================
LLMTrainer fine-tunes a language model (causal LM, seq2seq, or classifier)
on a single downstream task.  It handles:
  - Mixed precision (bf16 / fp16)
  - Gradient accumulation
  - Linear LR warmup
  - Periodic evaluation with custom metrics
  - Checkpoint saving

It is designed to work with LLMModel, which wraps any HuggingFace model
and exposes add_lora() for parameter-efficient fine-tuning.

This example shows the API with a mock model so it runs without downloading
any weights.  Replace MockLLMModel with a real LLMModel for actual use.
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_toolkit.contrib.llm import LLMTrainer, LLMTrainingConfig



# ---------------------------------------------------------------------------
# Data — sequence batches of (input_ids, attention_mask, labels)
# ---------------------------------------------------------------------------

def make_loader(n=64, seq_len=16, batch_size=8):
    input_ids      = torch.randint(0, 32, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    labels         = torch.randint(0, 32, (n, seq_len))
    ds = TensorDataset(input_ids, attention_mask, labels)

    # LLMTrainer expects batches as dicts with keys:
    #   input_ids, attention_mask, labels
    class DictLoader:
        def __init__(self, loader): self._loader = loader
        def __iter__(self):
            for input_ids, mask, labels in self._loader:
                yield {"input_ids": input_ids,
                       "attention_mask": mask,
                       "labels": labels}
        def __len__(self): return len(self._loader)

    return DictLoader(DataLoader(ds, batch_size=batch_size, shuffle=True))


train_loader = make_loader(n=64)
eval_loader  = make_loader(n=16)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = LLMTrainingConfig(
    output_dir                  = "/tmp/llm_output",
    num_epochs                  = 1,          # set higher in real use
    lr                          = 2e-4,
    weight_decay                = 0.01,
    gradient_accumulation_steps = 2,          # effective batch = batch_size * 2
    bf16                        = False,       # set True on Ampere+ GPUs
    fp16                        = False,
    gradient_checkpointing      = False,       # set True for large models
    warmup_ratio                = 0.1,
    logging_steps               = 5,
    eval_steps                  = 50,
    save_steps                  = 999,         # skip saving in this demo
    max_seq_len                 = 16,
)

# ---------------------------------------------------------------------------
# Optional: custom metrics function
# evaluate(eval_loader, model) → dict of metric_name → value
# ---------------------------------------------------------------------------
def compute_metrics(eval_loader, model):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in eval_loader:
            out = model(**{k: v.to(model.device) for k, v in batch.items()})
            total_loss += out.loss.item()
            n += 1
    return {"eval_loss": total_loss / max(n, 1)}

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


from cl_toolkit.contrib.llm import LLMModel

model = LLMModel(
    model_name_or_path = "bert-base-uncased",
    load_in_4bit       = True,     # QLoRA
)
model.add_lora(r=16, lora_alpha=32)

trainer = LLMTrainer(model, config, train_loader, eval_loader, compute_metrics)
trainer.train()

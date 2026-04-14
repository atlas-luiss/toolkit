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
sys.path.insert(0, os.path.join(_here, "../.."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ctk.trainers import LLMTrainer, LLMTrainingConfig


# ---------------------------------------------------------------------------
# Mock LLMModel (replace with ctk.models.LLMModel in real use)
# ---------------------------------------------------------------------------

class MockLLMModel(nn.Module):
    """
    Stands in for LLMModel so this example runs without HuggingFace weights.
    A real LLMModel wraps any HF AutoModelForCausalLM / AutoModelForSeq2SeqLM.
    """
    VOCAB = 32
    EMBED = 64

    def __init__(self):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed   = nn.Embedding(self.VOCAB, self.EMBED)
        self.transformer = nn.Sequential(
            nn.Linear(self.EMBED, self.EMBED), nn.GELU()
        )
        self.lm_head = nn.Linear(self.EMBED, self.VOCAB)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        import torch.nn.functional as F
        # input_ids: (batch, seq_len) → embed → (batch, seq_len, EMBED)
        x      = self.embed(input_ids)
        x      = self.transformer(x)          # (batch, seq_len, EMBED)
        logits = self.lm_head(x)              # (batch, seq_len, VOCAB)
        loss   = None
        if labels is not None:
            # Flatten to (batch*seq_len, VOCAB) vs (batch*seq_len,)
            loss = F.cross_entropy(logits.view(-1, self.VOCAB), labels.view(-1))
        return type("Out", (), {"loss": loss, "logits": logits})()

    def trainable_params(self):
        """Required by LLMTrainer."""
        return [p for p in self.parameters() if p.requires_grad]

    def enable_gradient_checkpointing(self):
        pass   # no-op for the mock


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
model = MockLLMModel()

trainer = LLMTrainer(
    model           = model,
    config          = config,
    train_loader    = train_loader,
    eval_loader     = eval_loader,
    compute_metrics = compute_metrics,
)

results = trainer.train()

print(f"\nLLMTrainer done.")
print(f"  Final train loss: {results.get('train_loss', 'n/a')}")

# ---------------------------------------------------------------------------
# Real usage with LLMModel + LoRA
# ---------------------------------------------------------------------------
# from ctk.models import LLMModel
#
# model = LLMModel(
#     model_name_or_path = "meta-llama/Llama-2-7b-hf",
#     load_in_4bit       = True,     # QLoRA
# )
# model.add_lora(task_id=0, rank=16, lora_alpha=32)
#
# trainer = LLMTrainer(model, config, train_loader, eval_loader)
# trainer.train()

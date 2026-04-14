"""
LoRA Fine-tuning of OPT-125M  ·  AG News  ·  4 tasks
======================================================
Demonstrates continual learning with a language model. AG News has 4 topic
classes (World, Sports, Business, Sci/Tech) — each treated as a separate task.

The model adapts to each topic in sequence using LoRA, which adds trainable
rank-decomposition matrices to the attention layers while keeping the base
OPT weights frozen. Only ~0.5% of parameters are updated per task.

  Model  : facebook/opt-125m  (causal LM, 125M parameters)
  Data   : AG News  (120k train / 7.6k test, 4 balanced classes)
  Tasks  : 4  (one topic per task, seen sequentially)
  Epochs : 2 per task
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import torch
from torch.utils.data import DataLoader, Dataset
from _setup import HF_CACHE, DEVICE, print_results   # noqa: F401 (print_results unused here)

from ctk.models.llm   import LLMModel
from ctk.trainers.llm import LLMTrainer, LLMTrainingConfig

os.environ["HF_HOME"] = HF_CACHE

# ── Config ────────────────────────────────────────────────────────────────────
NUM_TASKS   = 4
EPOCHS      = 2
BATCH_SIZE  = 8
MAX_SEQ_LEN = 256
LR          = 2e-4

AG_NEWS_TOPICS = ["World", "Sports", "Business", "Sci/Tech"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class AGNewsTask(Dataset):
    """Slice of AG News containing only examples for one topic (task_id)."""

    def __init__(self, hf_split, task_id: int, tokenizer):
        self.examples  = [ex for ex in hf_split if ex["label"] == task_id]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx]["text"],
            truncation    = True,
            max_length    = MAX_SEQ_LEN,
            padding       = "max_length",
            return_tensors = "pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignore padding positions in CE loss
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_ag_news():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    ds = load_dataset("ag_news", cache_dir=HF_CACHE)
    return ds["train"], ds["test"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    # ── Model ─────────────────────────────────────────────────────────────────
    # Load OPT-125m in fp32 (small enough not to need mixed precision).
    # LoRA is applied once here — the same adapter is updated across all tasks.
    print("Loading OPT-125M …")
    model = LLMModel(
        model_name_or_path = "facebook/opt-125m",
        task_type          = "causal_lm",
        cache_dir          = HF_CACHE,
        torch_dtype        = torch.float32,
    )
    model.add_lora(r=8, lora_alpha=16, lora_dropout=0.05)
    model.to(torch.device(DEVICE))

    n_trainable = sum(p.numel() for p in model.trainable_params())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Model      →  OPT-125M  |  trainable: {n_trainable:,} / {n_total:,} "
          f"({n_trainable / n_total * 100:.1f}%)")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading AG News …")
    train_full, test_full = load_ag_news()
    print(f"Data       →  AG News  |  train={len(train_full):,}  test={len(test_full):,}")
    print(f"Tasks      →  {NUM_TASKS} topics  |  {EPOCHS} epochs each\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    task_losses = []

    for task_id in range(NUM_TASKS):
        topic = AG_NEWS_TOPICS[task_id]
        print(f"── Task {task_id}: {topic} " + "─" * (44 - len(topic)))

        train_ds = AGNewsTask(train_full, task_id, model.tokenizer)
        eval_ds  = AGNewsTask(test_full,  task_id, model.tokenizer)
        print(f"   train={len(train_ds):,}  eval={len(eval_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=(DEVICE == "cuda"))
        eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                                  num_workers=2, pin_memory=(DEVICE == "cuda"))

        config = LLMTrainingConfig(
            output_dir             = f"/tmp/ctk_opt_task{task_id}",
            num_epochs             = EPOCHS,
            batch_size             = BATCH_SIZE,
            lr                     = LR,
            weight_decay           = 0.01,
            max_grad_norm          = 1.0,
            warmup_ratio           = 0.1,
            bf16                   = False,   # fp32; OPT-125m is small enough
            fp16                   = False,
            gradient_checkpointing = False,
            logging_steps          = 50,
            eval_steps             = 99999,   # skip mid-epoch eval
            save_steps             = 99999,
            max_seq_len            = MAX_SEQ_LEN,
            dataloader_num_workers = 2,
            dataloader_pin_memory  = (DEVICE == "cuda"),
        )

        t0      = time.time()
        results = LLMTrainer(model, config, train_loader, eval_loader).train()
        elapsed = time.time() - t0

        history    = results.get("history", [])
        final_loss = history[-1]["loss"] if history else float("nan")
        task_losses.append(history)
        print(f"   final loss: {final_loss:.4f}  |  time: {elapsed/60:.1f} min\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 60)
    print("  Summary")
    for task_id, history in enumerate(task_losses):
        topic = AG_NEWS_TOPICS[task_id]
        first = history[0]["loss"]  if history else float("nan")
        last  = history[-1]["loss"] if history else float("nan")
        print(f"  Task {task_id} ({topic:<10})  loss: {first:.4f} → {last:.4f}")
    print(f"\n  Total time: {(time.time() - t_total) / 60:.1f} min")
    print("─" * 60)


if __name__ == "__main__":
    main()

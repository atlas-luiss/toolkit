"""
LLM fine-tuning: Instruction tuning (Alpaca format).

Model:   any causal LM (default: facebook/opt-125m)
Task:    follow diverse natural language instructions
Method:  LoRA fine-tuning; loss on response tokens only (Alpaca template)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from _setup import HF_CACHE, DATA_CACHE, OUTPUT_DIR, DEFAULT_MODEL, print_results

from ctk.models.llm import LLMModel
from ctk.trainers.llm import LLMTrainer, LLMTrainingConfig
from ctk.benchmarks.nlp import Alpaca

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME = DEFAULT_MODEL   # swap for "meta-llama/Llama-2-7b-hf" etc.

cfg = LLMTrainingConfig(
    output_dir                  = f"{OUTPUT_DIR}/alpaca_instruction",
    num_epochs                  = 3,
    batch_size                  = 4,
    eval_batch_size             = 8,
    gradient_accumulation_steps = 8,    # effective batch = 32
    lr                          = 2e-4,
    warmup_ratio                = 0.03,
    bf16                        = True,
    gradient_checkpointing      = True,
    logging_steps               = 50,
    eval_steps                  = 500,
    save_steps                  = 1000,
    max_seq_len                 = 512,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

print(f"Loading {MODEL_NAME} ...")
model = LLMModel(
    model_name_or_path  = MODEL_NAME,
    task_type           = "causal_lm",
)

model.add_lora(
    r               = 16,
    lora_alpha      = 32.0,
    lora_dropout    = 0.05,
)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

print("Loading Alpaca ...")
bench = Alpaca(
    tokenizer           = model.tokenizer,
    max_len             = cfg.max_seq_len,
    cache_dir           = DATA_CACHE,
    max_train_samples   = 10_000,   # use a subset for a quick run; set None for full 52k
)

train_loader = bench.get_train_loader(batch_size=cfg.batch_size)
eval_loader  = bench.get_eval_loader(batch_size=cfg.eval_batch_size)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

trainer = LLMTrainer(
    model        = model,
    config       = cfg,
    train_loader = train_loader,
    eval_loader  = eval_loader,
)

results = trainer.train()
print_results(results)

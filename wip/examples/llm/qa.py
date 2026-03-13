"""
LLM fine-tuning: Question answering (SQuAD v2).

Model:   any HF causal LM (default: facebook/opt-125m)
Task:    extractive QA — given context + question, generate the answer
Method:  LoRA fine-tuning; loss only on the answer tokens
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from _setup import HF_CACHE, DATA_CACHE, OUTPUT_DIR, DEFAULT_MODEL, print_results

from ctk.models.llm import LLMModel
from ctk.trainers.llm import LLMTrainer, LLMTrainingConfig
from ctk.benchmarks.nlp import SQuAD2

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME = DEFAULT_MODEL   # swap for "meta-llama/Llama-2-7b-hf" etc.

cfg = LLMTrainingConfig(
    output_dir                  = f"{OUTPUT_DIR}/squad2_qa",
    num_epochs                  = 2,
    batch_size                  = 4,
    eval_batch_size             = 8,
    gradient_accumulation_steps = 4,
    lr                          = 2e-4,
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
    cache_dir           = HF_CACHE,
)

model.add_lora(
    r               = 16,
    lora_alpha      = 32.0,
    lora_dropout    = 0.05,
)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

print("Loading SQuAD v2 ...")
bench = SQuAD2(
    tokenizer   = model.tokenizer,
    max_len     = cfg.max_seq_len,
    cache_dir   = DATA_CACHE,
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

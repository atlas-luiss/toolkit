"""
LLM fine-tuning: Text classification (SST-2 sentiment).

Model:   any HF sequence-classification model (default: distilbert-base-uncased)
Task:    binary sentiment analysis (positive / negative)
Method:  LoRA fine-tuning via PEFT
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from _setup import HF_CACHE, DATA_CACHE, OUTPUT_DIR, DEFAULT_MODEL, print_results

from ctk.models.llm import LLMModel
from ctk.trainers.llm import LLMTrainer, LLMTrainingConfig
from ctk.benchmarks.nlp import SST2

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"   # fast; swap for larger model
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR         = 2e-4

cfg = LLMTrainingConfig(
    output_dir                  = f"{OUTPUT_DIR}/sst2_classification",
    num_epochs                  = NUM_EPOCHS,
    batch_size                  = BATCH_SIZE,
    eval_batch_size             = 64,
    lr                          = LR,
    bf16                        = True,
    gradient_checkpointing      = False,   # not needed for small models
    logging_steps               = 100,
    eval_steps                  = 500,
    save_steps                  = 500,
    max_seq_len                 = 128,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

print(f"Loading {MODEL_NAME} ...")
model = LLMModel(
    model_name_or_path  = MODEL_NAME,
    task_type           = "classification",
    num_labels          = 2,
    cache_dir           = HF_CACHE,
)

model.add_lora(
    r               = 8,
    lora_alpha      = 16.0,
    lora_dropout    = 0.05,
    target_modules  = ["q_lin", "v_lin"],   # DistilBERT attention
)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

print("Loading SST-2 ...")
bench = SST2(
    tokenizer           = model.tokenizer,
    max_len             = cfg.max_seq_len,
    cache_dir           = DATA_CACHE,
    model_task_type     = "classification",
)

train_loader = bench.get_train_loader(batch_size=cfg.batch_size)
eval_loader  = bench.get_eval_loader(batch_size=cfg.eval_batch_size)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

trainer = LLMTrainer(
    model       = model,
    config      = cfg,
    train_loader = train_loader,
    eval_loader  = eval_loader,
)

results = trainer.train()
print_results(results)

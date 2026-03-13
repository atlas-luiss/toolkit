"""
LLM fine-tuning: Dialogue summarization (SAMSum).

Model:   seq2seq (e.g. google/flan-t5-base) or causal LM
Task:    summarize a conversation into one sentence
Method:  LoRA fine-tuning; SAMSum is small (~16k) so trains fast
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from _setup import HF_CACHE, DATA_CACHE, OUTPUT_DIR, print_results

from ctk.models.llm import LLMModel
from ctk.trainers.llm import LLMTrainer, LLMTrainingConfig
from ctk.benchmarks.nlp import SAMSum

# ------------------------------------------------------------------
# Config — two options: seq2seq (T5) or causal LM
# ------------------------------------------------------------------

USE_SEQ2SEQ = True   # False → causal LM mode (e.g. OPT)

if USE_SEQ2SEQ:
    MODEL_NAME      = "google/flan-t5-base"
    MODEL_TASK_TYPE = "seq2seq_lm"
    LORA_TARGETS    = ["q", "v"]          # T5 attention
else:
    MODEL_NAME      = "facebook/opt-125m"
    MODEL_TASK_TYPE = "causal_lm"
    LORA_TARGETS    = None                # auto-detect

cfg = LLMTrainingConfig(
    output_dir                  = f"{OUTPUT_DIR}/samsum_summarization",
    num_epochs                  = 5,
    batch_size                  = 8,
    eval_batch_size             = 16,
    gradient_accumulation_steps = 2,
    lr                          = 3e-4,
    bf16                        = True,
    gradient_checkpointing      = True,
    logging_steps               = 50,
    eval_steps                  = 200,
    save_steps                  = 500,
    max_seq_len                 = 512,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

print(f"Loading {MODEL_NAME} ...")
model = LLMModel(
    model_name_or_path  = MODEL_NAME,
    task_type           = MODEL_TASK_TYPE,
)

model.add_lora(
    r               = 16,
    lora_alpha      = 32.0,
    lora_dropout    = 0.05,
    target_modules  = LORA_TARGETS,
)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

print("Loading SAMSum ...")
bench = SAMSum(
    tokenizer           = model.tokenizer,
    max_len             = cfg.max_seq_len,
    cache_dir           = DATA_CACHE,
    model_task_type     = MODEL_TASK_TYPE,
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

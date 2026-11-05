"""Any model: the strategies, heads, adapters and routers do not care what the encoder is.

1. Your own nn.Module:      clt.models.wrap(module)            (or pass the module as model=)
2. Hugging Face (text/LLM): model="hf:bert-base-uncased", data="hf:fancyzhx/ag_news"
"""
import torch.nn as nn

import cl_toolkit as clt
from cl_toolkit import adapters

# 1) any encoder: inputs -> (B, D) features
encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, 128))
model = clt.models.wrap(encoder, embed_dim=128)
print("LoRA can target:", adapters.lora_candidates(model))          # ['1', '3'] (the two Linear layers)

# 2) a Hugging Face text encoder on a Hugging Face text dataset; the tokenizer is the transform.
#    Any LoRA-based method works; choose the layers it adapts:
clt.run(data="hf:fancyzhx/ag_news", data_kwargs={"max_train_per_class": 2000, "max_test_per_class": 500},
        model="hf:bert-base-uncased", scenario="class", num_tasks=2,
        strategy="lora-er", strategy_kwargs={"memory_size": 500},
        lora_targets=["query", "value"], lora_layers=[8, 9, 10, 11], epochs=1, lr=5e-4, batch_size=32)

# Decoder LLMs work the same way (last-token pooling), e.g. model="hf:Qwen/Qwen2.5-0.5B".

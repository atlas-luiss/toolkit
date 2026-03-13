"""LoRA + Replay with CLIP (ViT-B/32) backbone on Split-CIFAR-100, Class-IL."""

import time
import torch
from _setup import *
from ctk.models     import CLIPModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAReplay
from ctk.trainer    import ContinualTrainer

BATCH_SIZE = 32

print("=" * 62)
print("  Split-CIFAR-100 | CLIP ViT-B/32 | LoRA+Replay | Class-IL")
print("=" * 62)
print(f"  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  "
      f"Epochs/task: {EPOCHS}  |  BS: {BATCH_SIZE}")
print()

t0 = time.time()

bench = make_bench()

model = CLIPModel(
    model_name      = "ViT-B-32",
    pretrained      = "openai",
    freeze_backbone = True,
    device          = torch.device(DEVICE),
    cache_dir       = MODEL_CACHE,
)

scenario_kw = {**SCENARIO_KW, "batch_size": BATCH_SIZE}
scenario = ClassIncremental(bench, **scenario_kw)
strategy = LoRAReplay(
    rank         = 8,
    lora_alpha   = 16,
    memory_size  = 200,
    replay_ratio = 0.5,
)

trainer_kw = {**TRAINER_KW, "lr": 1e-3}
results = ContinualTrainer(model, scenario, strategy, **trainer_kw).fit()

print_results(results, time.time() - t0)

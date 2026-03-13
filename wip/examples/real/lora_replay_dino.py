"""LoRA + Replay with DINOv2 backbone on Split-CIFAR-100, Class-IL."""

import time
import torch
from _setup import *
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAReplay
from ctk.trainer    import ContinualTrainer

BATCH_SIZE = 32

print("=" * 62)
print("  Split-CIFAR-100 | DINOv2-B/14 | LoRA+Replay | Class-IL")
print("=" * 62)
print(f"  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  "
      f"Epochs/task: {EPOCHS}  |  BS: {BATCH_SIZE}")
print()

t0 = time.time()

bench = make_bench()

os.environ["TORCH_HOME"] = MODEL_CACHE
model = DINOModel(
    model_name      = "dinov2_vitb14",
    pretrained      = True,
    freeze_backbone = True,
    device          = torch.device(DEVICE),
)

scenario_kw = {**SCENARIO_KW, "batch_size": BATCH_SIZE}
scenario = ClassIncremental(bench, **scenario_kw)
strategy = LoRAReplay(
    rank         = 8,
    lora_alpha   = 16,
    memory_size  = 500,
    replay_ratio = 0.5,
)

trainer_kw = {**TRAINER_KW, "lr": 1e-3}
results = ContinualTrainer(model, scenario, strategy, **trainer_kw).fit()

print_results(results, time.time() - t0)

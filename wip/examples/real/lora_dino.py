"""LoRA fine-tuning with DINOv2 backbone on Split-CIFAR-100."""

import time
from _setup import *
from ctk.models     import DINOModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainer    import ContinualTrainer

print("=" * 62)
print("  Split-CIFAR-100 | DINOv2-B/14 | LoRA | Class-IL")
print("=" * 62)
print(f"  Device: {DEVICE}  |  Tasks: {NUM_TASKS}  |  "
      f"Epochs/task: {EPOCHS}  |  BS: {BATCH_SIZE}")
print()

t0 = time.time()

bench = make_bench()

model = DINOModel(
    model_name      = "dinov2_vitb14",
    pretrained      = True,
    freeze_backbone = True,
    device          = torch.device(DEVICE),
)

scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = LoRAStrategy(rank=8, lora_alpha=16, merge_after_task=False)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()

print_results(results, time.time() - t0)

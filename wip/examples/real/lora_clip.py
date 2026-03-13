"""LoRA fine-tuning with CLIP (ViT-B/32) backbone on Split-CIFAR-100."""

import time
from _setup import *
from ctk.models     import CLIPModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainer    import ContinualTrainer

print("=" * 62)
print("  Split-CIFAR-100 | CLIP ViT-B/32 | LoRA | Class-IL")
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
)

print(model)

scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = LoRAStrategy(rank=8, lora_alpha=16,target_modules=["out_proj", "c_fc", "c_proj"], merge_after_task=False)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()

print_results(results, time.time() - t0)

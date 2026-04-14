"""
HOW-TO: LoRA (PEFT)
====================
LoRA is a parameter-efficient technique that attaches small trainable adapters
to a frozen backbone.  In CTK, it is not a strategy itself — it is a wrapper
that adds an adapter lifecycle on top of any existing strategy.

    LoRAAdapter(strategy, rank=8)

The inner strategy still controls the loss and forgetting prevention.
LoRAAdapter controls when adapters are attached, frozen, and merged.

Combinations shown:
  LoRAAdapter(NaiveStrategy())          — adapters only, no forgetting protection
  LoRAAdapter(ExperienceReplay(...))    — adapters + replay buffer
  LoRAAdapter(EWC(...))                 — adapters + EWC regularisation

Convenience shorthands (same thing, shorter to type):
  LoRAStrategy(rank=8)                  — alias for LoRAAdapter(NaiveStrategy(), ...)
  LoRAReplay(memory_size=500, rank=8)   — alias for LoRAAdapter(ExperienceReplay(...), ...)

Dataset : ImageNet-R  (5 tasks of 40 classes each)
Model   : DINOv2-B/14 — transformer backbone, ideal for LoRA adapters
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import LoRAAdapter, NaiveStrategy, ExperienceReplay, EWC

from _setup import make_benchmark, make_model, DEVICE

NUM_TASKS = 5


def run(strategy, label):
    benchmark = make_benchmark(num_tasks=NUM_TASKS)
    scenario  = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                                 pin_memory=(DEVICE == "cuda"))
    model     = make_model()

    results = Trainer(
        model, scenario, strategy,
        optimizer_cls=torch.optim.AdamW, lr=1e-4, weight_decay=1e-4,
        epochs_per_task=1, device=DEVICE, verbose=False,
    ).fit()

    print(f"  {label:<40}  "
          f"acc={results['avg_accuracy']:5.1f}%  "
          f"forgetting={results['avg_forgetting']:5.1f}%")


print(f"\nLoRA combinations ({NUM_TASKS} tasks, 1 epoch each):\n")

# Adapters only — one per task, frozen after each task, no other protection
run(LoRAAdapter(NaiveStrategy(), rank=8, lora_alpha=16),
    "LoRAAdapter + NaiveStrategy")

# Adapters + replay — useful when tasks share overlapping visual features
run(LoRAAdapter(ExperienceReplay(memory_size=500), rank=8, lora_alpha=16),
    "LoRAAdapter + ExperienceReplay")

# Adapters + EWC — regularises head weights while adapters protect backbone
run(LoRAAdapter(EWC(lam=500, fisher_samples=200), rank=8, lora_alpha=16),
    "LoRAAdapter + EWC")

# ---------------------------------------------------------------------------
# Convenience shorthands (identical to the first two above)
# ---------------------------------------------------------------------------
# from ctk.strategies import LoRAStrategy, LoRAReplay
#
# LoRAStrategy(rank=8, lora_alpha=16)
# LoRAReplay(memory_size=500, rank=8, lora_alpha=16)

# ---------------------------------------------------------------------------
# Customising the adapter lifecycle
# ---------------------------------------------------------------------------
# Subclass LoRAAdapter and override on_lora_task_start / on_lora_task_end:
#
# class SharedAdapterStrategy(LoRAAdapter):
#     \"\"\"One shared adapter across all tasks, never frozen.\"\"\"
#
#     def on_lora_task_start(self, task_id, model):
#         if task_id == 0:          # attach once
#             model.freeze_backbone()
#             model.add_lora(0, rank=self.rank, lora_alpha=self.lora_alpha)
#
#     def on_lora_task_end(self, task_id, model):
#         pass                      # keep adapter trainable

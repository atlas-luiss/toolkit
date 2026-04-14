"""
HOW-TO: Replay Strategies
==========================
Replay strategies store a small subset of past examples in a memory buffer
and mix them back into training when learning new tasks.

Strategies shown:
  ExperienceReplay (ER)  — reservoir-sampled buffer, replay alongside current batch
  DERPP                  — Dark Experience Replay++: also stores logits for distillation
  AGEM                   — Averaged GEM: projects gradients so past task loss doesn't rise
  GDumb                  — Ignores the stream; trains from scratch on a balanced buffer
  iCaRL                  — Herding exemplars + nearest-mean-of-exemplars at inference

Dataset : ImageNet-R  (5 tasks of 40 classes each)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import ExperienceReplay, DERPP, AGEM, GDumb, iCaRL

from _setup import make_benchmark, make_model, DEVICE

NUM_TASKS = 5   # use 10 for a full experiment
MEMORY    = 500  # total exemplar budget shared across all tasks


def run(strategy, label):
    benchmark = make_benchmark(num_tasks=NUM_TASKS)
    scenario  = ClassIncremental(benchmark, batch_size=128, num_workers=4,
                                 pin_memory=(DEVICE == "cuda"))
    model     = make_model()

    results = Trainer(
        model, scenario, strategy,
        optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
        epochs_per_task=1, device=DEVICE, verbose=False,
    ).fit()

    print(f"  {label:<22}  "
          f"acc={results['avg_accuracy']:5.1f}%  "
          f"forgetting={results['avg_forgetting']:5.1f}%")


print(f"\nReplay strategies (memory={MEMORY}, {NUM_TASKS} tasks, 1 epoch each):\n")

# ER — simplest replay; reservoir sampling keeps the buffer representative
run(ExperienceReplay(memory_size=MEMORY, replay_batch_ratio=0.5), "ExperienceReplay")

# DERPP — also stores model logits at insertion time and distills from them
run(DERPP(memory_size=MEMORY, alpha=0.1, beta=0.5), "DERPP")

# AGEM — projects the gradient so loss on memory samples never increases
run(AGEM(memory_size=MEMORY, ref_sample_size=128), "AGEM")

# GDumb — extreme simplicity: ignore stream ordering, retrain from memory only
run(GDumb(memory_size=MEMORY, replay_epochs=1), "GDumb")

# iCaRL — herding exemplars, BCE + knowledge distillation, NMC classifier
run(iCaRL(exemplars_per_class=10, temperature=2.0, kd_weight=1.0), "iCaRL")

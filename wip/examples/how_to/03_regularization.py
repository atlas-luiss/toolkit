"""
HOW-TO: Regularization Strategies
===================================
Regularization strategies protect against forgetting by penalising large
changes to parameters that were important for previous tasks.


"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer
from ctk.scenarios import ClassIncremental
from ctk.strategies import EWC, EWCOnline, LwF, MAS

from _setup import make_benchmark, make_model, DEVICE

NUM_TASKS = 5   # use 10 for a full experiment


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

    print(f"  {label:<18}  "
          f"acc={results['avg_accuracy']:5.1f}%  "
          f"forgetting={results['avg_forgetting']:5.1f}%")


print(f"\nRegularization strategies ({NUM_TASKS} tasks, 1 epoch each):\n")

# EWC — lam controls how strongly parameters are penalised
run(EWC(lam=500.0, fisher_samples=200), "EWC")

# EWCOnline — same penalty but Fisher decays; lighter memory footprint
run(EWCOnline(lam=500.0, gamma=0.9, fisher_samples=200), "EWCOnline")

# LwF — distills soft labels from the previous model; no Fisher needed
run(LwF(temperature=2.0, lam=0.5), "LwF")

# MAS — importance based on output sensitivity; no labels needed
run(MAS(lam=1.0, importance_samples=200), "MAS")

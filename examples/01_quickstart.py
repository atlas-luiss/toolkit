"""Quick start: data + model + scenario + strategy, one call.

    python examples/01_quickstart.py
    # same from the command line:
    python -m cl_toolkit --data cifar100 --root ./data/cifar100 --strategy er --set memory_size=2000

Logs, config and results are written to runs/<timestamp>_cifar100_er/.
"""
import os

import cl_toolkit as clt

ROOT = os.environ.get("CL_DATA", "./data")

results = clt.run(
    data="cifar100", root=f"{ROOT}/cifar100",   # or "imagenet-r", "cub200", "hf:fancyzhx/ag_news", an ImageFolder path
    model="vit_base_patch16_224.augreg_in21k",             # any timm model, "hf:<name>", or your own nn.Module
    scenario="class",                           # "class" | "task" | "domain"
    strategy="er",                              # print(clt.available()) for every option
    strategy_kwargs={"memory_size": 2000},
    num_tasks=10, epochs=1, batch_size=128,
)
print(f"final accuracy {results['avg_accuracy']:.2f}%, forgetting {results['avg_forgetting']:.2f}")

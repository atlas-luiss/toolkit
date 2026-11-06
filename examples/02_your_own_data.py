"""Point the toolkit at your own data.

Datasets only need to return (x, y) with integer y. x can be a tensor, or a dict
of tensors (tokenised text) - see 07_any_model.py.
"""
import torch
from torch.utils.data import TensorDataset

import cl_toolkit as clt
from cl_toolkit.benchmarks import CLBenchmark, TaskSplit, few_shot, split_by_class, split_by_domain


def toy(n_per_class=30, n_classes=20, shift=0.0, seed=0):        # stand-in for your dataset
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    x = torch.randn(len(y), 3, 32, 32, generator=g) * 0.5 + y.view(-1, 1, 1, 1) / n_classes + shift
    return TensorDataset(x, y)


train, test = toy(seed=0), toy(seed=1)

bench = split_by_class(train, test, num_tasks=4)                                  # class-incremental
bench_b10 = split_by_class(train, test, num_tasks=3, initial_classes=10)          # big first task
bench_fscil = few_shot(bench_b10, k=5, skip_first=True)                           # few-shot sessions
bench_domain = split_by_domain([(toy(shift=s, seed=0), toy(shift=s, seed=1)) for s in (0.0, 1.0, 2.0)])
bench_manual = CLBenchmark([TaskSplit(0, [0, 1], train, test)], total_classes=20)  # fully explicit
print(bench, bench_b10, bench_fscil, bench_domain, sep="\n")

# An image folder works directly:  clt.run("/path/to/folder", ...)
# A (train, test) pair works directly too:
model = clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, img_size=32)
clt.run((train, test), model=model, strategy="er", strategy_kwargs={"memory_size": 100},
        num_tasks=4, epochs=2, batch_size=32, num_workers=0, freeze_backbone=False)

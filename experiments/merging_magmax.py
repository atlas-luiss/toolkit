"""MagMax setting (Marczak et al., ECCV 2024) on Split CIFAR-100 / 10 tasks, and every merging method
on the *same* sequential fine-tuning trajectory.

Setting (paper Sec. 5 + authors' code): CLIP ViT-B/16 (openai), frozen zero-shot text head,
full image-encoder fine-tuning, training logits restricted to the current task's classes,
10 epochs/task, AdamW lr 1e-5 wd 0.1, batch 128, cosine schedule, grad-norm clip 1.0, each task
starts from the previous fine-tuned weights. Evaluation: task-agnostic, all seen classes.
Paper (Tab. 2, CIFAR100/10): MagMax 80.41, TIES 77.23, Avg 77.04.
Known differences: no LR warm-up (authors: 500 steps); class order from our seed.

Usage: python experiments/merging_magmax.py <seed>     logs: runs/merging/magmax_cifar100_s<seed>.out, results in .../merges.json
"""
import json
import os
import sys

import numpy as np
import torch

import cl_toolkit as clt
from cl_toolkit import merging
from cl_toolkit.callbacks import Callback, GradientClipping
from cl_toolkit.strategies import NaiveStrategy
from cl_toolkit.trainers.cl_trainer import compute_cl_metrics

seed = int(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10          # 2nd arg only for smoke tests
out = f"runs/merging/magmax_cifar100_s{seed}" + ("" if epochs == 10 else f"_ep{epochs}_smoke")

# name -> (method, kwargs). Library defaults, plus the MagMax paper's TIES / Avg baselines.
METHODS = {
    "average (1/N)": ("average", {}),
    "task arithmetic (0.3)": ("task_arithmetic", {}),
    "ties (lam 1)": ("ties", {}),
    "ties (lam 0.55, MagMax paper)": ("ties", {"lam": 0.55}),
    "dare + TA": ("dare", {}),
    "breadcrumbs": ("breadcrumbs", {}),
    "consensus TA": ("consensus", {}),
    "magmax (0.5)": ("magmax", {}),
    "tsv": ("tsv", {}),
    "iso-c": ("iso_c", {}),
    "iso-cts": ("iso_cts", {}),
}


class SaveFinetuned(Callback):
    def __init__(self):
        self.base, self.ft = None, []

    def on_train_begin(self, trainer, **kw):
        self.base = {n: p.detach().clone() for n, p in trainer.model._backbone.named_parameters()}

    def on_task_end(self, trainer, task_id, task_metrics, **kw):
        self.ft.append({n: p.detach().clone() for n, p in trainer.model._backbone.named_parameters()})


saver = SaveFinetuned()
tr = clt.build("cifar100", root="/scratch/eric/data/cifar100", data_kwargs={"download": False},
               model="open_clip:ViT-B-16/openai", scenario="class", strategy=NaiveStrategy(train_mask="current"),
               classifier="clip-zeroshot", freeze_backbone=False, num_tasks=10, epochs=epochs, lr=1e-5,
               optimizer_cls=torch.optim.AdamW, weight_decay=0.1, scheduler="cosine", batch_size=128,
               num_workers=8, seed=seed, callbacks=[GradientClipping(1.0), saver])
os.makedirs(out, exist_ok=True)
r = tr.fit()                                         # full log: runs/merging/magmax_cifar100_s<seed>.out
json.dump(r["config"], open(f"{out}/config.json", "w"), indent=2, default=str)
T = tr.scenario.num_tasks
results = {"sequential fine-tuning (no merge)": {k: r[k] for k in ("avg_accuracy", "avg_incremental_accuracy",
                                                                    "avg_forgetting")}}
results["sequential fine-tuning (no merge)"]["accuracy_matrix"] = np.round(r["accuracy_matrix"], 2).tolist()

params = dict(tr.model._backbone.named_parameters())
for name, (method, kw) in METHODS.items():
    acc = np.full((T, T), np.nan)
    for t in range(T):
        tvs = [merging.task_vector(saver.ft[i], saver.base) for i in range(t + 1)]
        merged = merging.apply(saver.base, merging.merge(tvs, method, **kw))
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(merged[n])
        acc[t, : t + 1] = tr.evaluate_seen(t)
    m = compute_cl_metrics(acc)
    results[name] = {**{k: m[k] for k in ("avg_accuracy", "avg_incremental_accuracy", "avg_forgetting")},
                     "accuracy_matrix": np.round(acc, 2).tolist()}
    print(f"MERGE {name:32s} final {m['avg_accuracy']:6.2f}  avg-incr {m['avg_incremental_accuracy']:6.2f}", flush=True)

json.dump(results, open(f"{out}/merges.json", "w"), indent=2)
print(f"\nCLIP ViT-B/16, Split CIFAR-100/10, seed {seed}  (paper: MagMax 80.41, TIES 77.23, Avg 77.04)")
for k, v in results.items():
    print(f"  {k:36s} final {v['avg_accuracy']:6.2f}  avg-incr {v['avg_incremental_accuracy']:6.2f}")

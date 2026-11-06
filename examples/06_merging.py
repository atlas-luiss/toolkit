"""Model merging for CL: fine-tune per task, merge task vectors.

clt.merging has the primitives (task_vector, average, task_arithmetic, ties, dare, magmax);
MergeAfterTask applies them inside the CL loop.
"""
import os

import cl_toolkit as clt
from cl_toolkit import merging
from cl_toolkit.strategies import MergeAfterTask, NaiveStrategy, LoRAAdapter

ROOT = os.environ.get("CL_DATA", "./data")
common = dict(root=f"{ROOT}/cifar100", num_tasks=10, epochs=1, batch_size=64)

# MagMax (Marczak et al. 2024): sequential full fine-tuning, max-magnitude merge
clt.run(data="cifar100", strategy=MergeAfterTask(NaiveStrategy(), method="magmax", restart="finetuned"),
        freeze_backbone=False, lr=1e-5, **common)

# TIES on independently fine-tuned AdaptFormer adapters (each task starts from the PTM)
model = clt.models.ViTModel("vit_base_patch16_224.augreg_in21k", freeze_backbone=True)
clt.adapters.add_adaptformer(model, bottleneck=64)
clt.run(data="cifar100", model=model,
        strategy=MergeAfterTask(NaiveStrategy(), method="ties", restart="base", density=0.2), **common)

# The primitives work on any {name: tensor} dicts:
a, b = {"w": model.vit.blocks[0].mlp.ctk_adapter["default"][0].weight.detach()}, \
       {"w": model.vit.blocks[1].mlp.ctk_adapter["default"][0].weight.detach()}
print("cos(a, b) =", merging.similarity(a, b))
merged = merging.dare([a, b], p=0.9, base=merging.ties, density=1.0)

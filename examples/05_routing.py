"""Routing: one expert per task (LoRA / AdaptFormer / prompts), a router picks the expert per input.

Each evaluation logs routing_accuracy: how often the router picked the right task's expert.

    python examples/05_routing.py
    python -m cl_toolkit --data cifar100 --strategy routed --experts lora --router hide
"""
import os

import cl_toolkit as clt

ROOT = os.environ.get("CL_DATA", "./data")
common = dict(data="cifar100", root=f"{ROOT}/cifar100", num_tasks=10, epochs=1, batch_size=128)

for experts in ["lora", "adaptformer", "prompt"]:
    for router in ["hide", "lda", "sprompts", "keyquery", "expertgate"]:
        r = clt.run(experts=experts, router=router, **common)
        print(f"{experts:12s} {router:11s} acc={r['avg_accuracy']:.2f} routing={r['final_routing_accuracy']:.1f}")

# Choose where LoRA experts go (layer names / regex / "all-linear", and block indices):
clt.run(experts="lora", router="hide", lora_targets=["qkv", "fc1"], lora_layers=[9, 10, 11], **common)

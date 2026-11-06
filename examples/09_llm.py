"""Continual learning with a decoder LLM (Qwen2.5-0.5B) on a Hugging Face text stream.

The LLM is a frozen feature extractor (last-token pooling) and its tokenizer is the data
transform. Every strategy, head, LoRA placement and router used for ViTs works unchanged.
This script runs a small set of baselines and prints one comparison table.

    python examples/09_llm.py                       # all baselines
    python examples/09_llm.py lora-er routed-lora   # a subset

Logs: runs/llm_examples/<name>/{log.txt,config.json,results.json}

The same from the command line:
    python -m cl_toolkit --data hf:fancyzhx/ag_news --model hf:Qwen/Qwen2.5-0.5B-Instruct \
        --num-tasks 2 --strategy lora-er --set memory_size=200 \
        --lora-targets q_proj,v_proj --lora-layers 20,21,22,23 \
        --data-set max_train_per_class=1000 max_test_per_class=300
"""
import os
import sys

import cl_toolkit as clt

MODEL = os.environ.get("CL_LLM", "hf:Qwen/Qwen2.5-0.5B-Instruct")
DATA = "hf:fancyzhx/ag_news"                     # 4 topics -> 2 tasks of 2 classes
COMMON = dict(data=DATA, model=MODEL, scenario="class", num_tasks=2, batch_size=16,
              data_kwargs={"max_train_per_class": 1000, "max_test_per_class": 300})
LORA = dict(lora_targets=["q_proj", "v_proj"], lora_layers=[20, 21, 22, 23])   # last 4 of 24 blocks

BASELINES = {
    # no training of the LLM: closed-form heads on frozen features
    "ncm":         dict(strategy="frozen", classifier="ncm", epochs=0),
    "lda":         dict(strategy="frozen", classifier="lda", epochs=0),
    "ranpac":      dict(strategy="frozen", classifier="ranpac", epochs=0),
    # a trained linear head on frozen features
    "naive":       dict(strategy="naive", epochs=1, lr=1e-3),
    "er-ace":      dict(strategy="er-ace", strategy_kwargs={"memory_size": 200}, epochs=1, lr=1e-3),
    # LoRA inside the LLM
    "lora":        dict(strategy="lora", epochs=1, lr=2e-4, **LORA),
    "lora-er":     dict(strategy="lora-er", strategy_kwargs={"memory_size": 200}, epochs=1, lr=2e-4, **LORA),
    # one LoRA expert per task, routed by an LDA router on the frozen features
    "routed-lora": dict(strategy="routed", experts="lora", router="lda", epochs=1, lr=2e-4, **LORA),
}


def main(names):
    rows = []
    for name in names:
        r = clt.run(**COMMON, **BASELINES[name], out_dir=f"runs/llm_examples/{name}")
        rows.append((name, r["avg_accuracy"], r["avg_forgetting"], r.get("final_routing_accuracy")))
    print(f"\n{MODEL} on {DATA}, Class-IL, 2 tasks")
    print(f"{'baseline':13s} {'final acc':>9s} {'forgetting':>10s} {'routing':>8s}")
    for name, acc, fgt, route in rows:
        print(f"{name:13s} {acc:9.2f} {fgt:10.2f} {'' if route is None else f'{route:8.2f}'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(BASELINES))

"""Command line: ``python -m cl_toolkit`` (or ``cl-toolkit``).

    cl-toolkit --list
    cl-toolkit --data cifar100 --root ./data/cifar100 --strategy er --set memory_size=2000
    cl-toolkit --data cifar100 --strategy routed --experts lora --router hide
    cl-toolkit --data cifar100 --strategy frozen --classifier ranpac --epochs 0
    cl-toolkit --data hf:fancyzhx/ag_news --model hf:bert-base-uncased --num-tasks 2 --strategy er
    cl-toolkit --data cifar100 --strategy lora --lora-targets qkv,fc1 --lora-layers 9,10,11

Every run writes log.txt, config.json and results.json to --out (default: runs/<timestamp>_<data>_<strategy>).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set/--data-set/--classifier-set expect key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = _value(v)
    return out


def _csv(s, cast=str):
    return None if s is None else [cast(x) for x in s.split(",") if x]


def main(argv=None):
    p = argparse.ArgumentParser(prog="cl-toolkit", description="Run a continual learning experiment.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--list", action="store_true", help="print every accepted name and exit")
    p.add_argument("--data", help="cifar100 | imagenet-r | cub200 | hf:<dataset> | <image folder> ...")
    p.add_argument("--root", help="dataset directory (default ./data/<name>)")
    p.add_argument("--preprocess", choices=["model", "pilot", "coda"],
                   help="image preprocessing: the model's own (default) or a code base's (see benchmarks/transforms.py)")
    p.add_argument("--model", default="vit_base_patch16_224.augreg_in21k", help="mlp (small, untrained; pairs with --data toy) | timm name | hf:<model> | open_clip:<arch>/<tag>")
    p.add_argument("--scenario", default="class", choices=["class", "task", "domain"])
    p.add_argument("--strategy", default="naive")
    p.add_argument("--classifier", help="linear | cosine | ncm | fecam | lda | klda | ranpac | slca | wa")
    p.add_argument("--experts", help="routing experts: lora | adaptformer | prompt")
    p.add_argument("--router", help="routing: hide | sprompts | keyquery | expertgate | <head name>")
    p.add_argument("--num-tasks", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None, help="default: 8 (0 for in-memory data such as toy)")
    p.add_argument("--train-mask", choices=["none", "seen", "current"])
    p.add_argument("--eval-mask", choices=["none", "seen"])
    p.add_argument("--lora-targets", help="comma list (qkv,fc1), a regex, or all-linear")
    p.add_argument("--lora-layers", help="comma list of block indices, e.g. 9,10,11")
    p.add_argument("--finetune-backbone", action="store_true", help="do not freeze the backbone")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--set", nargs="*", metavar="K=V", help="strategy arguments, e.g. memory_size=2000")
    p.add_argument("--classifier-set", nargs="*", metavar="K=V", help="classifier head arguments")
    p.add_argument("--data-set", nargs="*", metavar="K=V", help="dataset arguments, e.g. download=False")
    p.add_argument("--out", default="auto", help="output directory ('none' to write nothing)")
    a = p.parse_args(argv)

    import cl_toolkit as clt
    if a.list:
        for k, v in clt.available().items():
            print(f"{k:11s} {', '.join(map(str, v))}")
        return 0
    if not a.data:
        p.error("--data is required (or use --list)")

    targets = a.lora_targets                       # "qkv,fc1" -> list | "all-linear" | regex | "qkv" -> ["qkv"]
    if targets and targets != "all-linear":
        if "," in targets:
            targets = _csv(targets)
        elif not any(ch in targets for ch in ".*\\[]()|+?^$"):
            targets = [targets]
    results = clt.run(
        a.data, a.model, a.scenario, a.strategy, root=a.root, num_tasks=a.num_tasks, epochs=a.epochs, lr=a.lr,
        batch_size=a.batch_size, num_workers=a.num_workers, classifier=a.classifier, experts=a.experts,
        router=a.router, train_mask=a.train_mask, eval_mask=a.eval_mask, lora_targets=targets,
        lora_layers=_csv(a.lora_layers, int), freeze_backbone=False if a.finetune_backbone else None, seed=a.seed,
        device=a.device, strategy_kwargs=_kv(a.set), classifier_kwargs=_kv(a.classifier_set),
        data_kwargs={**_kv(a.data_set), **({"preprocess": a.preprocess} if a.preprocess else {})}, out_dir=None if a.out == "none" else a.out,
        **({"precision": a.precision} if a.precision else {}))
    print(json.dumps({k: results[k] for k in ("avg_accuracy", "avg_incremental_accuracy", "avg_forgetting")
                      if k in results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

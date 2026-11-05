"""One-call experiments: ``clt.run(data, model, scenario, strategy)``.

Each of the four pieces is a short string *or* your own object, so you can swap
in a custom dataset / model / strategy one at a time. ``clt.build`` returns the
trainer without running it; ``clt.available()`` lists every name.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import sys
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from cl_toolkit.benchmarks import CLBenchmark, get_benchmark, split_by_class
from cl_toolkit.models.base import CLModel
from cl_toolkit.scenarios import Scenario
from cl_toolkit.strategies import NaiveStrategy, Strategy, get_strategy
from cl_toolkit.trainers.cl_trainer import ContinualTrainer
from cl_toolkit.utils import set_seed


# ViT-B/16 pre-trained on ImageNet-21k: the "ViT-B/16-IN21K" of L2P, DualPrompt, RanPAC, SLCA, HiDe, FeCAM.
DEFAULT_MODEL = "vit_base_patch16_224.augreg_in21k"


# ----------------------------------------------------------------------------- pieces
def build_model(model: Union[str, nn.Module], freeze_backbone: Optional[bool] = None, device=None) -> CLModel:
    """``"vit_base_patch16_224"`` (any timm name) | ``"hf:bert-base-uncased"`` (Hugging Face,
    encoders and decoder LLMs) | ``"open_clip:ViT-B-32/openai"`` | ``"mlp"`` (small, untrained) |
    a CLModel | any nn.Module encoder.

    ``freeze_backbone=None`` (default): frozen, except ``"mlp"``, which is untrained and so trained."""
    if freeze_backbone is None:
        freeze_backbone = not (isinstance(model, str) and model == "mlp")
    if isinstance(model, str) and model == "mlp":
        from cl_toolkit.models.generic import mlp
        m = mlp(device=device)
        if freeze_backbone:
            m.freeze_backbone()
        return m
    if isinstance(model, CLModel):
        # freeze_backbone applies to passed-in models too, in both directions (e.g. HFModel and
        # FeatureModel are built frozen by default; freeze_backbone=False fine-tunes them)
        if freeze_backbone:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        return model
    if isinstance(model, nn.Module):
        from cl_toolkit.models.generic import wrap
        return wrap(model, freeze_backbone=freeze_backbone, device=device)
    if model.startswith("hf:"):
        from cl_toolkit.models.generic import HFModel
        return HFModel(model[3:], freeze_backbone=freeze_backbone, device=device)
    if model.startswith("open_clip:"):
        from cl_toolkit.models.clip import CLIPModel
        arch, _, pretrained = model[len("open_clip:"):].partition("/")
        return CLIPModel(arch, pretrained or "openai", freeze_backbone=freeze_backbone, device=device)
    from cl_toolkit.models.vit import ViTModel
    return ViTModel(model, pretrained=True, freeze_backbone=freeze_backbone, device=device)


def build_benchmark(data, root: Optional[str], num_tasks: int, model: Optional[CLModel] = None,
                    seed: int = 42, **kw) -> CLBenchmark:
    """A name ("cifar100", "imagenet-r", "hf:fancyzhx/ag_news", ...), a folder, ``(train, test)``
    datasets, or a CLBenchmark. Named datasets use the model's own preprocessing."""
    if isinstance(data, CLBenchmark):
        return data
    if isinstance(data, (tuple, list)) and len(data) == 2:
        bench = split_by_class(data[0], data[1], num_tasks, seed=seed, **kw)
        if all(isinstance(d, torch.utils.data.TensorDataset) for d in data):
            bench.default_num_workers = 0            # already in memory
        return bench
    if isinstance(data, str):
        from cl_toolkit.benchmarks.transforms import LOW_RES
        preset = kw.pop("preprocess", "model")
        if "transforms" not in kw and model is not None:
            if hasattr(model, "preprocess_config"):          # image backbone: preset + its normalisation
                kw["transforms"] = model.transforms(preset, low_res=data.lower() in LOW_RES)
            else:
                try:
                    kw["transforms"] = model.transforms()    # e.g. a tokenizer
                except AttributeError:
                    pass
        bench = get_benchmark(data, root, num_tasks, seed=seed, **kw)
        bench.preprocess = preset if "transforms" in kw else "dataset default"
        return bench
    raise TypeError(f"data must be a name, a folder, (train, test) datasets or a CLBenchmark; got {type(data)}")


def build_strategy(strategy: Union[str, Strategy], model: CLModel, **kw) -> Strategy:
    if isinstance(strategy, Strategy):
        return strategy
    from cl_toolkit.prompts import PROMPTS
    key = strategy.lower().replace("_", "-")
    if key in PROMPTS:
        if not hasattr(model, "attach_prompts"):
            raise TypeError(f"strategy {strategy!r} inserts prefix prompts into timm ViT attention; "
                            f"{type(model).__name__} is not a timm ViT (use e.g. model='vit_base_patch16_224')")
        model.freeze_backbone()
        model.attach_prompts(PROMPTS[key](**kw))
        return NaiveStrategy(train_mask="current")
    return get_strategy(key, **kw)


def build(
    data,
    model: Union[str, nn.Module] = DEFAULT_MODEL,
    scenario: Union[str, Scenario] = "class",
    strategy: Union[str, Strategy] = "naive",
    *,
    root: Optional[str] = None,
    num_tasks: int = 10,
    classifier=None,
    experts: Optional[str] = None,
    router: Optional[str] = None,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    data_kwargs: Optional[Dict[str, Any]] = None,
    freeze_backbone: Optional[bool] = None,
    lora_targets=None,
    lora_layers=None,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: Optional[int] = None,
    train_mask: Optional[str] = None,
    eval_mask: Optional[str] = None,
    seed: int = 42,
    device=None,
    trainer_cls=ContinualTrainer,
    **trainer_kw,
) -> ContinualTrainer:
    """Assemble everything and return a ready-to-``fit()`` trainer. Arguments as in :func:`run`."""
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = build_model(model, freeze_backbone, device)
    if lora_targets is not None:
        m.lora_targets = lora_targets          # default placement for every LoRA-based method
    if lora_layers is not None:
        m.lora_layers = list(lora_layers)
    bench = build_benchmark(data, root, num_tasks, m, seed, **(data_kwargs or {}))
    layer = "cosine" if classifier in ("cosine", "clip-zeroshot") else "linear"
    if num_workers is None:      # loader worker processes only pay off for data read from disk
        num_workers = getattr(bench, "default_num_workers", 8)
    scen = scenario if isinstance(scenario, Scenario) else Scenario(bench, scenario, batch_size=batch_size,
                                                                   num_workers=num_workers, head=layer)
    skw = dict(strategy_kwargs or {})
    if experts is not None or router is not None:            # routing shorthand
        if strategy == "naive":
            strategy = "routed"
        skw.setdefault("experts", experts or "lora")
        skw.setdefault("router", router or "hide")
    strat = build_strategy(strategy, m, **skw)
    if classifier not in (None, "linear", "cosine"):
        from cl_toolkit.heads import Head, get_head
        from cl_toolkit.strategies import WithHead
        head = classifier if isinstance(classifier, Head) else get_head(classifier, **(classifier_kwargs or {}))
        from cl_toolkit.strategies import RoutedExperts
        if isinstance(strat, RoutedExperts):          # one head per expert, on that expert's features
            from cl_toolkit.strategies.routed import _check_head
            strat.head = _check_head(head)
        else:
            strat = WithHead(strat, head)
    if train_mask is not None:
        strat.train_mask = train_mask
    if eval_mask is not None:
        strat.eval_mask = eval_mask
    return trainer_cls(m, scen, strat, lr=lr, epochs_per_task=epochs, device=device, **trainer_kw)


# ----------------------------------------------------------------------------- run
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _name(x) -> str:
    if isinstance(x, str):
        return x.replace("/", "-").replace(":", "-")
    if isinstance(x, (tuple, list)):
        return "custom"
    return getattr(x, "name", None) if isinstance(getattr(x, "name", None), str) else type(x).__name__


_RUN_FILES = ("log.txt", "config.json", "results.json")


def _archive_previous_run(out_dir: str) -> Optional[str]:
    """Move an earlier run's files out of the way (never deleted), so logs are never mixed."""
    old = [f for f in _RUN_FILES if os.path.exists(os.path.join(out_dir, f))]
    if not old:
        return None
    t = os.path.getmtime(os.path.join(out_dir, old[0]))
    dest = os.path.join(out_dir, "previous_" + datetime.datetime.fromtimestamp(t).strftime("%Y%m%d-%H%M%S"))
    while os.path.exists(dest):
        dest += "_"
    os.makedirs(dest)
    for f in old:
        os.replace(os.path.join(out_dir, f), os.path.join(dest, f))
    return dest


def run(data, model=DEFAULT_MODEL, scenario="class", strategy="naive", *,
        out_dir: Optional[str] = "auto", save: Optional[str] = None, **kw) -> dict:
    """Build, train, evaluate. Returns the results dict.

    Args:
        data: "cifar100", "cifar10", "imagenet-r", "imagenet-a", "cub200", "hf:<text dataset>",
            an ImageFolder path, a ``(train_ds, test_ds)`` pair, or a :class:`CLBenchmark`.
        model: timm name, "hf:<name>", "open_clip:<arch>/<tag>", a CLModel, or any nn.Module encoder.
        scenario: "class" | "task" | "domain" (or a :class:`Scenario`).
        strategy: a name from ``clt.available()["strategies"]`` or a :class:`Strategy`.
        classifier: "linear" | "cosine" | a head name ("ncm", "fecam", "lda", "klda", "ranpac", "slca", "wa").
        experts / router: routing shorthand, e.g. ``experts="lora", router="hide"``.
        lora_targets / lora_layers: where any LoRA-based method puts adapters, e.g.
            ``lora_targets=["qkv", "fc1"], lora_layers=[9, 10, 11]`` (or a regex / "all-linear").
            ``clt.adapters.lora_candidates(model)`` lists the layer names.
        out_dir: where ``log.txt``, ``config.json`` and ``results.json`` go. "auto" (default):
            ``runs/<timestamp>_<data>_<strategy>``. None: write nothing.
        save: extra path for the results JSON.
        **kw: root, num_tasks, epochs, lr, batch_size, num_workers, strategy_kwargs,
            classifier_kwargs, data_kwargs, train_mask, eval_mask, freeze_backbone, seed,
            device, precision, scheduler, callbacks, verbose, trainer_cls.
    """
    if out_dir == "auto":
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join("runs", f"{stamp}_{_name(data)}_{_name(strategy)}")
    ctx, log = contextlib.nullcontext(), None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        moved = _archive_previous_run(out_dir)
        log = open(os.path.join(out_dir, "log.txt"), "w")
        ctx = contextlib.redirect_stdout(_Tee(sys.stdout, log))
    try:
        with ctx:
            if out_dir:
                print(f"[cl_toolkit] logging to {os.path.abspath(out_dir)}", flush=True)
                if moved:
                    print(f"[cl_toolkit] previous run in this folder moved to {moved}", flush=True)
            trainer = build(data, model, scenario, strategy, **kw)
            if out_dir:
                cfg = {"data": _name(data), "model": _name(model), "scenario": _name(scenario),
                       "strategy": _name(strategy), **{k: v for k, v in kw.items()
                                                        if isinstance(v, (str, int, float, bool, dict, type(None)))},
                       "resolved": trainer.config()}
                with open(os.path.join(out_dir, "config.json"), "w") as f:
                    json.dump(cfg, f, indent=2, default=str)
            results = trainer.fit()
            if out_dir:
                trainer.save_results(os.path.join(out_dir, "results.json"))
                print(f"[cl_toolkit] results: {os.path.abspath(os.path.join(out_dir, 'results.json'))}", flush=True)
    finally:                                  # a crashed run still leaves its full log
        if log is not None:
            log.close()
    if save:
        trainer.save_results(save)
    results["out_dir"] = out_dir
    return results


# ----------------------------------------------------------------------------- discovery
def available() -> Dict[str, list]:
    """Every name accepted by :func:`run`, grouped by argument."""
    from cl_toolkit.benchmarks import DATASETS
    from cl_toolkit.heads import HEADS
    from cl_toolkit.prompts import PROMPTS
    from cl_toolkit.routing import EXPERTS, ROUTERS
    from cl_toolkit.scenarios import KINDS
    from cl_toolkit.strategies import STRATEGIES
    return {
        "data": sorted(DATASETS) + ["hf:<dataset>", "<image folder>", "(train_ds, test_ds)"],
        "model": ["mlp", "<timm name>", "hf:<model>", "open_clip:<arch>/<tag>", "<nn.Module>"],
        "scenario": list(KINDS),
        "strategy": sorted(set(STRATEGIES) | set(PROMPTS)),
        "classifier": ["linear", "cosine"] + sorted(HEADS),
        "experts": sorted(EXPERTS),
        "router": sorted(set(ROUTERS) - {"head"}) + ["<any classifier head name>"],
    }

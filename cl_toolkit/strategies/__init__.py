"""Continual learning strategies.

Families:
  baselines     NaiveStrategy
  replay        ExperienceReplay, ERACE, DER, DERPP, AGEM, iCaRL, GDumb
  regularize    EWC, EWCOnline, MAS, LwF
  PEFT          LoRAAdapter (wraps any strategy), LoRAStrategy, LoRAReplay
  pre-trained   FirstSessionAdaptation, SimpleCIL, RanPAC
  heads         WithHead(any_strategy, head) with heads from cl_toolkit.heads
  routing       RoutedExperts / Routed(experts, router) with cl_toolkit.routing
  merging       MergeAfterTask (wraps any strategy; any cl_toolkit.merging method),
                ModelAveraging (CoMA / CoFiMA)
  prompts       see cl_toolkit.prompts (L2P, DualPrompt, CODA-Prompt): attach to the model,
                train with NaiveStrategy(train_mask="current")
"""

from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.naive import NaiveStrategy, Naive
from cl_toolkit.strategies.replay import (ExperienceReplay, ERACE, DERPP, DER, AGEM, iCaRL, GDumb,
                                   ReservoirBuffer, ClassBalancedBuffer, HerdingBuffer)
from cl_toolkit.strategies.regularization import EWC, EWCOnline, LwF, MAS
from cl_toolkit.strategies.peft import LoRAAdapter, LoRAStrategy, LoRAReplay
from cl_toolkit.strategies.ptm import FirstSessionAdaptation, SimpleCIL, RanPAC, ProtoNet
from cl_toolkit.strategies.with_head import WithHead
from cl_toolkit.strategies.routed import RoutedExperts, Routed
from cl_toolkit.strategies.merge import MergeAfterTask, ModelAveraging

# name -> factory(**kwargs). Used by cl_toolkit.run(strategy="...").
STRATEGIES = {
    "naive": NaiveStrategy, "finetune": NaiveStrategy,
    "er": ExperienceReplay, "replay": ExperienceReplay, "er-ace": ERACE,
    "der": DER, "derpp": DERPP, "der++": DERPP,
    "agem": AGEM, "icarl": iCaRL, "gdumb": GDumb,
    "ewc": EWC, "ewc-online": EWCOnline, "mas": MAS, "lwf": LwF,
    "lora": LoRAStrategy, "lora-er": LoRAReplay,
    "simplecil": SimpleCIL, "ranpac": RanPAC, "frozen": FirstSessionAdaptation,
    "first-session": FirstSessionAdaptation,
    "routed": Routed,
    "hide-prompt": lambda **kw: Routed("prompt", "hide", **kw),
    "s-prompts": lambda **kw: Routed("prompt", "sprompts", **kw),
    "expert-gate": lambda **kw: Routed("adaptformer", "expertgate", **kw),
    **{name: (lambda m: lambda **kw: MergeAfterTask(NaiveStrategy(), method=m, **kw))(method)
       for name, method in {"magmax": "magmax", "ties": "ties", "task-arithmetic": "task_arithmetic",
                            "dare": "dare", "breadcrumbs": "breadcrumbs", "consensus": "consensus",
                            "tsv": "tsv", "iso-c": "iso_c", "iso-cts": "iso_cts",
                            "weight-average": "average"}.items()},
    "coma": lambda **kw: ModelAveraging(NaiveStrategy(), fisher=False, **kw),
    "cofima": lambda **kw: ModelAveraging(NaiveStrategy(), fisher=True, **kw),
}


def get_strategy(name: str, **kwargs) -> Strategy:
    key = name.lower().replace("_", "-")
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)} "
                       "(prompt methods: 'l2p', 'dualprompt', 'coda-prompt' via cl_toolkit.run)")
    return STRATEGIES[key](**kwargs)


__all__ = ["Strategy", "NaiveStrategy", "Naive", "ExperienceReplay", "ERACE", "DERPP", "DER", "AGEM", "iCaRL", "GDumb",
           "EWC", "EWCOnline", "LwF", "MAS", "LoRAAdapter", "LoRAStrategy", "LoRAReplay", "SimpleCIL", "RanPAC",
           "ProtoNet", "FirstSessionAdaptation", "WithHead", "RoutedExperts", "Routed", "MergeAfterTask", "ModelAveraging", "ReservoirBuffer", "ClassBalancedBuffer", "HerdingBuffer",
           "STRATEGIES", "get_strategy"]

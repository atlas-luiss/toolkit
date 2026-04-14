"""Continual learning strategies: replay, regularization, PEFT, few-shot, and merging utilities."""

from ctk.strategies.base import Strategy
from ctk.strategies.naive import NaiveStrategy
from ctk.strategies.replay import (
    ExperienceReplay,
    DERPP,
    AGEM,
    GDumb,
    iCaRL,
    ReservoirBuffer,
    HerdingBuffer,
)
from ctk.strategies.regularization import EWC, EWCOnline, LwF, MAS
from ctk.strategies.peft import LoRAAdapter, LoRAStrategy, LoRAReplay, HAM
from ctk.strategies.few_shot import ProtoNet

from ctk.strategies.merging import (
    merge_adapters_weighted,
    hierarchical_merge,
    task_arithmetic_merge,
    ties_merging,
    dare_ties_merging,
)

__all__ = [
    # New Strategy interface
    "Strategy",
    "NaiveStrategy",
    "ExperienceReplay",
    "DERPP",
    "AGEM",
    "GDumb",
    "iCaRL",
    "EWC",
    "EWCOnline",
    "LwF",
    "MAS",
    "LoRAAdapter",
    "LoRAStrategy",
    "LoRAReplay",
    "HAM",
    "ProtoNet",
    # Buffer primitives
    "ReservoirBuffer",
    "HerdingBuffer",
    # Merging utilities
    "merge_adapters_weighted",
    "hierarchical_merge",
    "task_arithmetic_merge",
    "ties_merging",
    "dare_ties_merging",
]

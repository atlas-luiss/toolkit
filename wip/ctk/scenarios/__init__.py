"""
CTK Scenarios — continual learning protocols.

Available scenarios
-------------------
ClassIncremental   Class-IL: growing output space, no task oracle.
TaskIncremental    Task-IL:  per-task heads, task identity known.
DomainIncremental  Domain-IL: fixed output space, shifting input domain.
Finetuning         Sequential fine-tuning (lower bound / baseline).
JointTraining      All data at once (upper bound).
"""

from .base import CLScenario, TaskInfo
from .class_incremental import ClassIncremental
from .domain_incremental import DomainIncremental
from .few_shot import FewShot
from .fscil import FSCIL
from .finetuning import Finetuning, JointTraining
from .task_incremental import TaskIncremental

__all__ = [
    "CLScenario",
    "TaskInfo",
    "ClassIncremental",
    "TaskIncremental",
    "DomainIncremental",
    "FewShot",
    "FSCIL",
    "Finetuning",
    "JointTraining",
]

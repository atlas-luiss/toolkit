"""CTK — Collage Toolkit (Collage Lab)."""

__version__ = "0.2.0"

from ctk.trainer import ContinualTrainer, Trainer
from ctk.evaluator import ContinualEvaluator
from ctk import models, benchmarks, scenarios, strategies, callbacks, trainers

__all__ = [
    'Trainer',
    'ContinualTrainer',
    'ContinualEvaluator',
    'models',
    'benchmarks',
    'scenarios',
    'strategies',
    'callbacks',
    'trainers',
]

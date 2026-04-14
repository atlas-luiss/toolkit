"""CTK — Collage Toolkit (Collage Lab)."""

__version__ = "0.2.0"

from ctk.trainers.cl_trainer   import ContinualTrainer, Trainer
from ctk.trainers.cl_evaluator import ContinualEvaluator
from ctk.trainers.vanilla      import StandardTrainer, VanillaTrainer
from ctk import models, benchmarks, scenarios, strategies, callbacks, trainers

__all__ = [
    # CL training
    'Trainer',
    'ContinualTrainer',
    'ContinualEvaluator',
    # Standard (non-CL) training
    'StandardTrainer',
    'VanillaTrainer',       # backward-compatible alias
    # Submodules
    'models',
    'benchmarks',
    'scenarios',
    'strategies',
    'callbacks',
    'trainers',
]

"""cl_toolkit — Continual Learning Toolkit.

Quick start::

    import cl_toolkit
    cl_toolkit.run("cifar100", root="./data", strategy="ranpac")

Building blocks (all swappable):
    cl_toolkit.benchmarks   data -> tasks         split_by_class, split_by_domain, few_shot, from_folder
    cl_toolkit.models       backbone + head        ViTModel (any timm ViT), CLIPModel, CLModel
    cl_toolkit.scenarios    protocol               Scenario(kind="class" | "task" | "domain")
    cl_toolkit.strategies   the CL method          replay, regularisation, LoRA, SimpleCIL, RanPAC, merging
    cl_toolkit.heads        classifier heads       NCM, FeCAM, LDA, KLDA, RanPAC, SLCA, WA (+ CosineLinear)
    cl_toolkit.prompts      prompt pools           L2P, DualPrompt, CODAPrompt
    cl_toolkit.routing      experts + routers      LoRA/AdaptFormer/Prompt experts; KeyQuery, SPrompts, HiDe, ExpertGate
    cl_toolkit.adapters     PEFT modules           LoRA, AdaptFormer
    cl_toolkit.merging      task-vector merging    task arithmetic, TIES, DARE, MagMax
    cl_toolkit.trainers     the loop               ContinualTrainer
"""

__version__ = "0.4.0"

from cl_toolkit import adapters, benchmarks, callbacks, heads, merging, models, prompts, routing, scenarios, strategies, trainers
from cl_toolkit.api import available, build, run
from cl_toolkit.scenarios import Scenario
from cl_toolkit.trainers import ContinualTrainer, Trainer

__all__ = ["run", "build", "available", "Scenario", "ContinualTrainer", "Trainer",
           "adapters", "benchmarks", "callbacks", "heads", "merging", "models", "prompts", "routing", "scenarios", "strategies",
           "trainers"]

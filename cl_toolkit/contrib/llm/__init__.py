"""LLM fine-tuning utilities (experimental, untested, separate from the CL core).

Causal-LM training (instruction tuning, summarization) with its own trainer.
For continual *classification* with text models, use the core API instead:
``clt.run(..., model="hf:bert-base-uncased")`` — see the README.

Requires ``pip install cl-toolkit[text]``.
"""
from .model import LLMModel
from .trainer import LLMTrainer, LLMTrainingConfig

__all__ = ["LLMModel", "LLMTrainer", "LLMTrainingConfig"]

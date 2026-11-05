"""LLMModel — HF-backed language model (causal, seq2seq, or classification) with LoRA support."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TaskType = Literal["causal_lm", "seq2seq_lm", "classification"]


class LLMModel(nn.Module):
    """Wraps any HF AutoModel* with optional LoRA via PEFT.

    Owns the tokenizer. Loss is computed by the HF model when labels= is passed to forward().
    Unlike CLModel, there is no separate encode/head split — the full LM forward pass is used.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task_type: TaskType = "causal_lm",
        num_labels: int = 2,                        # classification only
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        self.task_type = task_type
        self.model_name_or_path = model_name_or_path

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Causal LM: pad on left to preserve generation quality
        self.tokenizer.padding_side = "left" if task_type == "causal_lm" else "right"

        # Build load kwargs
        load_kwargs: dict = dict(
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            dtype=torch_dtype or (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                    bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
                )
            except Exception as e:
                logger.warning(f"Quantization not available: {e}")

        # Load model
        if task_type == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        elif task_type == "seq2seq_lm":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **load_kwargs)
        elif task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels, **load_kwargs
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type!r}")

        self._lora_applied = False
        logger.info(f"LLMModel: {model_name_or_path}  task={task_type}")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def add_lora(
        self,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ) -> "LLMModel":
        """Apply LoRA via PEFT. Returns self for chaining."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType as PeftTaskType
        except ImportError:
            raise ImportError("peft required: pip install peft")

        if self._lora_applied:
            logger.warning("LoRA already applied — skipping")
            return self

        _peft_type = {
            "causal_lm":      PeftTaskType.CAUSAL_LM,
            "seq2seq_lm":     PeftTaskType.SEQ_2_SEQ_LM,
            "classification": PeftTaskType.SEQ_CLS,
        }[self.task_type]

        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=bias,
                task_type=_peft_type,
            ),
        )
        self._lora_applied = True
        self.model.print_trainable_parameters()
        return self

    def merge_and_unload(self) -> "LLMModel":
        """Merge LoRA weights into the base model and remove adapter overhead."""
        if self._lora_applied and hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
        return self

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce GPU memory."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        # Required for PEFT + gradient checkpointing
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------

    def forward(self, **kwargs):
        """Delegate to HF model. Pass labels= to get loss in the output."""
        return self.model(**kwargs)

    def generate(self, **kwargs):
        """Delegate to HF model.generate()."""
        return self.model.generate(**kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not only_trainable or p.requires_grad)
        )

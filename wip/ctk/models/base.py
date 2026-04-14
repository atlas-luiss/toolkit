"""CLModel — abstract base for continual learning models; IncrementalClassifier; MultiHeadClassifier."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier heads
# ---------------------------------------------------------------------------

class IncrementalClassifier(nn.Module):
    """Single linear head that grows as new classes are added (Class-IL)."""

    def __init__(self, embed_dim: int, initial_classes: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self._fc: Optional[nn.Linear] = None
        if initial_classes > 0:
            self._fc = nn.Linear(embed_dim, initial_classes)

    def grow(self, num_new_classes: int) -> None:
        """Extend the output layer by num_new_classes neurons, preserving existing weights."""
        if self._fc is None:
            self._fc = nn.Linear(self.embed_dim, num_new_classes)
        else:
            old_n  = self._fc.out_features
            new_n  = old_n + num_new_classes
            device = self._fc.weight.device   # stay on whatever device _fc is on
            new_fc = nn.Linear(self.embed_dim, new_n, device=device)
            new_fc.weight.data[:old_n].copy_(self._fc.weight.data)
            new_fc.bias.data[:old_n].copy_(self._fc.bias.data)
            self._fc = new_fc

    @property
    def num_classes(self) -> int:
        return self._fc.out_features if self._fc is not None else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fc is None:
            raise RuntimeError("IncrementalClassifier has no classes yet; call grow() first")
        return self._fc(x)


class MultiHeadClassifier(nn.Module):
    """One linear head per task (Task-IL); selected by task_id at inference."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self._heads = nn.ModuleDict()

    def add_head(self, task_id: int, num_classes: int) -> None:
        self._heads[str(task_id)] = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        key = str(task_id)
        if key not in self._heads:
            raise KeyError(f"No head for task {task_id}; call add_head() first")
        return self._heads[key](x)


# ---------------------------------------------------------------------------
# CLModel
# ---------------------------------------------------------------------------

class CLModel(nn.Module, ABC):
    """Abstract base class for continual learning models.

    Subclasses implement _encode(), embed_dim, and _default_lora_targets().
    The base class handles head management and LoRA adapter lifecycle.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

        self._head: Optional[nn.Module] = None
        self._active_lora: Optional[int] = None

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of inputs into feature vectors (B, embed_dim)."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of feature embeddings produced by _encode."""

    @abstractmethod
    def _default_lora_targets(self) -> List[str]:
        """Default list of submodule names to apply LoRA to (e.g. ['qkv'])."""

    # ------------------------------------------------------------------
    # Encode / forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Public encode; subclasses implement _encode."""
        return self._encode(x)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Encode inputs, optionally select LoRA adapter, then apply head."""
        # Activate the right LoRA adapter for this task during eval
        if (self._active_lora != task_id
                and hasattr(self, "_backbone")
                and hasattr(self._backbone, "peft_config")
                and f"task_{task_id}" in self._backbone.peft_config):
            self._backbone.set_adapter(f"task_{task_id}")
            self._active_lora = task_id

        features = self.encode(x)

        if self._head is None:
            raise RuntimeError("No head set; call scenario.setup(model) first")

        if isinstance(self._head, MultiHeadClassifier):
            return self._head(features, task_id)
        return self._head(features)

    # ------------------------------------------------------------------
    # Head management
    # ------------------------------------------------------------------

    def set_head(self, head: nn.Module) -> None:
        self._head = head.to(self._device)

    def get_head(self) -> nn.Module:
        return self._head

    def grow_head(self, num_classes: int) -> None:
        """Grow an IncrementalClassifier head by num_classes neurons."""
        if not isinstance(self._head, IncrementalClassifier):
            raise TypeError("grow_head requires an IncrementalClassifier head")
        self._head.grow(num_classes)
        self._head.to(self._device)

    def add_task_head(self, task_id: int, num_classes: int) -> None:
        """Add a per-task head to a MultiHeadClassifier."""
        if not isinstance(self._head, MultiHeadClassifier):
            raise TypeError("add_task_head requires a MultiHeadClassifier head")
        self._head.add_head(task_id, num_classes)
        self._head.to(self._device)

    # ------------------------------------------------------------------
    # Parameter / freezing utilities
    # ------------------------------------------------------------------

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (non-LoRA, non-head).

        Keeps LoRA adapter weights and the classification head trainable,
        so this is safe to call before or after add_lora().
        """
        for name, param in self.named_parameters():
            if "lora" not in name.lower() and not name.startswith("_head"):
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # LoRA lifecycle
    # ------------------------------------------------------------------

    def add_lora(
        self,
        task_id: int,
        rank: int                        = 8,
        target_modules: Optional[List[str]] = None,
        lora_alpha: float                = 16.0,
        lora_dropout: float              = 0.0,
    ) -> None:
        """Attach a new LoRA adapter (named task_{task_id}) to the backbone."""
        try:
            from peft import LoraModel, LoraConfig
        except ImportError:
            raise ImportError("peft required: pip install peft")

        if not hasattr(self, "_backbone"):
            raise AttributeError(
                "CLModel subclass must store the backbone as self._backbone"
            )

        targets = target_modules or self._default_lora_targets()
        adapter_name = f"task_{task_id}"
        cfg = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=targets,
            lora_dropout=lora_dropout,
            bias="none",
        )

        if hasattr(self._backbone, "peft_config"):
            # Already a LoraModel — inject new adapter directly
            self._backbone.peft_config[adapter_name] = cfg
            self._backbone.inject_adapter(self._backbone.model, adapter_name)
        else:
            # LoraModel passes *args/**kwargs positionally — safe for vision backbones
            self._backbone = LoraModel(
                self._backbone, cfg, adapter_name=adapter_name
            )

        self._backbone.set_adapter(adapter_name)
        self._active_lora = task_id

        # Ensure new adapter weights are trainable
        for name, param in self._backbone.named_parameters():
            if adapter_name in name:
                param.requires_grad_(True)

    def freeze_lora(self, task_id: int) -> None:
        """Freeze all parameters of adapter task_{task_id}."""
        adapter_name = f"task_{task_id}"
        if not hasattr(self._backbone, "peft_config"):
            return
        for name, param in self._backbone.named_parameters():
            if adapter_name in name:
                param.requires_grad_(False)

    def eject_lora(self, task_id: int) -> Dict[str, torch.Tensor]:
        """Return a copy of adapter task_{task_id}'s weights."""
        adapter_name = f"task_{task_id}"
        weights: Dict[str, torch.Tensor] = {}
        if hasattr(self._backbone, "peft_config"):
            for name, param in self._backbone.named_parameters():
                if adapter_name in name:
                    weights[name] = param.data.clone()
        return weights

    def merge_lora(self, task_id: int) -> None:
        """Merge LoRA adapter task_{task_id} into the base weights and remove it.

        Only unwraps the LoraModel entirely when no adapters remain,
        so other task adapters are preserved.
        """
        if not hasattr(self._backbone, "peft_config"):
            return
        adapter_name = f"task_{task_id}"
        if adapter_name not in self._backbone.peft_config:
            return
        self._backbone.merge_adapter(adapter_names=[adapter_name])
        del self._backbone.peft_config[adapter_name]
        if not self._backbone.peft_config:
            # No adapters left — unwrap back to a plain nn.Module
            self._backbone = self._backbone.model
        if self._active_lora == task_id:
            self._active_lora = None

    def set_active_lora(self, task_id: int) -> None:
        """Activate adapter task_{task_id} for inference."""
        adapter_name = f"task_{task_id}"
        if (hasattr(self, "_backbone")
                and hasattr(self._backbone, "set_adapter")
                and hasattr(self._backbone, "peft_config")
                and adapter_name in self._backbone.peft_config):
            self._backbone.set_adapter(adapter_name)
        self._active_lora = task_id

    # ------------------------------------------------------------------
    # nn.Module overrides
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Keep _device in sync
        try:
            if args and isinstance(args[0], (torch.device, str)):
                self._device = torch.device(args[0])
            elif "device" in kwargs:
                self._device = torch.device(kwargs["device"])
        except Exception:
            pass
        return result

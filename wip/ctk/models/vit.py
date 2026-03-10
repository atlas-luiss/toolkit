"""ViTModel — timm-backed Vision Transformer for continual learning."""

import torch
import torch.nn as nn
from typing import Optional, List
import logging

from .base import CLModel

logger = logging.getLogger(__name__)


class ViTModel(CLModel):
    """timm-backed Vision Transformer; backbone loaded with num_classes=0 to return CLS embeddings."""

    def __init__(
        self,
        model_name:      str  = 'vit_base_patch16_224',
        pretrained:      bool = True,
        freeze_backbone: bool = False,
        device:          Optional[torch.device] = None,
        **timm_kwargs,
    ):
        super().__init__(device)

        try:
            import timm
        except ImportError:
            raise ImportError("timm required: pip install timm")

        self._backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **timm_kwargs,
        ).to(self._device)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self._device)
            self._embed_dim = self._backbone(dummy).shape[-1]

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(f"ViTModel: {model_name}  embed_dim={self._embed_dim}  "
                    f"pretrained={pretrained}  frozen={freeze_backbone}")

    # ------------------------------------------------------------------
    # CLModel contract
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _default_lora_targets(self) -> List[str]:
        return ["qkv"]

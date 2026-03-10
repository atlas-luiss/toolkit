"""DINOModel — DINOv2 backbone via torch.hub for continual learning."""

import torch
import torch.nn as nn
from typing import Optional, List
import logging

from .base import CLModel

logger = logging.getLogger(__name__)


class DINOModel(CLModel):
    """DINOv2 backbone via torch.hub; returns (B, embed_dim) CLS token embeddings."""

    def __init__(
        self,
        model_name:      str  = 'dinov2_vitb14',
        pretrained:      bool = True,
        freeze_backbone: bool = False,
        device:          Optional[torch.device] = None,
    ):
        super().__init__(device)

        # torch.hub always downloads pretrained DINOv2 weights; pretrained= is ignored
        self._backbone = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
        ).to(self._device)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self._device)
            out = self._backbone(dummy)
            if isinstance(out, dict):
                feat = out['x_norm_clstoken']
            else:
                feat = out
            self._embed_dim = feat.shape[-1]

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(f"DINOModel: {model_name}  embed_dim={self._embed_dim}  "
                    f"pretrained={pretrained}  frozen={freeze_backbone}")

    # ------------------------------------------------------------------
    # CLModel contract
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self._backbone(x)
        if isinstance(out, dict):
            return out['x_norm_clstoken']
        return out

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _default_lora_targets(self) -> List[str]:
        return ["qkv"]

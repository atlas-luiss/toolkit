"""CLIPModel — OpenCLIP-backed vision encoder for continual learning."""

import torch
import torch.nn as nn
from typing import Optional, List
import logging

from .base import CLModel

logger = logging.getLogger(__name__)


class CLIPModel(CLModel):
    """OpenCLIP vision-encoder backbone (image tower only); requires open_clip_torch."""

    def __init__(
        self,
        model_name:      str  = 'ViT-B-32',
        pretrained:      str  = 'openai',
        freeze_backbone: bool = True,    # CLIP is usually kept frozen
        device:          Optional[torch.device] = None,
        cache_dir:       Optional[str] = None,
    ):
        super().__init__(device)

        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip required: pip install open_clip_torch")

        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir

        full_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, **kwargs)
        full_model = full_model.to(self._device)

        self._backbone  = full_model.visual
        self._embed_dim = full_model.visual.output_dim

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(f"CLIPModel: {model_name}/{pretrained}  "
                    f"embed_dim={self._embed_dim}  frozen={freeze_backbone}")

    # ------------------------------------------------------------------
    # CLModel contract
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.float()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _default_lora_targets(self) -> List[str]:
        # open_clip VisionTransformer: attn has out_proj only (in_proj is fused);
        # MLP has c_fc and c_proj
        return ["out_proj", "c_fc", "c_proj"]

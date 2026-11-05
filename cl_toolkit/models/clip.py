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

        full_model, train_tf, val_tf = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, **kwargs)
        full_model = full_model.to(self._device)
        self._clip_transforms = (train_tf, val_tf)          # CLIP's own preprocessing
        # the text tower is kept outside the module tree (not trained, not saved) for zero-shot heads
        self._text = [full_model, open_clip.get_tokenizer(model_name)]

        self._backbone  = full_model.visual
        self._embed_dim = full_model.visual.output_dim
        self.model_name = f"open_clip:{model_name}/{pretrained}"
        pc = getattr(full_model.visual, "preprocess_cfg", None) or {}
        size = pc.get("size", getattr(full_model.visual, "image_size", 224))
        self._preprocess = {"size": size[0] if isinstance(size, (tuple, list)) else int(size),
                            "mean": tuple(pc.get("mean", open_clip.OPENAI_DATASET_MEAN)),
                            "std": tuple(pc.get("std", open_clip.OPENAI_DATASET_STD)),
                            "interpolation": pc.get("interpolation", "bicubic")}

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

    def preprocess_config(self) -> dict:
        """CLIP's own resolution and normalisation (not ImageNet's)."""
        return dict(self._preprocess)

    def transforms(self, preset: str = "model", low_res: bool = False):
        """``"model"``: open_clip's own train / val transforms; other presets as for any image model."""
        if preset == "model":
            return self._clip_transforms
        from cl_toolkit.benchmarks.transforms import image_transforms
        return image_transforms(preset, low_res, **self.preprocess_config())

    @torch.no_grad()
    def text_classifier(self, class_names: List[str], templates=None) -> torch.Tensor:
        """(C, D) zero-shot class embeddings: mean over templates of normalised text embeddings,
        re-normalised (as in CLIP / task_vectors / MagMax)."""
        full, tok = self._text
        templates = templates or CIFAR_TEMPLATES
        out = []
        for c in class_names:
            t = tok([tpl.format(c) for tpl in templates]).to(self._device)
            e = torch.nn.functional.normalize(full.encode_text(t).float(), dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=0))
        return torch.stack(out)


    def _default_lora_targets(self) -> List[str]:
        # open_clip VisionTransformer: attn has out_proj only (in_proj is fused);
        # MLP has c_fc and c_proj
        return ["out_proj", "c_fc", "c_proj"]


# CLIP's CIFAR templates (Radford et al. 2021; used by task_vectors and MagMax)
CIFAR_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.", "a photo of the {}.",
    "a blurry photo of the {}.", "a black and white photo of the {}.", "a low contrast photo of the {}.",
    "a high contrast photo of the {}.", "a bad photo of the {}.", "a good photo of the {}.",
    "a photo of the small {}.", "a photo of the big {}.",
]

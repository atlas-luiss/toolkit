"""ViTModel: any timm ViT (supervised, DINOv2, CLIP, EVA, ...) as a CL backbone.

Examples of model names::

    vit_base_patch16_224                      # timm default tag: augreg2_in21k_ft_in1k
    vit_base_patch16_224.augreg_in21k         # "ViT-B/16-IN21K" of L2P / DualPrompt / CODA / RanPAC
    vit_base_patch16_224.augreg_in21k_ft_in1k # "ViT-B/16-IN1K" of SimpleCIL / APER / RanPAC

The exact pretrained tag is recorded in ``model_name`` (and so in every run's config).
    vit_base_patch14_dinov2.lvd142m           # DINOv2
    vit_base_patch16_clip_224.openai          # CLIP image tower
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from .base import CLModel

logger = logging.getLogger(__name__)


class ViTModel(CLModel):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device=None,
        prompt=None,
        **timm_kwargs,
    ):
        super().__init__(device)
        import timm
        self._backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                           **timm_kwargs).to(self._device)
        cfg = getattr(self._backbone, "pretrained_cfg", {}) or {}
        arch, tag = cfg.get("architecture", model_name.split(".")[0]), cfg.get("tag")
        self.model_name = f"{arch}.{tag}" if (pretrained and tag) else model_name   # exact weights used
        self._embed_dim = self._backbone.num_features
        self._prompt = None
        if freeze_backbone:
            self.freeze_backbone()
        if prompt is not None:
            self.attach_prompts(prompt)
        logger.info("ViTModel %s embed_dim=%d frozen=%s", model_name, self._embed_dim, freeze_backbone)

    # -- CLModel contract -----------------------------------------------------
    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def vit(self):
        """The underlying timm VisionTransformer (unwrapping peft if present)."""
        return getattr(self._backbone, "model", self._backbone)

    def _encode(self, x):
        if self._prompt is None:
            return self._backbone(x)
        from cl_toolkit.prompts import set_prefixes
        q = None
        if getattr(self._prompt, "needs_query", True):
            with torch.no_grad():                   # query: frozen PTM [CLS], no prompts
                set_prefixes(self.vit, None)
                q = self._backbone(x)
            prefixes = self._prompt(q)
        else:
            prefixes = self._prompt(None, batch_size=x.size(0))
        set_prefixes(self.vit, prefixes)
        try:
            return self._backbone(x)
        finally:
            set_prefixes(self.vit, None)

    # -- prompts ----------------------------------------------------------------
    def attach_prompts(self, prompt) -> None:
        """Attach a prompt pool (see :mod:`cl_toolkit.prompts`). The backbone should be frozen."""
        from cl_toolkit.prompts import enable_prefix
        prompt.bind(self._embed_dim)
        enable_prefix(self.vit, prompt.layers)
        self._prompt = prompt.to(self._device)

    def begin_task(self, task_id, num_tasks):
        if self._prompt is not None:
            self._prompt.begin_task(task_id, num_tasks)
            self._prompt.to(self._device)

    @property
    def aux_loss(self):
        if self._prompt is not None and self._prompt._aux is not None:
            return self._prompt._aux
        return torch.zeros((), device=self._device)

    # -- preprocessing ------------------------------------------------------------
    def preprocess_config(self) -> dict:
        """Resolution, normalisation and interpolation this backbone was pre-trained with."""
        import timm.data
        cfg = timm.data.resolve_data_config({}, model=self.vit)
        return {"size": cfg["input_size"][-1], "mean": cfg["mean"], "std": cfg["std"],
                "interpolation": cfg.get("interpolation", "bicubic")}

    def transforms(self, preset: str = "model", low_res: bool = False) -> Tuple:
        """(train_tf, test_tf); see :mod:`cl_toolkit.benchmarks.transforms` for the presets."""
        from cl_toolkit.benchmarks.transforms import image_transforms
        return image_transforms(preset, low_res, **self.preprocess_config())


def DINOModel(model_name: str = "vit_base_patch14_dinov2.lvd142m", pretrained=True,
              freeze_backbone=False, device=None, **kw) -> ViTModel:
    """DINOv2 via timm. Accepts old torch.hub names like ``dinov2_vitb14``."""
    hub = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m", "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
           "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m", "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m"}
    return ViTModel(hub.get(model_name, model_name), pretrained, freeze_backbone, device,
                    img_size=kw.pop("img_size", 224), **kw)

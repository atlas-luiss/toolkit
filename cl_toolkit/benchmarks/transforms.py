"""Image preprocessing presets for the named image benchmarks.

``preprocess=`` (``clt.run(..., data_kwargs={"preprocess": ...})`` / ``--data-set preprocess=...``):

* ``"model"`` (default): the backbone's own resolution and normalisation (timm / CLIP
  config). Low-resolution datasets (CIFAR) are resized, never center-cropped: cropping a
  32x32 image upsampled to 224 throws away its border (-2 points for NCM on CIFAR-100).
* ``"pilot"``: the transforms of the PILOT / APER / SimpleCIL / RanPAC code bases.
  No normalisation (inputs in [0, 1]); train RandomResizedCrop(scale 0.05-1) + flip;
  test bicubic Resize (CIFAR) or Resize(256) + CenterCrop(224).
* ``"coda"``: the transforms of the CODA-Prompt code base (also used for its L2P++ and
  DualPrompt numbers). No normalisation; train RandomResizedCrop(224) + flip; test
  Resize(224) (CIFAR) or Resize(256) + CenterCrop(224).

Published PTM-CL numbers use the last two. With a frozen IN21K ViT-B/16 on Split
CIFAR-100, NCM gets 80.1 with "model" and 83.4 with "pilot" (paper: 83.4).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

PRESETS = ("model", "pilot", "coda")
LOW_RES = ("cifar10", "cifar100")


def image_transforms(preset: str = "model", low_res: bool = False, size: int = 224,
                     mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
                     interpolation: str = "bicubic") -> Tuple:
    from torchvision import transforms as T
    if preset not in PRESETS:
        raise ValueError(f"preprocess must be one of {PRESETS}, got {preset!r}")
    bicubic, bilinear = T.InterpolationMode.BICUBIC, T.InterpolationMode.BILINEAR
    if preset == "pilot":
        train = [T.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3 / 4, 4 / 3)), T.RandomHorizontalFlip(0.5),
                 T.ToTensor()]
        test = ([T.Resize(224, interpolation=bicubic), T.CenterCrop(224)] if low_res else
                [T.Resize(256, interpolation=bicubic), T.CenterCrop(224)]) + [T.ToTensor()]
        return T.Compose(train), T.Compose(test)
    if preset == "coda":
        train = [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]
        test = ([T.Resize(224)] if low_res else [T.Resize(256), T.CenterCrop(224)]) + [T.ToTensor()]
        return T.Compose(train), T.Compose(test)
    interp = bicubic if interpolation == "bicubic" else bilinear
    norm = [T.Normalize(mean, std)] if mean is not None else []
    train = [T.RandomResizedCrop(size, interpolation=interp), T.RandomHorizontalFlip(), T.ToTensor()] + norm
    test = ([T.Resize((size, size), interpolation=interp)] if low_res else
            [T.Resize(int(round(size / 0.875)), interpolation=interp), T.CenterCrop(size)]) + [T.ToTensor()] + norm
    return T.Compose(train), T.Compose(test)

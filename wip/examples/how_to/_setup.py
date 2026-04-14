"""
Shared setup for ImageNet-R how-to examples.

Paths are read from environment variables so the examples work outside
the default HPC environment.  Set them before running:

    export CTK_IMAGENET_R=/path/to/imagenet-r
    export CTK_MODEL_CACHE=/path/to/model_cache   # optional
    export CTK_HF_CACHE=/path/to/hf_cache         # optional
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGENET_R_ROOT = os.environ.get(
    "CTK_IMAGENET_R",
    "/leonardo_work/IscrC_ConFT/eric/data/imagenet-r",
)
MODEL_CACHE = os.environ.get(
    "CTK_MODEL_CACHE",
    "/leonardo_work/IscrC_ConFT/eric/model_cache",
)
HF_CACHE = os.environ.get(
    "CTK_HF_CACHE",
    "/leonardo_work/IscrC_ConFT/eric/hf_cache",
)

if not os.path.isdir(IMAGENET_R_ROOT):
    raise RuntimeError(
        f"ImageNet-R dataset not found at: {IMAGENET_R_ROOT!r}\n"
        "Point CTK_IMAGENET_R to your copy of the dataset:\n"
        "  export CTK_IMAGENET_R=/path/to/imagenet-r"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ.setdefault("HF_HOME", HF_CACHE)
torch.hub.set_dir(MODEL_CACHE)


# ── Factories ─────────────────────────────────────────────────────────────────

def make_benchmark(num_tasks: int = 10):
    """Return an ImageNetR benchmark split into num_tasks sequential tasks."""
    from ctk.benchmarks import ImageNetR
    return ImageNetR(root=IMAGENET_R_ROOT, num_tasks=num_tasks, seed=42)


def make_model():
    """Return a DINOv2-B/14 model with frozen backbone (only head is trained)."""
    from ctk.models import DINOModel
    return DINOModel("dinov2_vitb14", freeze_backbone=True,
                     device=torch.device(DEVICE))

"""
Shared paths and utilities for ImageNet-R continual learning experiments.

Paths are read from environment variables so experiments work outside
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


# ── Result printer ────────────────────────────────────────────────────────────

def print_results(results: dict, elapsed: float) -> None:
    """Pretty-print the accuracy matrix and summary metrics."""
    mat = results["accuracy_matrix"]
    n   = mat.shape[0]

    print("\n" + "─" * 60)
    print("  Accuracy matrix  (row = after task t, col = eval on task j)")
    print("  " + "  ".join(f"  T{j}" for j in range(n)))
    for i in range(n):
        row = f"  T{i} →"
        for j in range(n):
            v = mat[i, j]
            row += f"  {'---' if (v == 0 and j > i) else f'{v:4.1f}'}"
        print(row)
    print()
    print(f"  avg accuracy   : {results['avg_accuracy']:.1f}%")
    print(f"  avg forgetting : {results['avg_forgetting']:.1f}%")
    print(f"  wall time      : {elapsed / 60:.1f} min")
    print("─" * 60 + "\n")

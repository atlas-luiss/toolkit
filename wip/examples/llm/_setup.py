"""Shared setup for LLM fine-tuning examples (model path, cache dir, HPC env)."""

import os

# ------------------------------------------------------------------
# Paths — adjust to your cluster layout
# ------------------------------------------------------------------

HF_CACHE = "/leonardo_work/IscrC_ConFT/eric/hf_cache"
DATA_CACHE = "/leonardo_work/IscrC_ConFT/eric/hf_cache/datasets"
OUTPUT_DIR = "/leonardo_work/IscrC_ConFT/eric/llm_outputs"

# Default small model for quick tests; swap for larger models
DEFAULT_MODEL = "facebook/opt-125m"   # ~240MB, good for smoke tests

# ------------------------------------------------------------------
# Offline mode (required on compute nodes with no internet)
# ------------------------------------------------------------------

os.environ.setdefault("HF_HOME", HF_CACHE)
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def print_results(results: dict) -> None:
    """Pretty-print training results."""
    print(f"\n{'='*60}")
    print("Training complete")
    print(f"  Total steps: {results.get('total_steps', '?')}")
    print(f"  Elapsed:     {results.get('elapsed', 0):.0f}s")
    fm = results.get("final_metrics", {})
    for k, v in fm.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

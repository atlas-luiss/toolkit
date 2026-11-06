"""A representative slice of experiments/interop_grid.py (run that script for the full grid).

One method per family x every scenario on a CLModel, a timm ViT and a Hugging Face decoder LLM:
each cell must run (finite accuracies, no task-identity leak) or be refused with a clear error.
"""
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest

pytest.importorskip("timm")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import interop_grid as G  # noqa: E402

METHODS = ["naive", "derpp", "agem", "icarl", "ewc", "lwf", "lora-er", "head-fecam", "head-slca", "l2p",
           "routed-lora-hide", "routed-prompt-sprompts", "tsv", "cofima"]
MODELS = ["clmodel", "vit", "decoder"]


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    try:
        os.environ["CLT_INTEROP_HF"] = G.HF_DIR = str(tmp_path_factory.mktemp("hf"))   # also for workers
        G._hf_dirs()                                # needs the Qwen2.5 tokenizer (HF cache or network)
    except Exception as e:                          # noqa: BLE001
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    cells = [(m, k, s) for m in METHODS for k in MODELS for s in G.SCENARIOS]
    # spawn: earlier tests may have initialised CUDA, which forked workers cannot use
    with ProcessPoolExecutor(min(16, os.cpu_count() or 1), mp_context=multiprocessing.get_context("spawn"),
                             initializer=G.cpu_only_worker) as ex:
        return {c: r for c, r in zip(cells, ex.map(G.run_cell, *zip(*cells)))}


def test_every_cell_runs_or_is_refused_clearly(grid):
    bad = {c: r["note"] for c, r in grid.items() if r["status"] == "FAIL"}
    assert not bad, bad


def test_refusals_are_exactly_the_documented_ones(grid):
    for (m, k, s), r in grid.items():
        assert (r["status"] == "refused") == (G.expected_refusal(m, k, s) is not None), (m, k, s, r["note"])

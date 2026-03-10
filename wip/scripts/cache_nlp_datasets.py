"""
Pre-download NLP datasets to HF cache for offline use on compute nodes.

"""

import os
import sys

CACHE_DIR = os.environ.get("HF_HOME", "/leonardo_work/IscrC_ConFT/eric/hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Downloading to: {CACHE_DIR}\n")


from datasets import load_dataset



def download(name, config=None, split=None, **kwargs):
    label = f"{name}" + (f"/{config}" if config else "")
    print(f"  Downloading {label} ...", end=" ", flush=True)
    try:
        if config:
            ds = load_dataset(name, config, cache_dir=CACHE_DIR, **kwargs)
        else:
            ds = load_dataset(name, cache_dir=CACHE_DIR, **kwargs)
        splits = list(ds.keys()) if hasattr(ds, "keys") else ["?"]
        print(f"OK  ({', '.join(splits)})")
    except Exception as e:
        print(f"FAILED: {e}")


print("Classification:")
download("glue", "sst2")
download("glue", "mnli")
download("ag_news")

print("\nQuestion Answering:")
download("rajpurkar/squad_v2")

print("\nSummarization:")
download("knkarthick/samsum")
download("cnn_dailymail", "3.0.0")

print("\nInstruction / Reasoning:")
download("tatsu-lab/alpaca")
download("openai/gsm8k", "main")

print("\nAll done. Set TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1 on compute nodes.")

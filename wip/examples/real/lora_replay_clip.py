"""
LoRA + Replay on CLIP  ·  ImageNet-R  ·  backbone comparison
=============================================================
Mirrors lora_replay_dino.py exactly but with the CLIP ViT-B/32 backbone.
Holding the strategy constant and swapping only the backbone isolates the
contribution of pretraining — language supervision (CLIP) vs self-supervised
DINO — on continual learning performance.

See lora_clip.py for why CLIP requires explicit target_modules.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, MODEL_CACHE, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import CLIPModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAReplay
from ctk.trainers   import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
model = CLIPModel(
    model_name      = "ViT-B-32",
    pretrained      = "openai",
    freeze_backbone = True,
    device          = torch.device(DEVICE),
    cache_dir       = MODEL_CACHE,
)
print(f"Model      →  CLIP ViT-B/32  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental")

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy = LoRAReplay(
    rank           = 8,
    lora_alpha     = 16,
    target_modules = ["out_proj", "c_fc", "c_proj"],
    memory_size    = 1000,
    replay_ratio   = 0.5,
)
print(f"Strategy   →  LoRA + Replay  |  rank=8  |  memory=1000  |  replay_ratio=0.5\n")

# ── Train ─────────────────────────────────────────────────────────────────────
t0      = time.time()
results = ContinualTrainer(
    model, scenario, strategy,
    optimizer_cls   = torch.optim.AdamW,
    lr              = 1e-4,
    weight_decay    = 1e-4,
    epochs_per_task = 2,
    device          = DEVICE,
    verbose         = True,
).fit()

# ── Results ───────────────────────────────────────────────────────────────────
# Compare with lora_replay_dino.py — only the backbone differs.
print_results(results, time.time() - t0)

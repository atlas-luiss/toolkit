"""
LoRA on CLIP  ·  ImageNet-R  ·  backbone comparison
=====================================================
Identical experiment to lora_dino.py but with OpenAI CLIP (ViT-B/32) as the
backbone. Running both files shows how the pretraining objective — language
supervision (CLIP) vs self-supervised DINO — affects continual learning.

CLIP's attention projections use different layer names than DINOv2, so we
specify target_modules explicitly:
  "out_proj" → attention output projection
  "c_fc"     → first linear in the MLP block
  "c_proj"   → second linear in the MLP block
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, MODEL_CACHE, DEVICE, print_results

from ctk.benchmarks import ImageNetR
from ctk.models     import CLIPModel
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LoRAStrategy
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
)
print(f"Model      →  CLIP ViT-B/32  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental")

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy = LoRAStrategy(
    rank             = 8,
    lora_alpha       = 16,
    target_modules   = ["out_proj", "c_fc", "c_proj"],  # CLIP-specific layer names
    merge_after_task = False,
)
print(f"Strategy   →  LoRA  |  rank=8  |  α=16  |  CLIP target modules\n")

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
# Compare directly with lora_dino.py — all other settings are identical.
# Differences in avg_accuracy reflect the backbone pretraining quality.
print_results(results, time.time() - t0)

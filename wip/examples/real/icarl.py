"""
iCaRL (Class-Incremental Learning with Recall)  ·  ImageNet-R
==============================================================
iCaRL combines three ideas into one coherent algorithm:

  1. Herding exemplars — selects the K examples per class whose feature mean
     best approximates the full class mean in the embedding space.
  2. BCE + knowledge distillation — binary cross-entropy across all seen
     classes, with soft targets from a frozen snapshot of the previous model.
  3. Nearest-mean-of-exemplars (NME) classifier — at test time, assigns the
     class whose exemplar mean is closest in feature space (not the softmax head).

The NME classifier naturally handles the growing number of classes without
needing to resize the output head, making iCaRL well-suited to Class-IL.
"""

import time
import torch
from _setup import IMAGENET_R_ROOT, DEVICE, print_results

from ctk.benchmarks        import ImageNetR
from ctk.models            import DINOModel
from ctk.scenarios         import ClassIncremental
from ctk.strategies.replay import iCaRL
from ctk.trainers          import ContinualTrainer

# ── Benchmark ─────────────────────────────────────────────────────────────────
bench = ImageNetR(root=IMAGENET_R_ROOT, num_tasks=10, seed=42)
print(f"Benchmark  →  ImageNet-R  |  {bench.num_tasks} tasks  |  {bench.total_classes} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
model = DINOModel("dinov2_vitb14", freeze_backbone=True, device=torch.device(DEVICE))
print(f"Model      →  DINOv2-B/14  |  embed_dim={model.embed_dim}  |  backbone frozen")

# ── Scenario ──────────────────────────────────────────────────────────────────
scenario = ClassIncremental(bench, batch_size=64, num_workers=4,
                            pin_memory=(DEVICE == "cuda"))
print(f"Scenario   →  Class-Incremental")

# ── Strategy ──────────────────────────────────────────────────────────────────
# exemplars_per_class : stored examples per class; total = 200 classes × 20 = 4000
# temperature         : softens teacher logits for the distillation term
# kd_weight           : balance between the task loss and distillation
strategy = iCaRL(exemplars_per_class=20, temperature=2.0, kd_weight=1.0)
print(f"Strategy   →  iCaRL  |  exemplars/class=20  |  kd_weight=1.0\n")

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
# iCaRL is strong when the frozen backbone produces stable, well-separated features.
# DINOv2 features are excellent for NME, so expect competitive accuracy here.
print_results(results, time.time() - t0)

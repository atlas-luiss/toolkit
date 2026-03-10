# CTK — Collage Toolkit

A [Collage Lab](https://www.vincenzolomonaco.com/collage-lab) initiative.


## Structure

```
ctk/
  models/           model wrappers (ViT, CLIP, etc)
  strategies/       learning strategies 
  scenarios/        task splitting 
  benchmarks/       dataset loaders 
  trainers/         training loops 
  callbacks/        hooks: logging, checkpointing, metrics
  utils/            PEFT utilities

experiments/        vision experiment configs + SLURM scripts
examples/llm/       LLM fine-tuning examples

```



## Usage Example 

```python
import time
from _setup import *
from ctk.scenarios  import TaskIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainer    import ContinualTrainer
from ctk.benchmarks import SplitCIFAR100
from ctk.models     import ViTModel




DATA_ROOT   = "/leonardo_work/IscrC_ConFT/eric/data"
MODEL_CACHE = "/leonardo_work/IscrC_ConFT/eric/hf_cache"
NUM_TASKS   = 5
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.environ.setdefault("HF_HOME", MODEL_CACHE)

SCENARIO_KW = dict(batch_size=BATCH_SIZE, num_workers=4,
                   pin_memory=(DEVICE == "cuda"))

TRAINER_KW  = dict(
    optimizer_cls=torch.optim.AdamW,
    lr=LR,
    weight_decay=1e-4,
    epochs_per_task=EPOCHS,
    device=DEVICE,
    verbose=True,
)


benchmark = SplitCIFAR100(root=DATA_ROOT, num_tasks=NUM_TASKS,
                         image_size=224, seed=42, download=False)

model = ViTModel(
        model_name="vit_base_patch16_224",
        pretrained=True,
        freeze_backbone=True,
        device=torch.device(DEVICE),
    )
bench    = benchamrk
model    = model 
scenario = TaskIncremental(bench, **SCENARIO_KW)
strategy = LoRAStrategy(rank=8, lora_alpha=16, merge_after_task=False)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()



```


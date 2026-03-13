"""
LoRA Strategy — one adapter per task, Task-IL.

Backbone is trainable via LoRA (not frozen). Each task gets its own adapter.

"""
import time
from _setup import *
from ctk.scenarios  import TaskIncremental
from ctk.strategies import LoRAStrategy
from ctk.trainer    import ContinualTrainer
from ctk.benchmarks import SplitCIFAR100
from ctk.models     import ViTModel




t0 = time.time()
bench    = make_bench()
model    = make_model(freeze_backbone=True)  
scenario = TaskIncremental(bench, **SCENARIO_KW)
strategy = LoRAStrategy(rank=8, lora_alpha=16, merge_after_task=False)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()


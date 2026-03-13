"""Experience Replay — reservoir buffer, replay alongside current task."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import ExperienceReplay
from ctk.trainer    import ContinualTrainer

print_header("Experience Replay", "memory_size=2000, replay_ratio=0.5")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = ExperienceReplay(memory_size=2000, replay_batch_ratio=0.5)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

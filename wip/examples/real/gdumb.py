"""GDumb — ignore stream, retrain from scratch on a balanced buffer."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import GDumb
from ctk.trainer    import ContinualTrainer

print_header("GDumb", "memory_size=2000, replay_epochs=10")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = GDumb(memory_size=2000, replay_epochs=10, lr=1e-3)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

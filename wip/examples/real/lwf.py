"""LwF — distillation from frozen past-task model snapshot."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import LwF
from ctk.trainer    import ContinualTrainer

print_header("LwF", "lam=1.0, temperature=2.0")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = LwF(lam=1.0, temperature=2.0)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

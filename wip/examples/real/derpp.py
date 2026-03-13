"""DER++ — replay inputs + stored logits (dark knowledge)."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import DERPP
from ctk.trainer    import ContinualTrainer

print_header("DER++", "memory_size=2000, alpha=0.2, beta=0.5")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = DERPP(memory_size=2000, alpha=0.2, beta=0.5)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

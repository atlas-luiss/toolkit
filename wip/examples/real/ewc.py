"""EWC — Fisher-based penalty on important parameters."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import EWC
from ctk.trainer    import ContinualTrainer

print_header("EWC", "lam=1000, fisher_samples=500")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = EWC(lam=1000.0, fisher_samples=500)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

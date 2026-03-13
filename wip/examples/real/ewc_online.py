"""EWC Online — single decaying running Fisher estimate."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import EWCOnline
from ctk.trainer    import ContinualTrainer

print_header("EWC Online", "lam=1000, gamma=0.9, fisher_samples=500")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = EWCOnline(lam=1000.0, gamma=0.9, fisher_samples=500)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

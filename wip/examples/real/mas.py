"""MAS — unsupervised importance from gradient of output L2 norm."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import MAS
from ctk.trainer    import ContinualTrainer

print_header("MAS", "lam=1.0, importance_samples=500")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = MAS(lam=1.0, importance_samples=500)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

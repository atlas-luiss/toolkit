"""iCaRL — herding exemplars, BCE+KD loss, NMC inference."""
import time
from _setup import *
from ctk.scenarios       import ClassIncremental
from ctk.strategies.replay import iCaRL
from ctk.trainer         import ContinualTrainer

print_header("iCaRL", "exemplars_per_class=20, kd_weight=1.0")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = iCaRL(exemplars_per_class=20, temperature=2.0, kd_weight=1.0)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

"""Naive fine-tuning — lower bound. Shows full catastrophic forgetting."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import NaiveStrategy
from ctk.trainer    import ContinualTrainer

print_header("Naive")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = NaiveStrategy()
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

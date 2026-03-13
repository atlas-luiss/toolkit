"""Joint training — upper bound. All data at once, no forgetting possible."""
import time
from _setup import *
from ctk.scenarios.finetuning import JointTraining
from ctk.strategies import NaiveStrategy
from ctk.trainer    import ContinualTrainer

print_header("Joint Training (upper bound)")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = JointTraining(bench, **SCENARIO_KW)
strategy = NaiveStrategy()
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

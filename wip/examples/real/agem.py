"""A-GEM — project gradient to not increase past-task loss."""
import time
from _setup import *
from ctk.scenarios  import ClassIncremental
from ctk.strategies import AGEM
from ctk.trainer    import ContinualTrainer

print_header("A-GEM", "memory_size=2000, ref_sample_size=256")
t0 = time.time()
bench    = make_bench()
model    = make_model()
scenario = ClassIncremental(bench, **SCENARIO_KW)
strategy = AGEM(memory_size=2000, ref_sample_size=256)
results  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW).fit()
print_results(results, time.time() - t0)

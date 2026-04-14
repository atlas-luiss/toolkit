"""
HOW-TO: Scenarios
==================
The scenario controls how the benchmark data is presented to the model
and what kind of output head is attached.

JointTraining (non-CL baseline / upper bound)
  - ALL task data concatenated into a single training pass
  - No sequential task structure — standard supervised learning
  - Best possible accuracy; use this as the ceiling to compare CL against
  - If your CL method approaches this number, it is working well

ClassIncremental (Class-IL)
  - Single head that grows as new classes arrive
  - Task identity is NOT available at test time
  - Hardest CL setting — the model must separate all classes seen so far

TaskIncremental (Task-IL)
  - One separate head per task
  - Task identity IS available at test time (e.g. from a task oracle)
  - Easier; typically achieves higher accuracy than Class-IL

The same benchmark and strategy work with all three — only the scenario changes.

Dataset : ImageNet-R  (5 tasks of 40 classes each)
Model   : DINOv2-B/14 with frozen backbone
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "../.."))

import torch

from ctk import Trainer
from ctk.scenarios import ClassIncremental, TaskIncremental, JointTraining
from ctk.strategies import NaiveStrategy

from _setup import make_benchmark, make_model, DEVICE

NUM_TASKS = 5


def run(ScenarioClass, label, is_joint=False):
    benchmark = make_benchmark(num_tasks=NUM_TASKS)
    scenario  = ScenarioClass(benchmark, batch_size=128, num_workers=4,
                              pin_memory=(DEVICE == "cuda"))
    model     = make_model()

    results = Trainer(
        model, scenario, NaiveStrategy(),
        optimizer_cls=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4,
        epochs_per_task=1, device=DEVICE, verbose=False,
    ).fit()

    if is_joint:
        # JointTraining has one "task" — forgetting is not applicable
        print(f"  {label:<22}  acc={results['avg_accuracy']:5.1f}%  "
              f"forgetting=  n/a  (upper bound)")
    else:
        print(f"  {label:<22}  "
              f"acc={results['avg_accuracy']:5.1f}%  "
              f"forgetting={results['avg_forgetting']:5.1f}%")


print(f"\nScenario comparison (NaiveStrategy, {NUM_TASKS} tasks, 1 epoch each):\n")

# JointTraining — not CL; trains on everything at once.
# This is standard supervised learning and sets the accuracy ceiling.
run(JointTraining,    "JointTraining",    is_joint=True)

# ClassIncremental — hardest CL setting; task ID hidden at test time
run(ClassIncremental, "ClassIncremental")

# TaskIncremental — easier CL; task ID available at test time
run(TaskIncremental,  "TaskIncremental")


# ---------------------------------------------------------------------------

print("""
How to read these numbers:
  JointTraining    → ceiling (all data seen at once, no forgetting possible)
  ClassIncremental → hardest: single growing head, no task oracle
  TaskIncremental  → easier:  per-task head, task ID given at test time

A CL strategy is effective when ClassIncremental accuracy approaches
the JointTraining ceiling.
""")

# ---------------------------------------------------------------------------
# What changes between the scenarios?
#
# JointTraining._build_loaders() returns ONE TaskInfo with ConcatDataset
#   over all splits.  Standard supervised training.
#
# ClassIncremental.setup() attaches an IncrementalClassifier:
#   head grows by classes_per_task neurons each task
#   forward(x, task_id) → same head regardless of task_id
#
# TaskIncremental.setup() attaches a MultiHeadClassifier:
#   one nn.Linear per task, selected by task_id at inference
#   labels are remapped to 0-based per task (via _LabelRemap)
# ---------------------------------------------------------------------------

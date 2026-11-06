"""Customising the loop: callbacks, trainer subclassing, build-without-run."""
import os

import torch
import cl_toolkit as clt
from cl_toolkit.callbacks import Callback, EarlyStopping, GradientClipping, MetricLogger

ROOT = os.environ.get("CL_DATA", "./data")


class PrintSeenClasses(Callback):
    def on_task_end(self, trainer, task_id, task_metrics, **kw):
        print(f"  [cb] after task {task_id}: {len(trainer.scenario.seen_classes(task_id))} classes, "
              f"acc={task_metrics['accuracy']:.1f}")


# 1) subclass the trainer for anything the hooks don't cover
class SGDTrainer(clt.ContinualTrainer):
    def make_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.lr, momentum=0.9)


# 2) build() gives you the trainer without running it
trainer = clt.build("cifar100", root=f"{ROOT}/cifar100", strategy="er", num_tasks=5, epochs=3, lr=0.01,
                    scheduler="cosine", precision="bf16", trainer_cls=SGDTrainer,
                    callbacks=[PrintSeenClasses(), GradientClipping(1.0), EarlyStopping("loss", patience=1),
                               MetricLogger(["console", "csv"], csv_path="logs/er.csv")])
print(trainer.scenario, trainer.strategy, sep="\n")
results = trainer.fit()
trainer.save_results("results_er.json")

# 7. Your own CL method

A CL method in the toolkit is a **strategy**: a class whose methods are called at fixed moments of
training. You only write the moments you care about; everything else (data loading, the
optimiser, evaluation, logging) is done for you.

```
for each task:
    on_task_start(task_id, train_loader, model)        # before the task
    for each batch:
        compute_loss(batch, model, task_id)            # -> the loss to minimise
        before_update(model, optimizer)                # after backward, before the weight update
    on_task_end(task_id, train_loader, model)          # after the task
    evaluate on every task seen so far
```

```python
import torch
import torch.nn.functional as F
import cl_toolkit as clt
from cl_toolkit.strategies import Strategy

quick = dict(data="toy", model="mlp", num_tasks=5)
```

## Step 1: the simplest method, plain fine-tuning

`compute_loss` gets a batch and returns the loss (and the scores, used for the training accuracy
in the log):

```python
class FineTune(Strategy):
    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)        # move the batch to the model's device
        scores = model(x, task_id)                 # one score per class
        loss = F.cross_entropy(scores, y)
        return loss, scores

r = clt.run(strategy=FineTune(), **quick)
print("fine-tune", round(r["avg_accuracy"], 1))      # forgets: only the last task is remembered
```

## Step 2: remember a few old samples (replay)

Keep a small memory of past samples and train on them together with the new ones. The toolkit
has a ready-made memory, `ReservoirBuffer`:

```python
from cl_toolkit.strategies.replay import ReservoirBuffer

class MyReplay(Strategy):
    def __init__(self, memory_size=200):
        self.memory = ReservoirBuffer(memory_size)

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        scores = model(x, task_id)
        loss = F.cross_entropy(scores, y)

        old_x, old_y, old_t, _ = self.memory.sample(len(x), x.device)     # None while empty
        if old_x is not None:
            with self.memory_mask(model, task_id):                          # see the note below
                loss = loss + F.cross_entropy(model(old_x, old_t), old_y)

        self.memory.add(x, y, task_id)                                      # store some new samples
        return loss, scores

r = clt.run(strategy=MyReplay(), **quick)
print("my replay", round(r["avg_accuracy"], 1))
```

Note: while learning a task, the toolkit only lets the classes seen so far be predicted (the
others get a score of `-inf`). `memory_mask` makes sure old samples can use all of them.

## Step 3: do something between tasks

`on_task_start` and `on_task_end` run once per task. Here: keep a copy of the model from the
previous task and ask the new model to give similar scores on old classes (distillation):

```python
from copy import deepcopy

class Distill(Strategy):
    def __init__(self, weight=1.0):
        self.weight, self.old_model = weight, None

    def on_task_start(self, task_id, train_loader, model):
        if task_id > 0:
            self.old_model = deepcopy(model).eval()           # frozen snapshot of the past

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        scores = model(x, task_id)
        loss = F.cross_entropy(scores, y)
        if self.old_model is not None:
            old = self.scenario.seen_classes(task_id - 1)     # classes of the earlier tasks
            with torch.no_grad():
                target = self.old_model(x, task_id)[:, old]
            loss = loss + self.weight * F.mse_loss(scores[:, old], target)
        return loss, scores

r = clt.run(strategy=Distill(), **quick)
print("distill", round(r["avg_accuracy"], 1))
```

Inside a strategy, `self.scenario` gives the classes of each task (`seen_classes(t)`,
`get_task(t).classes`) and `self.task` is the current task.

## Step 4: watch training with callbacks

A **callback** watches the training loop without being part of the method: print, save, stop
early. Hooks: `on_task_end`, `on_epoch_end`, `on_batch_end`, `on_train_end`, and more; each gets
the trainer.

```python
class PrintProgress(clt.callbacks.Callback):
    def on_task_end(self, trainer, task_id, task_metrics, **kw):
        print(f"after task {task_id}: {task_metrics['accuracy']:.1f}% on the tasks seen so far")

r = clt.run(strategy=MyReplay(), callbacks=[PrintProgress()], **quick)
```

Built-in callbacks: `GradientClipping(1.0)`, `EarlyStopping`, `ModelCheckpointing`,
`MetricLogger` (CSV, W&B), `ProgressBar`.

So: the **strategy** decides *how the model learns*; a **callback** *observes* the loop.

## Step 5: compare and combine

Your method plugs into everything built in:

```python
for method in [FineTune(), MyReplay(), Distill(), "er", "lwf"]:
    r = clt.run(strategy=method, **quick)
    print(getattr(method, "name", method), round(r["avg_accuracy"], 1))

r = clt.run(strategy=MyReplay(), classifier="ncm", **quick)                   # + a head (step 5)
r = clt.run(strategy=clt.strategies.LoRAAdapter(MyReplay()), **quick)         # + LoRA per task (step 4)
```

To check that your method is fair (never looks at test data or the task id at test time), add it
to `tests/test_integrity.py` (one line in `CASES`) and run `pytest tests/test_integrity.py`.

**Next:** [8. Write the CL loop yourself](8_your_loop.md)

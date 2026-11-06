# Customize: more extension points

Your own classifier head, router, merging rule and training loop. Each example runs as-is in seconds.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cl_toolkit as clt
```

Your data, your model and your own CL method are covered step by step in the
[tutorial](tutorial/README.md) (steps 1, 7 and 8). This page adds the other extension points.

## Your classifier head

Collect statistics after each task (`update`), score test samples (`logits`):

```python
class NearestMean(clt.heads.Head):
    def update(self, feats, labels, task_id):
        if not hasattr(self, "means"):
            self.means = {}
        for c in labels.unique().tolist():
            self.means[c] = feats[labels == c].mean(0)

    def logits(self, feats):
        scores = torch.full((len(feats), self.num_classes), float("-inf"), device=feats.device)
        for c, m in self.means.items():
            scores[:, c] = -(feats - m).norm(dim=1)
        return scores

r = clt.run(data="toy", model="mlp", strategy="frozen", classifier=NearestMean(), epochs=0)
```

## Your router

With one expert per task, a router picks the expert for each test sample:

```python
class NearestTask(clt.routing.Router):
    def __init__(self):
        self.centers = []

    def fit(self, task_id, feats, labels):          # called after each task
        self.centers.append(feats.mean(0))

    def route(self, feats):                         # -> one task id per sample
        return torch.cdist(feats, torch.stack(self.centers)).argmin(1)

r = clt.run(data="toy", model="mlp", experts="lora", router=NearestTask())
print(r["final_routing_accuracy"])
```

## Your merging rule

Turn a list of task vectors (fine-tuned minus initial weights) into one:

```python
def median(task_vectors):
    return {k: torch.stack([tv[k] for tv in task_vectors]).median(0).values for k in task_vectors[0]}

clt.merging.METHODS["median"] = median
merge_replay = clt.strategies.MergeAfterTask(clt.strategies.ExperienceReplay(), "median")   # merge after every task
r = clt.run(data="toy", model="mlp", strategy=merge_replay)
```

## The training loop

Most things are arguments:

```python
r = clt.run(data="toy", model="mlp", strategy="er", epochs=3, lr=1e-2, batch_size=32,
            optimizer_cls=torch.optim.SGD, scheduler="cosine", seed=1)
```

To react to events, add a callback:

```python
class PrintAccuracy(clt.callbacks.Callback):
    def on_task_end(self, trainer, task_id, task_metrics, **kw):
        print(f"after task {task_id}: {task_metrics['accuracy']:.1f}%")

r = clt.run(data="toy", model="mlp", strategy="er", callbacks=[PrintAccuracy()])
```

For anything else, subclass the trainer and override one step (`make_optimizer`, `train_step`, ...):

```python
class SGDTrainer(clt.ContinualTrainer):
    def make_optimizer(self, params):
        return torch.optim.SGD(params, lr=self.lr, momentum=0.9)

r = clt.run(data="toy", model="mlp", strategy="er", trainer_cls=SGDTrainer, lr=0.05)
```

To run your own code *between* training and evaluation (merge after each task, evaluate only
some tasks, ...), write the loop yourself with `trainer.start()`, `trainer.train_task(task)`,
`trainer.evaluate(task_id)` and `trainer.finish()`: see
[tutorial step 8](tutorial/8_your_loop.md).

## Check your method

Add one line to `CASES` in `tests/test_integrity.py`, e.g.
`"my-method": (mlp, vec_bench, dict(strategy=MyMethod()))`, and run
`pytest tests/test_integrity.py -k my-method`. It checks that your method never touches test data,
never uses the task id at test time, does not peek at labels, and does not learn during evaluation.

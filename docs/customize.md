# Customize

Swap any piece for your own. Each example runs as-is in seconds.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cl_toolkit as clt
```

## Your data

Any datasets that return `(x, y)`, with `y` an integer class:

```python
from torch.utils.data import TensorDataset

x = torch.randn(6000, 32)
y = x[:, :10].argmax(1)                              # 10 classes
my_train, my_test = TensorDataset(x[:5000], y[:5000]), TensorDataset(x[5000:], y[5000:])

r = clt.run(data=(my_train, my_test), model="mlp", strategy="er", num_tasks=5, epochs=3)
```

An image folder works too: `clt.run(data="/path/to/images", ...)` (one sub-folder per class).

## Your model

Any `nn.Module` that turns an input into a feature vector:

```python
my_net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))

r = clt.run(data="toy", model=clt.models.wrap(my_net, embed_dim=64), strategy="er",
            freeze_backbone=False)          # your model is untrained, so train it
```

Pre-trained models by name: `model="vit_base_patch16_224.augreg_in21k"` (timm),
`model="hf:bert-base-uncased"` or `"hf:Qwen/Qwen2.5-0.5B-Instruct"` (Hugging Face),
`model="open_clip:ViT-B-16/openai"`. They are frozen by default; `freeze_backbone=False` fine-tunes.

## Your method

Subclass `Strategy` and change the loss:

```python
class SmallFeatures(clt.strategies.Strategy):
    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        feats = model.encode(x)                       # features
        logits = model.classify(feats, task_id)       # class scores
        loss = F.cross_entropy(logits, y) + 1e-3 * feats.pow(2).mean()
        return loss, logits

r = clt.run(data="toy", model="mlp", strategy=SmallFeatures())     # no memory, so it forgets old tasks
```

In class-incremental training, the scores of classes not seen yet are `-inf`, so they can never be
predicted. `F.cross_entropy` handles that; losses that spread probability over all classes
(e.g. label smoothing) must use only the seen ones: `logits[:, self.scenario.seen_classes(task_id)]`.

Do something between tasks with `on_task_start` / `on_task_end`:

```python
class CountSamples(clt.strategies.Strategy):
    def on_task_end(self, task_id, train_loader, model):
        print(f"task {task_id}: {len(train_loader.dataset)} training samples")

r = clt.run(data="toy", model="mlp", strategy=CountSamples(), num_tasks=3)
```

All hooks: `on_task_start`, `compute_loss`, `before_update` (after backward, before the step),
`on_task_end`, `before_eval`, `evaluate_task`. Inside them, `self.scenario` and `self.task` tell
you where you are.

Add memory to your method with the built-in replay buffer:

```python
from cl_toolkit.strategies.replay import ReservoirBuffer

class MyReplay(clt.strategies.Strategy):
    def __init__(self, memory_size=200):
        self.memory = ReservoirBuffer(memory_size)

    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        loss, logits = self.ce(model, x, y, task_id)                 # cross-entropy on new data
        old_x, old_y, old_t, _ = self.memory.sample(len(x), x.device)
        if old_x is not None:
            with self.memory_mask(model, task_id):                   # old samples: all seen classes
                loss = loss + F.cross_entropy(model(old_x, old_t), old_y)
        self.memory.add(x, y, task_id)
        return loss, logits

r = clt.run(data="toy", model="mlp", strategy=MyReplay())
```

Combine it with anything built in:

```python
r = clt.run(data="toy", model="mlp", strategy=MyReplay(), classifier="ncm")                 # + a head
r = clt.run(data="toy", model="mlp", strategy=clt.strategies.LoRAAdapter(MyReplay()))       # + LoRA per task
```

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

## Check your method

Add one line to `CASES` in `tests/test_integrity.py`, e.g.
`"my-replay": (mlp, vec_bench, dict(strategy=MyReplay()))`, and run
`pytest tests/test_integrity.py -k my-replay`. It checks that your method never touches test data,
never uses the task id at test time, does not peek at labels, and does not learn during evaluation.

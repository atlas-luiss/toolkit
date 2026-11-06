# 8. Write the CL loop yourself

`clt.run` and `trainer.fit()` run this loop for you:

```
start
for each task:
    train on the task          (the strategy's hooks from step 7 run in here)
    evaluate on every task seen so far
finish                         (metrics: final accuracy, forgetting, ...)
```

A strategy (step 7) changes what happens *inside* training. When you want to act *between*
the steps (merge models after each task, change what gets evaluated, save something, skip a
task), write the loop yourself. The trainer gives you each step as one call.

```python
import torch
import cl_toolkit as clt
from cl_toolkit import merging
from cl_toolkit.benchmarks import toy_data, split_by_class

train, test = toy_data()
tasks = split_by_class(train, test, num_tasks=5)       # steps 1 and 2
```

## a) The loop, written out

```python
torch.manual_seed(0)
model    = clt.models.mlp()
scenario = clt.Scenario(tasks, "class")
trainer  = clt.ContinualTrainer(model, scenario, clt.strategies.NaiveStrategy(),
                                epochs_per_task=1, verbose=False)   # 1 pass over each task's data

trainer.start()                                     # set up the classifier, the strategy, the results
for task in scenario:
    trainer.train_task(task)                        # train on this task's data only
    accs = trainer.evaluate(task.task_id)           # test on every task seen so far
    print(f"after task {task.task_id}:", [round(a) for a in accs])
results = trainer.finish()                          # the same results dict as clt.run returns

print("final accuracy", round(results["avg_accuracy"], 1))
```

This is exactly what `trainer.fit()` does. Each task is trained for `epochs_per_task` passes
over its data (default 1; `clt.run(epochs=...)` sets the same thing). Plain fine-tuning forgets: after each task only that
task is remembered (`[0, 0, 0, 0, 100]` at the end).

## b) Merge the models after each task

The idea of model merging, in plain words:

* `base` = the weights before any training.
* After training on task `t`, the weights changed. The change, `weights - base`, is the
  **task vector** of task `t`: what the model learned on it.
* A **merging rule** combines the task vectors of all tasks so far into one (average them, keep
  the largest change of each weight, ...). The model we test is `base + merged`.

In the loop, that goes between training and evaluation:

```python
def backbone_weights(model):
    return {name: p.detach().clone() for name, p in model.backbone.named_parameters()}

torch.manual_seed(0)
model    = clt.models.mlp()
scenario = clt.Scenario(tasks, "class")
trainer  = clt.ContinualTrainer(model, scenario, clt.strategies.NaiveStrategy(),
                                epochs_per_task=1, verbose=False)   # 1 pass over each task's data

trainer.start()
base = backbone_weights(model)
task_vectors = []
for task in scenario:
    trainer.train_task(task)

    finetuned = backbone_weights(model)
    task_vectors.append(merging.task_vector(finetuned, base))         # what this task changed
    merged = merging.apply(base, merging.merge(task_vectors, "ta"))  # base + combined change
    model.backbone.load_state_dict(merged, strict=False)             # test the merged model

    trainer.evaluate(task.task_id)

    model.backbone.load_state_dict(finetuned, strict=False)          # keep training from here
results = trainer.finish()
print("fine-tune + merging", round(results["avg_accuracy"], 1))
```

Plain fine-tuning ends at 20.0%; with merging, 31.4%. Every rule from `merging.METHODS` works
here: `"ta"` (task arithmetic), `"average"`, `"ties"`, `"magmax"`, `"dare"`, `"tsv"`, `"iso_c"`...

This loop is the built-in `MergeAfterTask` written out, and gives the same result:

```python
torch.manual_seed(0)
builtin = clt.strategies.MergeAfterTask(clt.strategies.NaiveStrategy(), method="ta")
r = clt.ContinualTrainer(clt.models.mlp(), clt.Scenario(tasks, "class"), builtin, epochs_per_task=1,
                         verbose=False).fit()
print("built-in", round(r["avg_accuracy"], 1))                       # also 31.4
```

The gain is small here because merging is meant for **pre-trained** models: their task vectors
are small changes around good weights, and combine well. With CLIP ViT-B/16 on CIFAR-100,
MagMax reaches 81.1% (the paper reports 80.4%); see `experiments/merging.sh`.

## c) Change it your way

To try variations, put the same loop in a function with a few choices:

```python
def merge_after_tasks(rule="ta", continue_from="finetuned", strategy=None):
    torch.manual_seed(0)
    model    = clt.models.mlp()
    scenario = clt.Scenario(tasks, "class")
    trainer  = clt.ContinualTrainer(model, scenario, strategy or clt.strategies.NaiveStrategy(),
                                    epochs_per_task=1, verbose=False)
    trainer.start()
    base, task_vectors = backbone_weights(model), []
    for task in scenario:
        trainer.train_task(task)
        finetuned = backbone_weights(model)
        task_vectors.append(merging.task_vector(finetuned, base))

        combined = rule(task_vectors) if callable(rule) else merging.merge(task_vectors, rule)
        model.backbone.load_state_dict(merging.apply(base, combined), strict=False)

        trainer.evaluate(task.task_id)

        if continue_from == "finetuned":
            model.backbone.load_state_dict(finetuned, strict=False)
        # continue_from="merged": keep the merged weights and train the next task from them
    return round(trainer.finish()["avg_accuracy"], 1)

print("magmax               ", merge_after_tasks("magmax"))
print("continue from merged ", merge_after_tasks("ta", continue_from="merged"))
print("replay + merging     ", merge_after_tasks("ta", strategy=clt.strategies.ExperienceReplay(memory_size=200)))
```

`strategy=` takes any method, built in or your own from step 7. Measure each choice: here,
continuing from the merged model is worse (22.3% vs 31.4%), and merging *lowers* replay
(81.9%, against 90.9% for replay alone), because this small model starts untrained.

**Your own merging rule** is a function: a list of task vectors in (each a `{name: tensor}`
dict), one combined task vector out. For example, newer tasks count more:

```python
def recent_first(task_vectors, decay=0.5):
    n = len(task_vectors)
    weights = [decay ** (n - 1 - i) for i in range(n)]       # ..., 0.25, 0.5, 1 for the newest
    return {name: sum(w * tv[name] for w, tv in zip(weights, task_vectors))
            for name in task_vectors[0]}

print("recent first         ", merge_after_tasks(recent_first))
```

To use your rule by name everywhere (also in `clt.run(strategy=...)` through
`MergeAfterTask(..., method="recent_first")`), register it:
`merging.METHODS["recent_first"] = recent_first`.

## d) No trainer at all: plain PyTorch

The trainer only saves you bookkeeping. Here is a whole Class-IL run in plain PyTorch, so you can
see every step:

```python
import torch.nn.functional as F

torch.manual_seed(0)
model    = clt.models.mlp()
scenario = clt.Scenario(tasks, "class")
scenario.setup(model)                          # adds the classifier: one output per class

for task in scenario:
    t = task.task_id

    # train on this task
    model.train()
    model.set_class_mask(scenario.seen_classes(t))         # classes not seen yet can't be predicted
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=1e-3)
    for x, y in task.train_loader:                        # 1 epoch: one pass over the data
        x, y = x.to(model.device), y.to(model.device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate on every task seen so far (no task id given to the model: Class-IL)
    model.eval()
    accs = []
    with torch.no_grad():
        for old in scenario.tasks[: t + 1]:
            correct = total = 0
            for x, y in old.test_loader:
                x, y = x.to(model.device), y.to(model.device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
            accs.append(round(100 * correct / total))
    print(f"after task {t}:", accs)
```

The trainer adds the rest on top of this loop: the strategy's hooks (step 7), callbacks,
mixed precision, the accuracy matrix and metrics, and the log.

**Next:** [9. Your own model](9_your_model.md)

# Getting started

## 1. Run something (5 seconds, CPU)

```python
import cl_toolkit as clt

results = clt.run(data="toy", model="mlp", strategy="er")
print(results["avg_accuracy"])
```

`toy` is a small built-in dataset and `mlp` a small model, so you can try anything without a
download. For real experiments, change the names:

```python
# results = clt.run(data="cifar100", root="./data", strategy="er")    # ViT-B/16 by default
```

The log, the settings and the results are saved in `runs/<time>_toy_er/`.

## 2. Try other methods

```python
for method in ["naive", "er", "derpp", "lwf"]:
    r = clt.run(data="toy", model="mlp", strategy=method)
    print(method, round(r["avg_accuracy"], 1))
```

```python
r = clt.run(data="toy", model="mlp", strategy="er", strategy_kwargs={"memory_size": 100})   # settings
r = clt.run(data="toy", model="mlp", strategy="frozen", classifier="ncm", epochs=0)           # a head, no training
r = clt.run(data="toy", model="mlp", experts="lora", router="lda")                           # one expert per task
r = clt.run(data="toy", model="mlp", strategy="er", scenario="task", num_tasks=5, epochs=3)  # protocol, tasks, epochs
print(clt.available())                                                                       # every name
```

## 3. Read the results

```python
r = clt.run(data="toy", model="mlp", strategy="er", num_tasks=5)

print(r["accuracy_matrix"].round(1))     # row i: accuracy on each task after learning task i
print(r["avg_accuracy"])                 # final accuracy (mean of the last row)
print(r["avg_forgetting"])               # how much old tasks dropped
```

## 4. The same thing by hand

`clt.run` puts five pieces together. You can build them yourself and swap any one:

```python
from cl_toolkit.benchmarks import toy_data, split_by_class

train, test = toy_data()                                         # any datasets returning (x, y)
tasks    = split_by_class(train, test, num_tasks=5)              # 1. data, split into tasks
model    = clt.models.mlp()                                      # 2. model (encoder)
scenario = clt.Scenario(tasks, "class")                          # 3. protocol
strategy = clt.strategies.ExperienceReplay(memory_size=200)      # 4. CL method
trainer  = clt.ContinualTrainer(model, scenario, strategy)       # 5. training loop

results = trainer.fit()
```

Next: [customize.md](customize.md) — your own data, model, method, head, router or merging rule.

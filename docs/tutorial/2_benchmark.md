# 2. Create the benchmark

In continual learning the model does not see all the data at once. It sees a **sequence of
tasks**, one after the other, and must not forget the earlier ones. A **benchmark** is that
sequence: which samples belong to task 1, task 2, and so on.

```python
import cl_toolkit as clt
from cl_toolkit.benchmarks import toy_data, split_by_class

train, test = toy_data()                               # your (train, test) from step 1
benchmark = split_by_class(train, test, num_tasks=5)   # 20 classes -> 5 tasks of 4 classes

for task in benchmark.get_splits():
    print(f"task {task.task_id}: classes {task.classes}, "
          f"{len(task.train_dataset)} train / {len(task.test_dataset)} test samples")
```

`split_by_class` puts every class in exactly one task. Useful options:

```python
fixed  = split_by_class(train, test, num_tasks=5, class_order=list(range(20)))   # your class order
other  = split_by_class(train, test, num_tasks=5, seed=1)                        # another random order
big_t0 = split_by_class(train, test, num_tasks=6, initial_classes=10)            # 10 classes first, then 2 per task
print(big_t0)
```

## The same classes, a changing input distribution

Sometimes every task has the same classes but the inputs change, e.g. photos, then sketches,
then paintings. Give one `(train, test)` pair per task (a "domain") to `split_by_domain`:

```python
from cl_toolkit.benchmarks import split_by_domain

photos   = toy_data(seed=0)
sketches = toy_data(seed=1)          # stand-ins for your three domains
drawings = toy_data(seed=2)

domains = split_by_domain([photos, sketches, drawings])
print(domains)
```

## Few examples per class

`few_shot` keeps only `k` training samples per class (the test sets are unchanged). With
`skip_first=True` the first task stays complete, which is the usual few-shot CL protocol:

```python
from cl_toolkit.benchmarks import few_shot

fewshot = few_shot(big_t0, k=5, skip_first=True)
print(fewshot)
```

## Use it

Pass the benchmark to `clt.run`. You can also skip this step: `clt.run(data=(train, test),
num_tasks=5)` calls `split_by_class` for you, and named datasets (`data="cifar100"`) do it too.

```python
results = clt.run(data=benchmark, model="mlp", strategy="er")
```

**Next:** [3. Choose a scenario](3_scenario.md)

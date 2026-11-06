# 3. Choose a scenario

The benchmark says *what* the tasks are. The **scenario** says *how the model is tested*:
what it knows at test time and among which classes it must choose.

| `scenario=` | at test time the model... | example |
|---|---|---|
| `"class"` | gets an image and must pick among **all classes seen so far**; it is not told which task the image comes from | after learning cats/dogs, then cars/bikes: "which of the 4 is this?" |
| `"task"` | is **told which task** the image comes from and picks among that task's classes | "this is from task 2: car or bike?" |
| `"domain"` | gets an input from any domain; the classes are the same in every task | "cat or dog?" for photos, sketches and paintings |

`"class"` is the hardest and the most common in papers; `"task"` is much easier because the task
id narrows the choice.

```python
import cl_toolkit as clt

for scenario in ["class", "task"]:
    r = clt.run(data="toy", model="mlp", strategy="naive", scenario=scenario, num_tasks=5)
    print(scenario, round(r["avg_accuracy"], 1))
```

Plain fine-tuning (`naive`) forgets almost everything in `"class"`, but much less in `"task"`.

For `"domain"`, give a domain benchmark (step 2):

```python
from cl_toolkit.benchmarks import toy_data, split_by_domain

domains = split_by_domain([toy_data(seed=0), toy_data(seed=1), toy_data(seed=2)])
r = clt.run(data=domains, model="mlp", strategy="naive", scenario="domain")
print("domain", round(r["avg_accuracy"], 1))
```

## Reading the results

After each task, the model is tested on every task seen so far. That gives the accuracy matrix:

```python
r = clt.run(data="toy", model="mlp", strategy="er", scenario="class", num_tasks=4)

print(r["accuracy_matrix"].round(1))   # row i = after learning task i, column j = test on task j
print(r["avg_accuracy"])               # final accuracy: mean of the last row
print(r["avg_forgetting"])             # how much the old tasks dropped
```

The same numbers, the full log and every setting are saved in `runs/<time>_toy_er/`.

**Next:** [4. LoRA, adapters and prompts](4_lora_adapters_prompts.md)

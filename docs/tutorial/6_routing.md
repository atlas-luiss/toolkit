# 6. Routing

Another way to avoid forgetting: train **one small expert per task** (a LoRA, an adapter or a
set of prompts) and never change it again. At test time a **router** looks at the input and
decides which task it probably comes from, and that task's expert makes the prediction.

```
input ──► router: "this looks like task 2" ──► expert 2 ──► class
```

Nothing is overwritten, so the experts do not forget. The weak point is the router: if it picks
the wrong task, the answer is wrong. The toolkit reports how often it picks correctly
(`routing_accuracy`).

```python
import cl_toolkit as clt

quick = dict(data="toy", model="mlp", num_tasks=5)

r = clt.run(experts="lora", router="lda", **quick)
print("accuracy", round(r["avg_accuracy"], 1), "| routing", round(r["final_routing_accuracy"], 1))
```

## Choosing the experts and the router

`experts=` is what each task gets:

| `experts=` | works with |
|---|---|
| `"lora"` | any model |
| `"adaptformer"` | timm ViTs |
| `"prompt"` | timm ViTs |

`router=` is how the task is guessed, always from the features of the plain pre-trained model:

| `router=` | how it guesses | from the paper |
|---|---|---|
| `"hide"` | a classifier over all classes seen so far (kept unbiased with samples drawn from stored class statistics) | HiDe-Prompt |
| `"sprompts"` | nearest of a few cluster centres per task | S-Prompts |
| `"keyquery"` | a learned key per task, closest to the input | L2P / DualPrompt |
| `"expertgate"` | one small autoencoder per task; the one that reconstructs the input best | Expert Gate |
| `"lda"`, `"ncm"`, ... | any statistical head from step 5: predicted class, then its task | |

```python
for router in ["hide", "sprompts", "expertgate"]:
    r = clt.run(experts="lora", router=router, **quick)
    print(router, "routing", round(r["final_routing_accuracy"], 1), "accuracy", round(r["avg_accuracy"], 1))
```

With a ViT, prompts or adapters as experts:

```python
vit = dict(data="toy-images", model="vit_tiny_patch16_224.augreg_in21k", num_tasks=5, epochs=5)
r = clt.run(experts="adaptformer", router="hide", **vit)
print("adaptformer experts", round(r["avg_accuracy"], 1))
```

## Expert settings

```python
r = clt.run(experts="lora", router="hide",
            strategy_kwargs={"experts_kwargs": {"rank": 4},       # LoRA rank of each expert
                             "router_kwargs": {"epochs": 10}},    # train the HiDe router for 10 epochs
            **quick)
```

## A statistical head for every expert

By default each expert predicts with a trained linear head. Add `classifier=` to give every
expert its own statistical head (step 5), fitted on that expert's features:

```python
r = clt.run(experts="lora", router="hide", classifier="lda", **quick)
print("lora experts + lda heads", round(r["avg_accuracy"], 1))
```

In the `"task"` scenario the task id is given, so the router is not used.

**Next:** [7. Your own CL method](7_your_method.md)

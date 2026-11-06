# 4. LoRA, adapters and prompts

With a large **pre-trained** model (e.g. a ViT trained on ImageNet), you usually don't change its
weights. You freeze it and train only a few **small extra parameters**. This is cheaper and forgets
less. The toolkit has the three common kinds:

| kind | what it adds | works with |
|---|---|---|
| **LoRA** | a small low-rank update next to chosen weight matrices | any model |
| **AdaptFormer** (adapter) | a small bottleneck MLP next to each block's MLP | timm ViTs |
| **prompts** | a few learned tokens given to the attention layers | timm ViTs |

The examples below use a small pre-trained ViT (`vit_tiny_patch16_224.augreg_in21k`, downloaded
once, about 22 MB) on the synthetic image data, so they run in seconds. For real work, use
`data="cifar100"` and the default ViT-B/16.

```python
import cl_toolkit as clt

vit = "vit_tiny_patch16_224.augreg_in21k"
quick = dict(data="toy-images", model=vit, num_tasks=5, epochs=5)
```

## LoRA

`strategy="lora"` adds a new LoRA to the frozen model for each task. Old LoRAs are frozen and
stay active, so the model is "pre-trained + all task updates":

```python
r = clt.run(strategy="lora", **quick)
print("lora", round(r["avg_accuracy"], 1))
```

On its own, LoRA still forgets in `"class"` scenarios (the classifier is biased towards the
newest classes). Combine it with replay of a few stored samples:

```python
r = clt.run(strategy="lora-er", strategy_kwargs={"memory_size": 200}, **quick)
print("lora + replay", round(r["avg_accuracy"], 1))
```

**Where LoRA goes.** List the layers it can be put on, then pick names and blocks:

```python
model = clt.models.ViTModel(vit)
print(clt.adapters.lora_candidates(model))        # e.g. ['fc1', 'fc2', 'proj', 'qkv']

r = clt.run(strategy="lora", lora_targets=["qkv", "fc1"], lora_layers=[10, 11],
            strategy_kwargs={"rank": 4}, **quick)  # LoRA rank 4 on qkv and fc1, last two blocks only
```

**LoRA around any method.** `LoRAAdapter` gives any strategy the one-LoRA-per-task setup:

```python
method = clt.strategies.LoRAAdapter(clt.strategies.DERPP(memory_size=200), rank=8)
r = clt.run(strategy=method, **quick)
print("lora + der++", round(r["avg_accuracy"], 1))
```

## Adapters (AdaptFormer)

The common way to use an adapter in CL is **first-session adaptation** (APER, RanPAC): train the
adapter on the first task only, then freeze everything and classify with a head that needs no
training (step 5):

```python
r = clt.run(strategy="first-session", strategy_kwargs={"first_session": "adaptformer"},
            classifier="ncm", **quick)
print("adaptformer on task 0 + NCM", round(r["avg_accuracy"], 1))
```

`first_session="lora"` does the same with LoRA. Adapters as one expert per task are in step 6.

## Prompts

L2P, DualPrompt and CODA-Prompt keep the ViT frozen and learn prompts that are chosen per
image:

```python
for method in ["l2p", "dualprompt", "coda-prompt"]:
    r = clt.run(strategy=method, lr=1e-2, **quick)
    print(method, round(r["avg_accuracy"], 1))
```

Change their settings with `strategy_kwargs`, e.g. `strategy="coda-prompt",
strategy_kwargs={"pool_size": 50, "length": 8}`. Prompt methods need many more training steps
than this tiny example has, so the numbers above mean little. On CIFAR-100 with 20 epochs and
`--preprocess coda`, the toolkit's CODA-Prompt gets 85.7% (paper: 86.3%).

**Next:** [5. Choose the classifier](5_classifier.md)

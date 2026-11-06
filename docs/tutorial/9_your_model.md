# 9. Your own model

The toolkit needs one thing from a model: turn an input into a **feature vector**. It adds the
classifier (step 5) itself. So any network without its final classification layer works.

```python
import torch
import torch.nn as nn
import cl_toolkit as clt
```

## a) Wrap any network

`clt.models.wrap` takes any `nn.Module` and the size of its output features:

```python
net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64))   # 32 inputs -> 64 features

model = clt.models.wrap(net, embed_dim=64)
r = clt.run(data="toy", model=model, strategy="er", num_tasks=5, freeze_backbone=False)
print(round(r["avg_accuracy"], 1))
```

`freeze_backbone=False` because this network is untrained, so it must learn. Models you pass in
are frozen by default, because that is what you want for a pre-trained model.

An image network works the same way:

```python
cnn = nn.Sequential(
    nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
    nn.AdaptiveAvgPool2d(4), nn.Flatten(),          # 32 channels x 4 x 4 = 512 features
)
r = clt.run(data="toy-images", model=clt.models.wrap(cnn, embed_dim=512), strategy="er",
            num_tasks=5, epochs=3, freeze_backbone=False)
print(round(r["avg_accuracy"], 1))
```

## b) Write a model class

For full control (several outputs, pooling, pre-processing), subclass `CLModel`. Store the network
as `self._backbone` (freezing, LoRA and merging act on it) and say how to get features:

```python
class MyModel(clt.models.CLModel):
    def __init__(self):
        super().__init__()
        self._backbone = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64))

    def _encode(self, x):            # input -> (batch, embed_dim) features
        return self._backbone(x)

    @property
    def embed_dim(self):
        return 64

r = clt.run(data="toy", model=MyModel(), strategy="er", num_tasks=5, freeze_backbone=False)
```

## c) Pre-trained models by name

| source | example |
|---|---|
| timm (ViT, DINOv2, CLIP, EVA, ConvNeXt...) | `model="vit_base_patch16_224.augreg_in21k"` (the default) |
| Hugging Face (BERT, Llama, Qwen...) | `model="hf:bert-base-uncased"`, `model="hf:Qwen/Qwen2.5-0.5B-Instruct"` |
| OpenCLIP | `model="open_clip:ViT-B-16/openai"` |

These are frozen by default; `freeze_backbone=False` fine-tunes them. The tokenizer (text) or the
image preprocessing comes with the model. For example, text classification with an LLM:
`clt.run(data="hf:fancyzhx/ag_news", model="hf:Qwen/Qwen2.5-0.5B-Instruct", strategy="lora-er")`.

## LoRA on your model

LoRA (step 4) works on any model. List the layers it can go on, then choose:

```python
model = MyModel()
print(clt.adapters.lora_candidates(model))           # layer names, here ['0', '2']

r = clt.run(data="toy", model=model, strategy="lora", lora_targets=["2"], num_tasks=5)
```

That's the whole toolkit. For more extension points (your own classifier head, router or merging
rule; changing the optimiser or the training loop), see [customize.md](../customize.md).

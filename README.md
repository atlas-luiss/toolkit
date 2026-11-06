# cl_toolkit — Continual Learning Toolkit

A [Collage Lab](https://www.vincenzolomonaco.com/collage-lab) initiative. Continual learning
experiments in one line: pick the **data**, a **model**, a **scenario** and a **strategy**, and
replace any of them with your own when you need to. Every run is logged.

```python
import cl_toolkit as clt

results = clt.run(data="toy", model="mlp", strategy="er")          # built-in toy data: seconds on a CPU
results = clt.run(data="cifar100", root="./data", strategy="er")   # real data, ViT-B/16 by default
```

**Guides:** [getting started](docs/getting_started.md) (one page) ·
[tutorial](docs/tutorial/README.md) (step by step: data, benchmark, scenario, LoRA / adapters / prompts,
classifiers, routing, your own method, your own model) · [customize](docs/customize.md) ·
[examples](examples/) (scripts on real data)

## Install

```bash
pip install -e .            # torch, torchvision, timm, peft
pip install -e ".[dev]"     # + pytest     (other extras: text, clip, log)
```

## Command line

```bash
python -m cl_toolkit --data toy --model mlp --strategy er
python -m cl_toolkit --data cifar100 --strategy er --set memory_size=2000
python -m cl_toolkit --data cifar100 --strategy frozen --classifier ranpac --epochs 0
python -m cl_toolkit --data cifar100 --experts lora --router hide
python -m cl_toolkit --list                                    # every accepted name
```

Each run writes `log.txt`, `config.json` (every setting that ran) and `results.json` (accuracy
matrix and metrics) to `runs/<time>_<data>_<strategy>/`.

## The four pieces

| piece | names you can pass | or your own |
|---|---|---|
| `data` | `toy`, `cifar100`, `cifar10`, `imagenet-r`, `imagenet-a`, `cub200`, `hf:<text dataset>`, an image folder | `(train, test)` datasets, a `CLBenchmark` |
| `model` | `mlp`, any timm name (ViT, DINOv2, CLIP, EVA), `hf:<model>` (BERT, LLMs), `open_clip:<arch>/<tag>` | any `nn.Module` encoder, a `CLModel` |
| `scenario` | `class`, `task`, `domain` | a `Scenario` |
| `strategy` | see below | a `Strategy` subclass |

The default model is ViT-B/16 pre-trained on ImageNet-21k (`vit_base_patch16_224.augreg_in21k`).
Pre-trained backbones are frozen unless `freeze_backbone=False` (`--finetune-backbone`).

Image preprocessing (`--preprocess`): `model` (default) uses the backbone's own normalisation;
`pilot` and `coda` are the exact transforms of the RanPAC/APER and CODA-Prompt code bases, for
comparing with papers.

## What's included

**Strategies** (`strategy=`)

| family | names |
|---|---|
| baselines | `naive`, `frozen` (no training), `first-session` (adapt on task 0 only) |
| replay | `er`, `er-ace`, `der`, `derpp`, `agem`, `icarl`, `gdumb` |
| regularisation | `ewc`, `ewc-online`, `mas`, `lwf` |
| PEFT | `lora`, `lora-er` (one LoRA per task; wrap any strategy with `LoRAAdapter`) |
| prompts | `l2p` (the L2P++ prefix variant), `dualprompt`, `coda-prompt` |
| pre-trained models | `simplecil`, `ranpac` (without Phase 1 by default; `--set first_session=adaptformer` adds it) |
| routing | `routed`, and presets `hide-prompt`, `s-prompts`, `expert-gate` (their routing part) |
| merging | `magmax`, `ties`, `task-arithmetic`, `dare`, `breadcrumbs`, `consensus`, `tsv`, `iso-c`, `iso-cts`, `weight-average`, `coma`, `cofima` (need `--finetune-backbone`) |

**Classifier heads** (`classifier=`), on top of any strategy

| name | paper |
|---|---|
| `cosine` | LUCIR — Hou et al., CVPR 2019 |
| `ncm` | iCaRL NME / SimpleCIL — Zhou et al., IJCV 2024 |
| `lda` | SLDA — Hayes & Kanan, CVPRW 2020 |
| `klda` | KLDA — Momeni et al., 2024 (`ensembles=5`: KLDA-E) |
| `fecam` | FeCAM — Goswami et al., NeurIPS 2023 |
| `ranpac` | RanPAC — McDonnell et al., NeurIPS 2023 |
| `slca` | SLCA classifier alignment — Zhang et al., ICCV 2023 |
| `wa` | Weight Aligning — Zhao et al., CVPR 2020 (alignment step) |
| `clip-zeroshot` | frozen CLIP text head, as in MagMax (with `model=open_clip:ViT-B-16/openai`) |

**Routing** (`experts=` + `router=`): one expert per task; at test time a router picks the expert
for each input from the plain pre-trained features, and the expert predicts among its task's
classes. Each evaluation logs `routing_accuracy`.

| `experts=` | `router=` |
|---|---|
| `lora`, `adaptformer`, `prompt` | `hide` (HiDe-Prompt), `sprompts` (S-Prompts), `keyquery` (L2P / DualPrompt), `expertgate` (Expert Gate), or any head (`lda`, `fecam`, ...) |

Add `classifier=lda` (or another statistical head) to give every expert its own head, fitted on
that expert's features.

**Building blocks** for your own methods: `cl_toolkit.adapters` (LoRA placement, AdaptFormer),
`cl_toolkit.prompts`, `cl_toolkit.heads`, `cl_toolkit.routing`, `cl_toolkit.merging`.

## Scenarios

| `scenario=` | at test time | head |
|---|---|---|
| `class` | no task id; predict among all classes seen so far | one shared head |
| `task` | the task id is given | one head per task |
| `domain` | no task id; same classes in every task | one shared head |

Strategies receive `task_id=None` at evaluation unless the scenario is Task-IL, so no method can
use the task identity by accident.

## Results

**Reproduced published numbers** (Split CIFAR-100, 10 tasks, 3 seeds, each paper's backbone,
preprocessing and training settings; scripts in `experiments/`):

| method | ours | paper |
|---|---|---|
| NCM (ViT-B/16 IN21K, frozen) | 83.34 | 83.4 (RanPAC) |
| RanPAC, no Phase 1 | 88.93 | 89.0 (RanPAC) |
| FeCAM | 85.67 | 85.7 (FeCAM) |
| CODA-Prompt | 85.68 ± 0.43 | 86.25 ± 0.74 (CODA-Prompt) |
| DualPrompt | 81.87 ± 0.19 | 83.05 ± 1.16 (CODA-Prompt) |
| L2P++ | 81.22 ± 0.72 | 82.50 ± 1.10 (CODA-Prompt) |
| MagMax (CLIP ViT-B/16, sequential fine-tuning) | 81.11 ± 0.37 | 80.41 (MagMax) |
| TIES, same setting | 80.27 ± 0.36 | 77.23 (MagMax) |

**Toolkit defaults** (Split CIFAR-100, 10 tasks, Class-IL, ViT-B/16 IN21K frozen, AdamW lr 1e-3,
batch 64, 1 epoch per task, 3 seeds). Each row is the command shown plus `--data cifar100`
(`experiments/reference.sh`); use them to check your setup.

| command | final accuracy | forgetting |
|---|---|---|
| `--strategy frozen --classifier ranpac --epochs 0` | 89.7 ± 0.0 | 3.7 |
| `--strategy naive --classifier slca` | 88.7 ± 0.2 | 2.8 |
| `--experts lora --router hide --classifier lda` | 88.6 ± 0.2 | 4.4 |
| `--strategy frozen --classifier lda --epochs 0` | 88.3 ± 0.0 | 4.3 |
| `--strategy frozen --classifier klda --epochs 0` | 88.3 ± 0.0 | 4.4 |
| `--experts lora --router hide` | 88.3 ± 0.1 | 4.4 |
| `--experts adaptformer --router hide` | 88.3 ± 0.0 | 4.3 |
| `--experts prompt --router hide` | 87.6 ± 0.2 | 3.9 |
| `--strategy lora-er --set memory_size=2000` | 85.0 ± 0.6 | 13.5 |
| `--strategy er-ace --set memory_size=2000` | 81.4 ± 0.3 | 7.7 |
| `--strategy frozen --classifier ncm --epochs 0` | 80.1 ± 0.0 | 5.6 |
| `--strategy frozen --classifier fecam --epochs 0` | 78.5 ± 0.0 | 5.5 |
| `--strategy coda-prompt` | 78.3 ± 0.4 | 6.4 |
| `--strategy er --set memory_size=2000` | 77.5 ± 0.4 | 19.2 |
| `--strategy naive` | 12.0 ± 0.6 | 93.6 |

These differ from the reproductions above because the defaults are not the papers' settings
(e.g. 1 epoch, and `--preprocess model` instead of `pilot`: NCM 80.1 vs 83.3).

**Metrics** in `results`: `accuracy_matrix[i, j]` (accuracy on task j after training task i),
`avg_accuracy` (final), `avg_incremental_accuracy`, `avg_forgetting`, `backward_transfer`,
`accuracy_all_seen` (over all seen test samples), and method-specific ones such as
`final_routing_accuracy`.

## Tests

```bash
python -m pytest            # CPU, ~2 minutes
```

`tests/test_integrity.py` runs every method and checks that it never reads test data during
training, never uses the task identity at test time, does not depend on test labels, does not
learn during evaluation, keeps frozen weights frozen, and respects its memory budget.
`experiments/interop_grid.py` runs every method on every scenario with five kinds of model
(a `CLModel`, a wrapped `nn.Module`, a timm ViT, BERT and a decoder LLM).

## Repository

```
cl_toolkit/         the package (api.py: clt.run / clt.build; __main__.py: the CLI)
  benchmarks/       datasets -> tasks         models/      CLModel, ViT, Hugging Face, CLIP, wrap()
  strategies/       every method              heads.py     classifier heads
  routing.py        experts + routers         merging.py   task-vector merging
  adapters.py       LoRA, AdaptFormer         prompts.py   L2P, DualPrompt, CODA-Prompt
  trainers/         the training loop         callbacks/   hooks into the loop
  contrib/llm/      experimental causal-LM fine-tuning (not part of the tested core)
docs/               getting started, tutorial (8 steps), customize; every code block is tested
examples/           scripts on real data: quick start, own data, own method, heads, routing, merging, LLMs
experiments/        scripts behind the results above (outputs go to runs/, not tracked)
tests/              pytest suite
```

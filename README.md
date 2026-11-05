# cl_toolkit — Continual Learning Toolkit

A [Collage Lab](https://www.vincenzolomonaco.com/collage-lab) initiative: continual learning
experiments that don't start from scratch. Pick **data**, a **model**, a **scenario** and a
**strategy**; replace any of them with your own when you need to. Every run is logged.

Every method is checked for test-data leakage, task-id leakage and frozen-weight changes, and
compared with its paper; several published numbers are reproduced exactly. What each name
implements, how it differs from its paper, the reproductions, and an interoperability grid
(every method x scenario x model kind): [VERIFICATION.md](VERIFICATION.md).

**Guides:** [getting started](docs/from_scratch.md) · [customize](docs/customize.md) (your own
data, model, method, head, router, merging rule, training loop) · [examples/](examples/) (scripts on real data).

## Install

```bash
pip install -e .            # core: torch, torchvision, timm, peft
pip install -e ".[dev]"     # + pytest        (other extras: text, clip, log)
```

## 30 seconds

```python
import cl_toolkit as clt

results = clt.run(data="toy", model="mlp", strategy="er")     # built-in toy data: seconds on a CPU
results = clt.run(data="cifar100", root="./data", strategy="er")   # real data, ViT-B/16 by default
```

```bash
python -m cl_toolkit --data toy --model mlp --strategy er
python -m cl_toolkit --list                                   # every accepted name
python -m cl_toolkit --data cifar100 --strategy er --set memory_size=2000
python -m cl_toolkit --data cifar100 --strategy frozen --classifier ranpac --epochs 0
python -m cl_toolkit --data cifar100 --experts lora --router hide
```

Each run writes `log.txt`, `config.json` and `results.json` (accuracy matrix + metrics) to
`runs/<timestamp>_<data>_<strategy>/` (`out_dir=` / `--out` to change, `None` / `none` to disable).

## The four pieces

| piece | strings you can pass | or your own |
|---|---|---|
| `data` | `cifar100`, `cifar10`, `imagenet-r`, `imagenet-a`, `cub200`, `hf:<text dataset>`, an image folder | `(train_ds, test_ds)`, a `CLBenchmark` |
| `model` | any timm name (ViT, DINOv2, CLIP, EVA...), `hf:<model>` (BERT, LLMs), `open_clip:<arch>/<tag>` | any `nn.Module` encoder, a `CLModel` |
| `scenario` | `class`, `task`, `domain` | a `Scenario` |
| `strategy` | see below | a `Strategy` subclass |

Default model: ViT-B/16 pre-trained on ImageNet-21k (`vit_base_patch16_224.augreg_in21k`, the
"ViT-B/16-IN21K" of most PTM-CL papers). The backbone is frozen unless `freeze_backbone=False`
(`--finetune-backbone`). bf16 is used on GPU. The exact weights, preprocessing, optimizer and
the number of trainable backbone parameters are written to every run's `config.json`.

**Preprocessing** (`--preprocess`, image datasets): `model` (default) uses the backbone's own
normalisation and never crops low-resolution images; `pilot` and `coda` are the exact transforms
of the RanPAC/APER and CODA-Prompt code bases (no normalisation), for comparing with papers.
It matters: NCM on CIFAR-100 gets 80.1 with `model` and 83.4 with `pilot` (see VERIFICATION.md).

Optional on top of any strategy:

* `classifier=` — a head from a recent paper (below)
* `experts=` + `router=` — routing, one expert per task (below)
* `lora_targets=` / `lora_layers=` — where every LoRA-based method puts adapters
* `train_mask=` (`seen` / `current` / `none`) — which logits compete during training (Class-IL)

## What's included

**Strategies** (`strategy=`)

| family | names |
|---|---|
| baseline | `naive`, `frozen` (no training), `first-session` (adapt on task 0 only) |
| replay | `er`, `er-ace`, `der`, `derpp`, `agem`, `icarl`, `gdumb` |
| regularisation | `ewc`, `ewc-online`, `mas`, `lwf` |
| PEFT | `lora`, `lora-er` (one LoRA per task, all active; any strategy via `LoRAAdapter`) |
| prompts | `l2p` (= L2P++, prefix variant), `dualprompt`, `coda-prompt` |
| pre-trained models | `simplecil`, `ranpac` (default without Phase 1; `--set first_session=adaptformer` adds it) |
| routing | `routed`, presets `hide-prompt`, `s-prompts`, `expert-gate` (routing part only) |
| merging (needs `--finetune-backbone`) | `magmax`, `ties`, `task-arithmetic`, `dare`, `breadcrumbs`, `consensus`, `tsv`, `iso-c`, `iso-cts`, `weight-average` (any strategy via `MergeAfterTask`); `coma`, `cofima` (continual model averaging) |

**Classifier heads** (`classifier=`)

| name | paper |
|---|---|
| `cosine` | LUCIR — Hou et al., CVPR 2019 (trainable cosine layer) |
| `ncm` | iCaRL NME / SimpleCIL — Zhou et al., IJCV 2024 |
| `fecam` | FeCAM — Goswami et al., NeurIPS 2023 |
| `lda` | SLDA — Hayes & Kanan, CVPRW 2020 |
| `klda` | KLDA — Momeni et al., arXiv 2024 (≈ RanDumb); `ensembles=5` = KLDA-E |
| `ranpac` | RanPAC — McDonnell et al., NeurIPS 2023 |
| `slca` | SLCA classifier alignment — Zhang et al., ICCV 2023 |
| `wa` | Weight Aligning — Zhao et al., CVPR 2020 (alignment step only) |
| `clip-zeroshot` | frozen CLIP text-embedding head (with `model=open_clip:ViT-B-16/openai`), as in MagMax |

**Routing** (`experts=` × `router=`): one expert per task, picked per input at test time from
the plain pre-trained features. Each evaluation logs `routing_accuracy`.

| `experts=` | `router=` | paper |
|---|---|---|
| `lora` | `hide` | HiDe-Prompt — Wang et al., NeurIPS 2023 |
| `adaptformer` | `sprompts` | S-Prompts — Wang et al., NeurIPS 2022 |
| `prompt` | `keyquery` | L2P / DualPrompt — Wang et al., 2022 |
| | `expertgate` | Expert Gate — Aljundi et al., CVPR 2017 |
| | any head name (`lda`, `fecam`, ...) | class prediction → its task |

These implement the routing part of those papers on a shared backbone, not every
component of the original methods. After routing, the prediction is among the routed task's
classes (`within_task=True`, as in Expert Gate).

Adding `classifier=` with a statistical head (`lda`, `ncm`, `fecam`, `klda`, `ranpac`) replaces each
expert's linear head: after task t the head is fitted on task t's training data through expert t,
and at test time the routed task's head scores the routed expert's features, among that task's
classes (`--experts lora --router hide --classifier lda`). SLCA / WA are refused here: they correct
one shared head across tasks, and routed prediction is already within one task.

**Building blocks** for your own methods: `cl_toolkit.adapters` (LoRA, AdaptFormer),
`cl_toolkit.prompts`, `cl_toolkit.heads`, `cl_toolkit.routing`, `cl_toolkit.merging`
(task arithmetic, TIES, DARE, MagMax, Breadcrumbs, Consensus, TSV, Iso-C, Iso-CTS).

## Scenarios

| kind | head | labels | task id at test time |
|---|---|---|---|
| `class` | shared, restricted to classes seen so far | global | never |
| `task` | one per task | task-local | yes |
| `domain` | shared | global | never |

Strategies receive `task_id=None` at evaluation unless the scenario is Task-IL, so no method can
use task identity in Class-IL by accident.

## Your own pieces

```python
from cl_toolkit.benchmarks import split_by_class, split_by_domain, few_shot
bench = split_by_class(train_ds, test_ds, num_tasks=5)      # any dataset yielding (x, int y)
bench = few_shot(bench, k=5, skip_first=True)                # FSCIL

model = clt.models.wrap(my_encoder)                          # any nn.Module: inputs -> (B, D)
                                                             # inputs may be tensors or dicts (text)

from cl_toolkit.strategies import Strategy
class MyMethod(Strategy):
    def compute_loss(self, batch, model, task_id):
        x, y = self.to_device(batch, model)
        loss, logits = self.ce(model, x, y, task_id)
        return loss + my_regulariser(model), logits
    # optional: on_task_start, before_update, on_task_end, evaluate_task

clt.run(data=bench, model=model, strategy=MyMethod())
```

**LoRA placement**: `clt.adapters.lora_candidates(model)` lists the layer names; pass
`lora_targets=["qkv", "fc1"]` (names), a regex, or `"all-linear"`, and `lora_layers=[9, 10, 11]`
to restrict to blocks. A target that matches nothing is an error that lists the valid names.
The same settings apply to `lora`, `lora-er`, routed LoRA experts and LoRA first-session adaptation.

**The loop**: `trainer = clt.build(...)` returns the trainer without running it; add
`callbacks=[...]` (`EarlyStopping`, `GradientClipping`, `MetricLogger`, ...), or subclass
`ContinualTrainer` and pass `trainer_cls=`.

## Metrics

`results["accuracy_matrix"][i, j]` = accuracy on task j after training task i, plus
`avg_accuracy` (final, mean over tasks), `avg_incremental_accuracy`, `avg_forgetting`,
`backward_transfer`, `accuracy_all_seen` / `avg_incremental_accuracy_all_seen` (over all seen
test *samples*, as PTM-CL papers report; equal to the task means when tasks have equal test
sets), and any extra metric a strategy reports (e.g. `final_routing_accuracy`).

## Reference results

Split CIFAR-100, 10 tasks, Class-IL, toolkit defaults (ViT-B/16 IN21K frozen, `preprocess=model`,
AdamW lr 1e-3, batch 64, 1 epoch/task), mean ± std over 3 seeds. Each row is exactly the command
shown plus `--data cifar100 --seed <s>` (`experiments/reference.sh`). Use them to sanity-check your setup.

| run | final acc | forgetting |
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
| `--strategy coda-prompt` (1 epoch; 20 epochs + `--preprocess coda` reproduce the paper) | 78.3 ± 0.4 | 6.4 |
| `--strategy er --set memory_size=2000` | 77.5 ± 0.4 | 19.2 |
| `--strategy naive` | 12.0 ± 0.6 | 93.6 |

Frozen-feature methods have ~0 std: with no training, final accuracy over all 100 classes does
not depend on the class order. With `--preprocess pilot`, NCM and FeCAM gain 3-6 points and
RanPAC loses 0.8, which is why papers differ from this table; see VERIFICATION.md for the
paper-matched reproductions.

Merging, in the MagMax paper's setting (CLIP ViT-B/16, sequential full fine-tuning, CIFAR-100/10,
3 seeds), final accuracy: TSV-M 81.6, MagMax 81.1 (paper 80.4), TIES 80.3 (paper 77.2), averaging 79.3
(paper 77.0), no merge 72.9. The full table, including methods that are not suited to sequential
fine-tuning, is in VERIFICATION.md.

## Repository

```
cl_toolkit/
  api.py, __main__.py   clt.run / clt.build / clt.available, and the CLI
  benchmarks/           datasets -> tasks (images, text, folders, your own)
  models/               CLModel, ViTModel (timm), HFModel, CLIPModel, wrap()
  scenarios.py          Scenario(kind="class" | "task" | "domain")
  strategies/           every method above + the Strategy base class
  heads.py              classifier heads
  routing.py            expert pools + routers
  adapters.py           LoRA (placement, merge) and AdaptFormer
  prompts.py            L2P, DualPrompt, CODA-Prompt, per-task prompts
  merging.py            task-vector merging
  trainers/             ContinualTrainer
  callbacks/            hooks into the loop
  batch.py              tensor / dict batch helpers (what makes models interchangeable)
  contrib/llm/          experimental causal-LM fine-tuning (not part of the tested core)
examples/               01_quickstart ... 08_customize_the_loop, 09_llm
tests/                  pytest suite (CPU, ~2 min):  python -m pytest  (test_integrity.py: leakage checks)
VERIFICATION.md         what every method implements, deviations from papers, reproductions
docs/                   guides: from scratch, customise (every code block is tested)
experiments/            scripts behind VERIFICATION.md and the reference table (outputs in runs/, gitignored)
```

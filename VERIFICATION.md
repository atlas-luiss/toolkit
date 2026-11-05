# Verification

What every reported number relies on, how each method compares with its paper, and which
published numbers have been reproduced. Re-run everything with:

```bash
python -m pytest tests/test_integrity.py      # integrity + maths checks, CPU, ~30 s
bash experiments/repro.sh                     # reproductions of published numbers (GPU)
bash experiments/merging.sh <gpu>             # merging methods in the MagMax setting (GPU)
bash experiments/reference.sh <gpu>           # the README reference table (GPU)
python experiments/interop_grid.py            # every method x scenario x model kind (CPU)
```

## 1. Integrity checks (run on every registered method)

`tests/test_integrity.py` runs each of the 50 configurations (every strategy name, every
classifier head, every router, every expert type, every prompt method) on small models and
checks:

| check | how | catches |
|---|---|---|
| test data is read only during evaluation | test set is a tripwire dataset that raises if read outside an evaluation call | training / fitting / hyper-parameter selection on test data |
| no task identity at test time | final accuracy from the per-task test loaders must equal the accuracy on one merged loader | any use of *which* loader is being evaluated (a task-id leak) |
| predictions do not use test labels | relabel all test samples as class c, for every seen c: accuracies must sum to exactly 100% | any peeking at labels during evaluation |
| evaluation is read-only | evaluate twice: identical accuracies, bit-identical weights | test-time learning / state changed by evaluation |
| frozen means frozen | backbone weights bit-identical after training (all frozen-backbone methods) | accidental backbone updates |
| old tasks' modules stay fixed | after task k, task k's LoRA / expert / AdaptFormer / prompt parameters never change | leaking updates into old experts |
| memory budgets | buffer size <= `memory_size` after every task (ER, ER-ACE, DER, DER++, A-GEM, iCaRL, GDumb, LoRA-ER) | over-budget replay |
| metrics | accuracy matrix -> every metric, checked by hand | metric bugs |

`test_checks_catch_cheating` plants deliberately cheating strategies (peeks at labels,
infers the task from the loader, learns at test time, reads test data in training) and
asserts each check fails for them, so the checks are not vacuous.

Maths checked against independent numpy implementations: LDA (and streaming == batch),
NCM, FeCAM, RanPAC (exact ridge regression on the accumulated Gram matrix), KLDA's random
features (approximate the RBF kernel), reservoir sampling (uniform), A-GEM (projected
gradient never increases the memory loss), EWC (Fisher = mean per-sample squared
gradient; one anchor per task), TIES / MagMax (hand example), WA (equal mean norms).

## 2. Reproductions of published numbers

Split CIFAR-100, 10 tasks, Class-IL, 3 seeds (= 3 class orders), mean ± std. Each run uses
the paper's backbone, preprocessing and training settings (`experiments/repro.py`; logs in
`runs/repro/<name>_s<seed>/`). "Final" = accuracy on all classes after the last task.

| method | setting | ours (final) | paper | verdict |
|---|---|---|---|---|
| NCM | frozen ViT-B/16 IN21K, `preprocess=pilot` | **83.34 ± 0.01** | 83.4 (RanPAC paper, Tab. 1 / A4) | reproduced |
| RanPAC, no Phase 1 | same | **88.93 ± 0.00** | 89.0 (RanPAC paper, Tab. 1 / A4) | reproduced |
| FeCAM | frozen ViT-B/16 IN21K, authors' transform (ToTensor, Resize 224) | **85.67 ± 0.01** | 85.7 (FeCAM paper, Tab. 2) | reproduced |
| CODA-Prompt | ViT-B/16 IN1K, `preprocess=coda`, Adam 1e-3, bs 128, cosine, 20 epochs | **85.68 ± 0.43** | 86.25 ± 0.74 (CODA-Prompt paper, Tab. 2a) | within 1 std |
| DualPrompt | same | 81.87 ± 0.19 | 83.05 ± 1.16 (same table) | 1.2 below (1.0 std) |
| L2P++ | same | 81.22 ± 0.72 | 82.50 ± 1.10 (same table) | 1.3 below (1.2 std) |
| CLIP ViT-B/16 zero-shot | `classifier=clip-zeroshot`, no training | 62.74 | 62.78 (independent plain open_clip script, same weights / templates); 68.7 (CLIP paper) | toolkit = independent reference; the gap to the CLIP paper is not from the toolkit |

Not compared: forgetting (the CODA-Prompt paper uses a different forgetting definition);
full RanPAC with Phase 1, SLCA, HiDe-Prompt, CoMA (their full methods are not implemented or
need settings not run here).

**Preprocessing and weights matter as much as many method differences** (same code, frozen
ViT-B/16, 3 seeds):

| | NCM | RanPAC (no Phase 1) | FeCAM |
|---|---|---|---|
| IN21K, `pilot` (no normalisation) | 83.34 | 88.93 | 84.90 (1 seed) |
| IN21K, `model` (timm normalisation, toolkit default) | 80.09 | **89.69** | 78.49 |
| IN21K -> IN1K weights, `pilot` | 78.25 | 87.47 | — |
| IN21K, timm center-crop (old default) | 78.02 (1 seed) | — | — |

No setting is best for every method, so the preset is an explicit, logged choice.

**Merging, MagMax setting** (Marczak et al., ECCV 2024; `experiments/merging_magmax.py`, logs
`runs/merging/magmax_cifar100_s<seed>.out`, results `.../merges.json`). CLIP ViT-B/16 (openai),
frozen zero-shot text head, full image-encoder fine-tuning, training logits restricted to the
current task, 10 epochs per task, AdamW lr 1e-5, wd 0.1, bs 128, cosine, grad clip 1.0. Each task
starts from the previous fine-tuned weights, and every method merges the *same* 3 trajectories.
Task-agnostic evaluation on all seen classes; 3 seeds.

| method (library defaults unless noted) | ours final | ours avg-incr | paper (Tab. 2) |
|---|---|---|---|
| sequential fine-tuning, no merge | 72.89 ± 0.85 | 83.25 ± 1.20 | — |
| average (1/N) | 79.25 ± 0.20 | 86.96 ± 1.03 | 77.04 |
| TIES, λ = 0.55 (MagMax paper's setting) | 80.27 ± 0.36 | 87.45 ± 1.13 | 77.23 |
| TIES, λ = 1 (default) | 78.32 ± 0.46 | 86.70 ± 1.03 | — |
| **MagMax, λ = 0.5** | **81.11 ± 0.37** | 88.44 ± 0.99 | **80.41** |
| Breadcrumbs | 78.64 ± 0.29 | 86.22 ± 1.05 | — |
| TSV-M | **81.56 ± 0.44** | 88.47 ± 1.00 | — |
| Iso-CTS | 67.36 ± 1.34 | 83.14 ± 0.65 | — |
| Iso-C | 66.20 ± 1.64 | 83.31 ± 0.50 | — |
| Consensus (TALL k = 2, over TA) | 64.48 ± 0.46 | 79.74 ± 1.04 | — |
| Task Arithmetic, λ = 0.3 | 49.99 ± 0.49 | 77.75 ± 0.96 | — |
| DARE (p = 0.9) + TA | 49.51 ± 0.63 | 77.63 ± 0.94 | — |

Verdict: the paper's ranking reproduces (MagMax > TIES > Avg > sequential fine-tuning), with
all three of our numbers 0.7–3 points higher. Known differences: no LR warm-up (authors use 500
steps), and our class orders come from our seeds.

How to read the rest: TA, DARE+TA, Consensus, Iso-C and Iso-CTS were designed for task vectors
fine-tuned *independently* from θ0. In sequential fine-tuning each τ_t already contains τ_1..τ_{t-1},
so summing methods count early tasks many times. Their low numbers here reflect this setting, not
the methods in their own setting. Their own settings (independent fine-tuning, 8/14/20 vision
tasks) were not run, so those papers' numbers are **not** claimed as reproduced. Each is checked
against its paper's definition by the unit tests below.


## 3. Methods vs papers

"Follows" = checked against the paper and, where available, the authors' code.
"Differs" = everything that is not the paper's method. Names in `strategy=` / `classifier=`
mean exactly what this table says.

**Replay and regularisation**

| name | paper | follows | differs |
|---|---|---|---|
| `er` | Chaudhry et al. 2019 | reservoir memory, joint batch of stream + memory | — |
| `er-ace` | Caccia et al., ICLR 2022 | asymmetric CE: memory over all seen classes | incoming batch masked to the *current task's* classes; the paper masks to classes *present in the batch* (identical when every task class appears in a batch) |
| `der`, `derpp` | Buzzega et al., NeurIPS 2020 | MSE to stored logits + CE on memory, separate memory batches | logits of units that did not exist when stored are ignored |
| `agem` | Chaudhry et al., ICLR 2019 | projection `g - (g.g_ref / g_ref.g_ref) g_ref` if `g.g_ref < 0`, memory = past tasks only | equal memory quota per task |
| `icarl` | Rebuffi et al., CVPR 2017 | herding, sigmoid distillation from the old model, nearest-mean-of-exemplars | — |
| `gdumb` | Prabhu et al., ECCV 2020 | greedy class-balanced memory, retrain only on memory | re-initialises and retrains the *trainable* parameters (the head when the backbone is frozen); the paper retrains a network from scratch |
| `ewc` | Kirkpatrick et al., PNAS 2017 | diagonal Fisher with labels sampled from the model, one penalty per past task | Fisher from 512 training samples by default |
| `ewc-online` | Schwarz et al., ICML 2018 | decayed running Fisher, latest anchor | — |
| `mas` | Aljundi et al., ECCV 2018 | importance = mean per-sample \|d‖f(x)‖²/dθ\|, accumulated | 512 samples by default |
| `lwf` | Li & Hoiem, TPAMI 2017 | KD on old outputs, T = 2, λ = 1 | — |

**Pre-trained-model methods and classifier heads**

| name | paper | follows | differs |
|---|---|---|---|
| `ncm`, `simplecil` | SimpleCIL, Zhou et al., IJCV 2024 | cosine NCM on frozen features | `first_session=` adds APER-style adaptation |
| `ranpac` | McDonnell et al., NeurIPS 2023 | ReLU random projection M = 10000, ridge on accumulated Gram, λ chosen per task by validation MSE on an 80/20 split of that task's training data over 1e-8..1e8 (paper App. C.1 / code) | **default has no Phase 1** (the paper's "No Phase 1" ablation); `first_session="adaptformer"` adds Phase 1 |
| `lda` | SLDA, Hayes & Kanan, CVPRW 2020 | class means + one shared covariance, exact streaming | shrinkage 1e-2 × mean variance (SLDA: 1e-4) |
| `klda` | Momeni et al., arXiv 2412.15571 (2024) | RFF D = 5000, class means + shared covariance (argmax-equivalent to the code's negative Mahalanobis score); `ensembles=5` = KLDA-E (mean of member softmaxes, as in the paper's Eq. 14) | kernel width from the median heuristic (paper: "empirically determined" per foundation model, no rule given); small covariance shrinkage (code: pseudo-inverse, none); default is single-model KLDA (the code's `run.sh` uses 5 members) |
| `fecam` | Goswami et al., NeurIPS 2023 | checked line by line against the authors' `FeCAM_vit_cifar100.py`: L2-normalised [CLS] features, class means of those (not re-normalised), per-class covariance shrunk once (α1 = 1, α2 = 0), correlation normalisation, pseudo-inverse, squared Mahalanobis, no Tukey | — (their ResNet path uses Tukey β = 0.5 and shrinks twice; `tukey=True, shrink2=1` gets close) |
| `slca` | Zhang et al., ICCV 2023 | classifier alignment as in the authors' `models/slca.py`: full-covariance Gaussian per class (+1e-4 I), from task 1 on, 5 epochs, 256 fresh samples per class every epoch, batches of 256, SGD 0.01, m 0.9, wd 5e-4, cosine; means of task k scaled by 0.9 + 0.1 (k+1)/(t+1); logits divided by the mean of the per-task logit norms, temperature 0.1 | **classifier alignment only**; the slow-learner backbone fine-tuning (backbone lr = head lr / 100) is up to the strategy |
| `wa` | Zhao et al., CVPR 2020 | rescale new-class weight norms to the old classes' mean | **weight-aligning step only**: no distillation, no weight clipping |
| `cosine` | LUCIR, Hou et al., CVPR 2019 | cosine classifier with learned scale | the LUCIR losses are not included |

**Prompts** (defaults = the CODA-Prompt repo's CIFAR-100 settings; a prompt of length L = L/2 key + L/2 value prefix tokens)

| name | paper | follows | differs |
|---|---|---|---|
| `l2p` | Wang et al., CVPR 2022 | key-query top-5 from a pool, query = frozen [CLS] from a separate no-prompt pass, keys trained only by the pull loss, head on the prompted [CLS], training logits = current task (all as in the CODA-Prompt repo) | **this is "L2P++"** (prefix-tuning, pool 30, length 20, layer 0) as re-implemented in the CODA-Prompt paper; the original prompt-tuning L2P is not implemented. Pull loss: 0.1 × batch mean (L2P paper) vs the repo's 1 × batch sum; it only reaches the keys, and Adam makes key updates nearly scale-invariant |
| `dualprompt` | Wang et al., ECCV 2022 | G-prompt layers 0-1 (length 6), E-prompt per task layers 2-4 (length 20), true task's E-prompt and pull loss in training, top-1 key matching at test (CODA-Prompt repo) | the repo allocates all 10 E-prompts up front and matches against all 10 keys at test, including not-yet-trained ones; here one E-prompt/key is added per task, so test-time matching only sees trained keys |
| `coda-prompt` | Smith et al., CVPR 2023 | checked against the repo's `CodaPrompt`: 100 components split evenly over tasks, length 8, layers 0-4, weights = cosine(q ⊙ A, K) without softmax, new slice initialised orthogonal to the old ones (Gram-Schmidt there, QR here), old slices frozen, keys/attention trained by the CE loss, ortho loss 0 (repo default) | — |

**Routing** (one expert per task; the router never sees labels at test time)

| name | follows | differs |
|---|---|---|
| `router=hide` | HiDe-Prompt's task-identity inference (Wang et al., NeurIPS 2023): a linear classifier over all seen classes on the plain PTM's features, kept unbiased with Gaussian samples of old classes; predicted class -> its task (authors' `hide_promtp_wtp_and_tap_engine.py`) | the authors first train the head with CE on real features of the current task and then correct it on samples from 10 k-means centroids per class with diagonal variance (SGD 0.005, 30 epochs); here the head is refit each task on samples only, from one full-covariance Gaussian per class (Adam 1e-3, 20 epochs). **HiDe-Prompt's within-task contrastive loss and task-adaptive prediction are not implemented**, so `hide-prompt` = routing only. Prompt experts: 10 key + 10 value tokens in layers 0-4 (authors' CIFAR script: 5 + 5) |
| `router=sprompts` | S-Prompts (Wang et al., NeurIPS 2022): k-means centroids per task on frozen features, nearest centroid, then the selected task's classifier only (authors' `slinet.py` keeps only the selected domain's logits) | **routing only**; no language-image prompts. K = 5 follows the paper; the authors' selection code (`methods/sprompt.py`) could not be fetched, so K and the distance (here Euclidean on normalised features) are not code-verified |
| `router=expertgate` | Expert Gate (Aljundi et al., CVPR 2017): one autoencoder per task, lowest reconstruction error, then that task's expert | autoencoder on normalised PTM features (paper: on AlexNet conv features, sigmoid output); experts are adapters, not fine-tuned networks |
| `router=keyquery` | L2P / DualPrompt key matching | — |
| `router=<head>` | ours: a classifier head's predicted class -> its task | — |
| prediction after routing | `within_task=True` (default): among the routed task's classes (as in Expert Gate) | `within_task=False`: all seen classes compete |
| `classifier=<statistical head>` with routing | ours: one copy of the head per expert, fitted on that task's training data through that expert (e.g. LDA with the task's own covariance); the routed task's head scores the routed expert's features among that task's classes | — (tested against an independent re-computation of every prediction) |

**Merging** (`cl_toolkit.merging`; in CL via `MergeAfterTask` / `ModelAveraging`, which need a
trainable backbone: `--finetune-backbone`; with a frozen backbone they raise an error instead
of silently training only the head)

| name | paper | follows | differs / note |
|---|---|---|---|
| `weight-average` | weight averaging | mean of task vectors | — |
| `task-arithmetic` | Ilharco et al., ICLR 2023 | θ0 + λ Σ τ_t | λ = 0.3 fixed (paper: tuned on validation). With sequential fine-tuning each τ_t already contains the earlier tasks, so Σ over-counts; use `weight-average` or a small λ there |
| `ties` | Yadav et al., NeurIPS 2023 | trim to top 20% over each **whole** task vector, sign by summed mass, zero signs -> majority sign, disjoint mean, λ = 1 (authors' notebook) | `scope="tensor"` trims per tensor |
| `dare` | Yu et al., ICML 2024 | Bernoulli drop p, rescale 1/(1-p), then Task Arithmetic (default) or `base=ties` | p = 0.9 fixed (paper: tuned) |
| `magmax` | Marczak et al., ECCV 2024 | per-coordinate max magnitude over task vectors w.r.t. θ0, sequential fine-tuning from the previous fine-tuned model, λ = 0.5 (paper) | — |
| `breadcrumbs` | Davari & Belilovsky, ECCV 2024 | per layer, drop the smallest β = 90% and largest 1-γ = 0.8% of \|τ\| (paper's ViT-B/16 values), sum | λ = 0.3 fixed (paper: tuned) |
| `consensus` | Wang et al., ICML 2024 | TALL masks m_t = \|τ_t\| ≥ λ_t \|τ_MTL − τ_t\|, keep weights selected by ≥ k = 2 tasks, on Task Arithmetic | one λ_t = 0.4 for all tasks (paper: tuned per task on validation, range 0.2-0.6) |
| `tsv` | Gargiulo et al., CVPR 2025 (TSV-M) | per matrix: top r/T singular components per task, Procrustes-orthogonalised U and V, α = 1 (paper); non-matrix parameters averaged | — |
| `iso-c` | Marczak et al., ICML 2025 | per matrix: SVD of Σ τ_t, all singular values -> their mean; non-matrix averaged | α = 1 fixed (paper: tuned) |
| `iso-cts` | Marczak et al., ICML 2025 | common subspace k/r = 0.8 (paper default) + each task's top (r−k)/T directions orthogonal to it, orthogonalised, isotropic spectrum | α = 1 fixed (paper: tuned) |
| `coma` | Marouf et al. 2024 ("Weighted Ensemble Models Are Strong Continual Learners") | θ*_t = λ θ_t + (1−λ) θ*_{t−1}, λ = 0.4 (supervised pre-training), task t trains from θ*_{t−1} | backbone only (head not averaged); the paper builds on SLCA |
| `cofima` | same paper | Fisher-weighted version: F_t from θ_t on task t, F*_{t−1} from θ*_{t−1} on task t−1 (computed at the end of task t−1) | where both Fishers are 0, CoMA weights |
| `classifier="clip-zeroshot"` | CLIP (Radford et al. 2021), as used by task_vectors / MagMax | frozen head = 100 · cos(image, mean normalised text embedding over CLIP's 18 CIFAR templates) | — |

Unit tests (`tests/test_merging.py`): hand examples (TA, average, MagMax, TIES global trimming,
sign election, zero-sign rule), DARE is unbiased and drops p, Breadcrumbs removes exactly both
tails, Consensus keeps only shared weights, TSV is exact for task matrices in orthogonal
subspaces, Iso-C's spectrum is flat at the mean of Σ τ_t, Iso-CTS with k = r equals Iso-C and
is isotropic, CoMA / CoFiMA recursions by hand. Every merging strategy also runs through all
integrity checks in Section 1.

**AdaptFormer** (`first_session="adaptformer"`, `experts="adaptformer"`; Chen et al., NeurIPS 2022):
parallel bottleneck adapter (64, ReLU, scale 0.1, zero-init up-projection) whose input is the
block's input *before* `norm2`, as in the AdaptFormer / APER / RanPAC code.

**Replay memories** store the *augmented* training tensors and replay them as stored; mammoth
stores un-augmented inputs and re-augments them on replay.

**Ours, not from a paper**: `lora` / `lora-er` (one LoRA per task, all active and summed,
old ones frozen), `routed` with LoRA / AdaptFormer / prompt experts, `first-session` with LoRA.

**Preprocessing**: named image datasets use `preprocess="model"` (the backbone's own
normalisation, CIFAR resized without cropping) by default. Published PTM-CL numbers use
`"pilot"` (RanPAC / APER / SimpleCIL code) or `"coda"` (CODA-Prompt code): no input
normalisation. This alone moves NCM on CIFAR-100 by 3 points (Section 2). The preset and
the exact pretrained weights (`model_name`, e.g. `vit_base_patch16_224.augreg_in21k`) are
written to every run's `config.json`.

## 4. Problems this audit found and fixed

Numbers produced before these fixes (e.g. in earlier logs under `runs/`) are not comparable.

| problem | effect | fix |
|---|---|---|
| default CIFAR preprocessing center-cropped 32x32 images upsampled to 248 | NCM on CIFAR-100: 78.0 instead of 80.1 | low-res datasets are resized, never cropped; presets `pilot` / `coda` for paper-matched runs |
| default `vit_base_patch16_224` resolved to `augreg2_in21k_ft_in1k`, weights no paper uses; the exact tag was not logged | results not comparable to any paper | default is now `vit_base_patch16_224.augreg_in21k`; the exact tag is logged |
| CLIP models used ImageNet normalisation instead of CLIP's | wrong inputs for CLIP | CLIP's own preprocessing config |
| `clt.build(model=<CLModel>, freeze_backbone=True)` silently left the model trainable while the config said frozen | trained backbone reported as frozen | the setting is applied; the log and config record the actual trainable backbone parameters |
| RanPAC chose the ridge λ by validation *accuracy* over 1e-8..1e13 (docstring claimed "as the authors") | different λ | validation MSE over 1e-8..1e8, as in the paper/code |
| EWC / MAS importances from squared / absolute *batch-mean* gradients | importances too small and wrong | per-sample gradients |
| EWC kept one anchor (the latest weights) for all past tasks | not EWC | one penalty per task, each at its own anchor (Online EWC keeps one, as in its paper) |
| `magmax` / `ties` with the default frozen backbone merged nothing | silently plain head training | error with an explanation |
| FeCAM default off-diagonal shrinkage 1 (authors' ViT script: 0) | different method settings | authors' ViT defaults |
| L2P / DualPrompt used 2x the prefix tokens of the papers' convention; L2P defaults were not those of the reported L2P++ | different prompt sizes | the authors' length convention and the CODA-Prompt repo defaults |
| SLCA missing weight decay 5e-4 | minor | added |
| routed LoRA: peft's `enable_adapter_layers()` re-enabled gradients of old experts | old experts kept training (trainable params doubled) | trainable flags set explicitly; regression test |
| routed prediction let every seen class compete after routing | e.g. LLM routed LoRA 61% with 97% correct routing | `within_task=True` default (Expert Gate rule); `False` available |
| docstrings claimed HiDe-Prompt / S-Prompts restrict prediction to the routed task | wrong claim | corrected (HiDe recalibrates a shared head instead) |
| frozen strategies printed as `FirstSessionAdaptation(bottleneck=64, lora_rank=8)` | misleading logs | printed as `Frozen()` |
| only task-averaged accuracy was reported; PTM-CL papers report accuracy over all seen test samples | not comparable on imbalanced benchmarks (ImageNet-R, CUB) | `accuracy_all_seen` and `avg_incremental_accuracy_all_seen` added |
| optimizer, weight decay, scheduler, batch size were not in `config.json` | runs not fully specified | logged |
| TIES trimmed each tensor separately; zero signs were not resolved | not TIES as published | trims over the whole task vector, zero signs -> majority (authors' code); `scope="tensor"` kept as an option |
| MagMax default λ = 1 | paper uses 0.5 | 0.5 |
| DARE defaulted to TIES without trimming as its base | not the paper's main combination | Task Arithmetic by default, `base=ties` for DARE-TIES |
| SLCA normalised logits by their global L2 norm, sampled once, batch 128, aligned after task 0 too, no mean decay | ≈ √T softer temperature than the authors'; different alignment | per-task-norm logit normalisation, mean decay, fresh samples every epoch, batch 256, from task 1 on (authors' code) |
| AdaptFormer adapter read the `norm2` output | not AdaptFormer's parallel adapter | reads the pre-`norm2` block input (authors' code) |
| KLDA docstring claimed the paper reports no ensemble gain on images | unsupported claim; KLDA-E missing | claim removed; `ensembles=5` = KLDA-E |
| decoder-LLM last-token pooling assumed right padding | wrong token pooled with a left-padding tokenizer | last non-pad token found for any padding side |
| **A-GEM in Task-IL never projected**: parameters unused by the memory loss (the current head) had no reference gradient, the vectors' lengths differed and the code returned silently (crash with a frozen backbone) | Task-IL A-GEM was plain fine-tuning | zero reference gradient for unused parameters; regression test (found by the interoperability grid) |
| `clt.build(model=<prebuilt frozen model>, freeze_backbone=False)` kept it frozen (e.g. `HFModel`) | merging / full fine-tuning impossible for Hugging Face models | the setting is applied in both directions (found by the grid) |
| TSV / Iso-C / Iso-CTS: LAPACK SVD can fail to converge on ill-conditioned task vectors | crash | float64 retry, then tiny jitter (found by the grid) |
| prompt methods on a non-ViT model raised `AttributeError` | unclear error | `TypeError` naming the requirement |
| `classifier=<head>` with `experts=` / `router=`: the head was fitted on and scored the plain backbone's features | experts trained and silently ignored (LDA + LoRA experts = plain LDA, 83.75 both); no routing accuracy | statistical heads now become per-expert heads on the experts' own features; SLCA / WA (shared-head corrections) are refused |
| **the same seed gave different results in different processes** for merging strategies: `MergeAfterTask` ordered parameters with a Python `set` (hash-seed dependent), and DARE drew its masks in that order | DARE runs not reproducible (e.g. 41.7% vs 8.3% on the same cell) | parameter order kept; DARE uses one random stream per (seed, task, parameter name); test runs 5 hash seeds |
| data loaders pinned memory whenever a GPU existed, also for CPU runs | wasted pinned memory; CPU runs broke in processes forked after CUDA init | pinned only when the run's device is CUDA (unless `pin_memory` is given) |
| `log.txt` was appended to when a run reused an output folder; the first two log lines were printed before the log was attached | old and new runs mixed in one log | the previous run's files move to `previous_<time>/`; the log is complete and closed even if the run crashes |

## 5. Interoperability

`experiments/interop_grid.py` runs **every method x every scenario x five kinds of model** through
`clt.build(...).fit()` on tiny CPU models (1110 cells, ~2 CPU-hours, ~5 min on 140 cores):

| model | what it exercises |
|---|---|
| `clmodel` | a minimal `CLModel` subclass (`_encode` + `embed_dim`), tensor inputs |
| `wrapped` | `clt.models.wrap(nn.Sequential(...))`, LoRA on `all-linear` |
| `vit` | timm ViT (prefix prompts, AdaptFormer, LoRA on `qkv`), image inputs |
| `bert` | `HFModel` on a Hugging Face BERT, tokenised-text dict inputs, LoRA on `query`/`value` |
| `decoder` | `HFModel` on a Hugging Face Qwen2 decoder with the real Qwen2.5 tokenizer, last-token pooling |

A cell passes if it trains and evaluates, all accuracies are finite, in Class-/Domain-IL the
per-task test loaders give the same result as one merged loader (no task identity used), and
(with `--epochs 5 --learn-check`) the task just trained is above chance. A known incompatibility
must be refused with a clear `ValueError` / `TypeError` before or at the start of training;
any other error is a failure.

```bash
python experiments/interop_grid.py                              # plumbing, 1 epoch -> runs/interop/grid.md
python experiments/interop_grid.py --epochs 5 --learn-check --out runs/interop/learn
```

Grid workers run with the GPUs hidden: PyTorch's `Adam.step()` initialises CUDA whenever a GPU is
visible, even for CPU tensors, so 100+ CPU workers would each open a CUDA context.

Result: **819 ok, 291 refused by design, 0 failures** in both modes. C / T / D = Class- / Task- /
Domain-IL; – = refused by design; ok* = runs correctly, newest task at chance (explained below).

| method | clmodel C | clmodel T | clmodel D | wrapped C | wrapped T | wrapped D | vit C | vit T | vit D | bert C | bert T | bert D | decoder C | decoder T | decoder D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `naive` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `naive-finetune` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `er` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `er-ace` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `der` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `derpp` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `agem` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `icarl` | ok | – | – | ok | – | – | ok | – | – | ok | – | – | ok | – | – |
| `gdumb` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok* | ok | ok | ok | ok |
| `ewc` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `ewc-online` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `mas` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lwf` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora-merge` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora-er` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora+derpp` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora+ewc` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `simplecil` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `ranpac` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `first-session-lora` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `first-session-adaptformer` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `head-ncm` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-fecam` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-lda` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-klda` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-ranpac` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-klda-e` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `head-slca` | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok |
| `head-wa` | ok | – | – | ok | – | – | ok | – | – | ok | – | – | ok | – | – |
| `head-cosine` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `er+lda` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `lora+ncm` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `clip-zeroshot` | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| `l2p` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `dualprompt` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `coda-prompt` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-lora-hide` | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – |
| `routed-lora-lda` | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – |
| `routed-lora-fecam` | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – |
| `routed-lora-sprompts` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `routed-lora-keyquery` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `routed-lora-expertgate` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `routed-adaptformer-hide` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-adaptformer-lda` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-adaptformer-fecam` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-adaptformer-sprompts` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-adaptformer-keyquery` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-adaptformer-expertgate` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-prompt-hide` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-prompt-lda` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-prompt-fecam` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-prompt-sprompts` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-prompt-keyquery` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-prompt-expertgate` | – | – | – | – | – | – | ok | ok | ok | – | – | – | – | – | – |
| `routed-lora-hide+lda` | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – |
| `routed-lora-sprompts+ncm` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `routed-prompt-hide+fecam` | – | – | – | – | – | – | ok | ok | – | – | – | – | – | – | – |
| `routed-lora-keyquery+ranpac` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `routed-lora-hide+slca` | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| `routed-lora-er` | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – | ok | ok | – |
| `magmax` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `ties` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok* | ok | ok | ok | ok | ok |
| `task-arithmetic` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `dare` | ok | ok | ok | ok* | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `breadcrumbs` | ok* | ok | ok | ok* | ok | ok | ok | ok | ok | ok* | ok | ok | ok* | ok | ok |
| `consensus` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok* | ok | ok | ok* | ok | ok |
| `tsv` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `iso-c` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `iso-cts` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `weight-average` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `coma` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `cofima` | ok | ok | ok | ok | ok | ok | ok | ok* | ok | ok | ok | ok | ok | ok | ok |
| `magmax+er` | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |

Refusals by design:

- SLCA / WA correct one shared head across tasks; routed prediction is already within one task
- SLCA aligns one shared head; Task-IL has per-task heads
- WA aligns new vs old *classes*: Class-IL only
- class-level routers map a predicted class to its task: needs disjoint classes per task
- iCaRL is Class-IL only (as in the paper)
- needs a CLIP model with a text tower
- needs a timm ViT (prefix prompts / AdaptFormer patch timm attention / MLP blocks)

ok* cells (10, all checked by hand): Breadcrumbs, Consensus, TIES and DARE in Class-IL on some models
keep too little of the newest task vector at their default sparsity on these tiny models;
denser settings learn it (TIES density 0.5: 66.7%, Breadcrumbs λ = 1: 50%, DARE p = 0.5: 41.7%), and the earlier
tasks are learned. GDumb / BERT / Task-IL is borderline (41.7% vs 50% chance on 12 test samples).
CoFiMA / ViT / Task-IL is seed noise (other seeds: 100 / 100 / 100).

Bugs this grid found and that are now fixed (Section 4): A-GEM never projected in Task-IL,
`freeze_backbone=False` was ignored for pre-built frozen models, SVD non-convergence in
TSV / Iso-C / Iso-CTS, unclear `AttributeError`s for prompts on non-ViT models, and a classifier
head combined with routed experts silently ignoring the experts.

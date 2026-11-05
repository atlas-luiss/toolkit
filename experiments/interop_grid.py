"""Interoperability grid: every method x every scenario x every kind of model, on tiny CPU models.

Each cell is built with ``clt.build`` and trained for one epoch per task. A cell is

    ok       trained and evaluated; finite accuracies; in Class-/Domain-IL the per-task test
             loaders give the same result as one merged loader (no task identity used)
    refused  raised a clear error before training (a known, documented incompatibility,
             listed in ``REFUSALS`` with the reason)
    FAIL     anything else: a crash during training/evaluation, an unexpected refusal, a NaN,
             or a task-identity leak

Models: a CLModel subclass (``clmodel``), ``wrap(nn.Module)`` (``wrapped``), a timm ViT
(``vit``), a Hugging Face encoder (``bert``) and a Hugging Face decoder LLM (``decoder``,
Qwen2 architecture with the real Qwen2.5 tokenizer), all tiny and randomly initialised.

    python experiments/interop_grid.py                 # full grid -> runs/interop/grid.{md,json}
    python experiments/interop_grid.py --models vit --methods l2p,dualprompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cl_toolkit as clt  # noqa: E402
from cl_toolkit.benchmarks import split_by_class, split_by_domain  # noqa: E402
from cl_toolkit.benchmarks.text import TextDataset  # noqa: E402

D = 32
SCENARIOS = ("class", "task", "domain")
HF_DIR = os.environ.get("CLT_INTEROP_HF", "runs/interop/hf_models")


# ============================================================================ models
class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.nn.functional.gelu(self.qkv(x))


class TinyCL(clt.models.CLModel):
    """The minimal CLModel subclass a user writes: ``_encode`` + ``embed_dim``."""

    def __init__(self):
        super().__init__("cpu")
        self._backbone = nn.Sequential(nn.Linear(D, 64), _Block(64), _Block(64))

    def _encode(self, x):
        return self._backbone(x)

    @property
    def embed_dim(self):
        return 64


def _hf_dirs():
    """Tiny random BERT and Qwen2 models saved locally (no download except the Qwen tokenizer)."""
    import transformers as tf
    bert, dec = os.path.join(HF_DIR, "bert"), os.path.join(HF_DIR, "qwen2")
    if not os.path.exists(os.path.join(bert, "config.json")):
        os.makedirs(bert, exist_ok=True)
        words = [f"w{i}" for i in range(60)]
        with open(os.path.join(bert, "vocab.txt"), "w") as f:
            f.write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
        tf.BertTokenizer(os.path.join(bert, "vocab.txt")).save_pretrained(bert)
        torch.manual_seed(0)
        tf.BertModel(tf.BertConfig(vocab_size=65, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                   intermediate_size=64, max_position_embeddings=64)).save_pretrained(bert)
    if not os.path.exists(os.path.join(dec, "config.json")):
        os.makedirs(dec, exist_ok=True)
        tok = tf.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tok.save_pretrained(dec)
        torch.manual_seed(0)
        tf.Qwen2Model(tf.Qwen2Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
                                     max_position_embeddings=64)).save_pretrained(dec)
    return bert, dec


def make_model(kind):
    torch.manual_seed(0)
    if kind == "clmodel":
        return TinyCL()
    if kind == "wrapped":
        return clt.models.wrap(nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 64)), embed_dim=64,
                               device="cpu")
    if kind == "vit":
        return clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, device="cpu", img_size=32)
    bert, dec = _hf_dirs()
    # random-init BERT's [CLS] is ~constant across inputs (std 3e-3), so the grid mean-pools it;
    # the decoder uses the default last-token pooling
    if kind == "bert":
        return clt.models.HFModel(bert, max_length=12, device="cpu", pooling="mean")
    return clt.models.HFModel(dec, max_length=12, device="cpu")


MODELS = ("clmodel", "wrapped", "vit", "bert", "decoder")


# ============================================================================ data
def _vec(n_classes, n, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n_classes, D, generator=torch.Generator().manual_seed(0)) * 1.0
    y = torch.arange(n_classes).repeat_interleave(n)
    return TensorDataset(mu[y] + torch.randn(len(y), D, generator=g) + shift, y)


def _img(n_classes, n, seed, shift=0.0):
    """Class = a fixed random 32x32 template (+ noise); domains add a brightness shift."""
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(n_classes, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    y = torch.arange(n_classes).repeat_interleave(n)
    x = templates[y] + 0.5 * torch.randn(len(y), 3, 32, 32, generator=g) + shift
    return TensorDataset(x, y)


# whole-word tokens for the decoder's real (Qwen) tokenizer, which splits "w12" into digits
_ENGLISH = ("apple river stone cloud green table music light paper horse winter garden "
            "silver ocean forest bread chair window glass mountain tiger yellow summer candle "
            "planet copper violin desert rabbit marble pepper rocket island lemon thunder "
            "orange castle feather spider basket mirror cotton dragon engine falcon helmet "
            "jacket kitten ladder magnet needle oyster pillow quartz saddle tunnel velvet "
            "walnut zebra anchor bottle cactus donkey").split()


def _txt(n_classes, n, seed, tf, domain=0, english=False):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    word = (lambda i: _ENGLISH[i]) if english else (lambda i: f"w{i}")
    for c in range(n_classes):
        for _ in range(n):
            words = [word((c * 10 + int(rng.integers(10)) + 7 * domain) % 60) for _ in range(8)]
            texts.append(" ".join(words[: int(rng.integers(3, 9))]))      # variable length: padding matters
            labels.append(c)
    return TextDataset(texts, labels, tf)


def make_bench(kind, scenario, model):
    n_classes, T = (6, 3) if scenario != "domain" else (4, 2)
    if kind in ("clmodel", "wrapped"):
        mk = lambda n, s, d=0: _vec(n_classes, n, s, shift=1.5 * d)
    elif kind == "vit":
        mk = lambda n, s, d=0: _img(n_classes, n, s, shift=0.7 * d)
    else:
        tr_tf, te_tf = model.transforms()
        mk = lambda n, s, d=0: _txt(n_classes, n, s, tr_tf, domain=d, english=kind == "decoder")
    if scenario == "domain":
        return split_by_domain([(mk(16, 10 + d, d), mk(6, 20 + d, d)) for d in range(T)])
    return split_by_class(mk(16, 1), mk(6, 2), T, shuffle_classes=False)


# ============================================================================ methods
NOFREEZE = {"freeze_backbone": False}
MEM = {"memory_size": 24}


def _m(**kw):
    return kw


METHODS = {
    # baselines / replay / regularisation
    "naive": _m(strategy="naive"),
    "naive-finetune": _m(strategy="naive", **NOFREEZE),
    "er": _m(strategy="er", strategy_kwargs=MEM, **NOFREEZE),
    "er-ace": _m(strategy="er-ace", strategy_kwargs=MEM, **NOFREEZE),
    "der": _m(strategy="der", strategy_kwargs=MEM, **NOFREEZE),
    "derpp": _m(strategy="derpp", strategy_kwargs=MEM, **NOFREEZE),
    "agem": _m(strategy="agem", strategy_kwargs=MEM, **NOFREEZE),
    "icarl": _m(strategy="icarl", strategy_kwargs=MEM, **NOFREEZE),
    "gdumb": _m(strategy="gdumb", strategy_kwargs={"memory_size": 24, "epochs": 20, "batch_size": 8}),
    "ewc": _m(strategy="ewc", strategy_kwargs={"fisher_samples": 8}, **NOFREEZE),
    "ewc-online": _m(strategy="ewc-online", strategy_kwargs={"fisher_samples": 8}, **NOFREEZE),
    "mas": _m(strategy="mas", strategy_kwargs={"importance_samples": 8}, **NOFREEZE),
    "lwf": _m(strategy="lwf", **NOFREEZE),
    # PEFT
    "lora": _m(strategy="lora", strategy_kwargs={"rank": 2}),
    "lora-merge": _m(strategy="lora", strategy_kwargs={"rank": 2, "merge_after_task": True}),
    "lora-er": _m(strategy="lora-er", strategy_kwargs={"rank": 2, **MEM}),
    "lora+derpp": _m(strategy=lambda: clt.strategies.LoRAAdapter(clt.strategies.DERPP(24), rank=2)),
    "lora+ewc": _m(strategy=lambda: clt.strategies.LoRAAdapter(clt.strategies.EWC(fisher_samples=8), rank=2)),
    # pre-trained-model methods and heads
    "simplecil": _m(strategy="simplecil", epochs=0),
    "ranpac": _m(strategy="ranpac", strategy_kwargs={"proj_dim": 128}, epochs=0),
    "first-session-lora": _m(strategy="first-session", strategy_kwargs={"first_session": "lora", "lora_rank": 2},
                             classifier="ncm"),
    "first-session-adaptformer": _m(strategy="first-session", strategy_kwargs={"first_session": "adaptformer",
                                                                               "bottleneck": 8}, classifier="lda"),
    **{f"head-{h}": _m(strategy="frozen", classifier=h, epochs=0,
                       **({"classifier_kwargs": {"proj_dim": 128}} if h == "ranpac" else
                          {"classifier_kwargs": {"rff_dim": 256}} if h == "klda" else {}))
       for h in ("ncm", "fecam", "lda", "klda", "ranpac")},
    "head-klda-e": _m(strategy="frozen", classifier="klda", epochs=0,
                      classifier_kwargs={"rff_dim": 128, "ensembles": 3}),
    "head-slca": _m(strategy="naive", classifier="slca", classifier_kwargs={"samples_per_class": 16, "epochs": 1}),
    "head-wa": _m(strategy="er", strategy_kwargs=MEM, classifier="wa"),
    "head-cosine": _m(strategy="naive", classifier="cosine"),
    "er+lda": _m(strategy="er", strategy_kwargs=MEM, classifier="lda", **NOFREEZE),
    "lora+ncm": _m(strategy="lora", strategy_kwargs={"rank": 2}, classifier="ncm"),
    "clip-zeroshot": _m(strategy="naive", classifier="clip-zeroshot"),
    # prompts (timm ViT only)
    "l2p": _m(strategy="l2p", strategy_kwargs={"pool_size": 6, "top_k": 2}),
    "dualprompt": _m(strategy="dualprompt"),
    "coda-prompt": _m(strategy="coda-prompt", strategy_kwargs={"pool_size": 6}),
    # routing: every expert type x every router
    **{f"routed-{e}-{r}": _m(experts=e, router=r,
                             strategy_kwargs={"experts_kwargs": {"rank": 2} if e == "lora" else
                                              {"bottleneck": 8} if e == "adaptformer" else {}})
       for e in ("lora", "adaptformer", "prompt") for r in ("hide", "lda", "fecam", "sprompts", "keyquery", "expertgate")},
    **{f"routed-{e}-{r}+{h}": _m(experts=e, router=r, classifier=h,
                                 strategy_kwargs={"experts_kwargs": {"rank": 2} if e == "lora" else {}})
       for e, r, h in (("lora", "hide", "lda"), ("lora", "sprompts", "ncm"), ("prompt", "hide", "fecam"),
                       ("lora", "keyquery", "ranpac"))},
    "routed-lora-hide+slca": _m(experts="lora", router="hide", classifier="slca",
                                strategy_kwargs={"experts_kwargs": {"rank": 2}}),
    "routed-lora-er": _m(strategy=lambda: clt.strategies.Routed("lora", "lda", strategy=clt.strategies.ExperienceReplay(24),
                                                                 experts_kwargs={"rank": 2})),
    # merging (sequential full fine-tuning) and model averaging
    **{m: _m(strategy=m, **NOFREEZE) for m in ("magmax", "ties", "task-arithmetic", "dare", "breadcrumbs",
                                               "consensus", "tsv", "iso-c", "iso-cts", "weight-average", "coma")},
    "cofima": _m(strategy="cofima", strategy_kwargs={"fisher_samples": 8}, **NOFREEZE),
    "magmax+er": _m(strategy=lambda: clt.strategies.MergeAfterTask(clt.strategies.ExperienceReplay(24), method="magmax"),
                    **NOFREEZE),
}

# (method predicate, model predicate, scenario predicate) -> reason. Clear errors expected here.
_VIT_ONLY = "needs a timm ViT (prefix prompts / AdaptFormer patch timm attention / MLP blocks)"
REFUSALS = [
    (lambda m: m in ("l2p", "dualprompt", "coda-prompt") or "-prompt-" in m or m.startswith("routed-prompt"),
     lambda k: k != "vit", lambda s: True, _VIT_ONLY),
    (lambda m: "adaptformer" in m, lambda k: k != "vit", lambda s: True, _VIT_ONLY),
    (lambda m: m == "clip-zeroshot", lambda k: True, lambda s: True, "needs a CLIP model with a text tower"),
    (lambda m: m == "routed-lora-hide+slca", lambda k: True, lambda s: True,
     "SLCA / WA correct one shared head across tasks; routed prediction is already within one task"),
    (lambda m: m == "icarl", lambda k: True, lambda s: s != "class", "iCaRL is Class-IL only (as in the paper)"),
    (lambda m: m == "head-wa", lambda k: True, lambda s: s != "class", "WA aligns new vs old *classes*: Class-IL only"),
    (lambda m: m == "head-slca", lambda k: True, lambda s: s == "task",
     "SLCA aligns one shared head; Task-IL has per-task heads"),
    (lambda m: m.startswith("routed-") and any(r in m for r in ("hide", "lda", "fecam")) and m != "routed-lora-er" or
     m == "routed-lora-er", lambda k: True, lambda s: s == "domain",
     "class-level routers map a predicted class to its task: needs disjoint classes per task"),
]


# Cells that run correctly but whose newest task stays at chance with these tiny models and the
# methods' default hyper-parameters. Each was checked by hand (see VERIFICATION.md, section 5).
_SPARSE = ("sparsifying / consensus merge at default settings keeps too little of the newest task "
           "vector (denser settings learn it: TIES density 0.5 -> 66.7%, Breadcrumbs lam 1 -> 50%, "
           "DARE p 0.5 -> 41.7%)")
LEARN_EXEMPT = {
    **{(m, k, "class"): _SPARSE for m in ("breadcrumbs", "consensus", "ties")
       for k in ("clmodel", "wrapped", "bert", "decoder")},
    ("dare", "wrapped", "class"): _SPARSE,
    ("gdumb", "bert", "task"): "borderline: 41.7% vs chance 50% on 12 test samples (24-sample memory)",
    ("cofima", "vit", "task"): "seed noise: other seeds learn every task (100/100/100)",
}


def expected_refusal(method, kind, scenario):
    for pm, pk, ps, why in REFUSALS:
        if pm(method) and pk(kind) and ps(scenario):
            return why
    return None


# ============================================================================ one cell
def _merged_check(tr):
    """Per-task test loaders vs one merged loader must agree (no task id from the loader)."""
    T = tr.scenario.num_tasks
    sets = [t.test_loader.dataset for t in tr.scenario.tasks]
    sizes = np.array([len(d) for d in sets])
    expected = float((tr.acc[T - 1, :T] / 100 * sizes).sum())
    model, strat = tr.model, tr.strategy
    model.eval()
    tr._set_eval_mask(T - 1)
    strat.before_eval(model, None)
    loader = DataLoader(ConcatDataset(sets), batch_size=7, collate_fn=tr.scenario.tasks[0].test_loader.collate_fn)
    with torch.no_grad(), tr.autocast():
        r = strat.evaluate_task(loader, model, None) or tr.evaluate_task(loader, None)
    got = r["accuracy"] / 100 * sizes.sum()
    return abs(got - expected) <= 1.0 + 1e-6, got, expected


def run_cell(method, kind, scenario, epochs=1, learn_check=False):
    torch.set_num_threads(1)
    kw = dict(METHODS[method])
    if callable(kw.get("strategy")):
        kw["strategy"] = kw["strategy"]()
    t0 = time.time()
    why = expected_refusal(method, kind, scenario)
    out = {"method": method, "model": kind, "scenario": scenario}
    stage = "build"
    try:
        model = make_model(kind)
        bench = make_bench(kind, scenario, model)
        torch.manual_seed(0)
        lr = 5e-4 if kw.get("freeze_backbone") is False else 5e-3     # full fine-tuning: smaller lr
        tr = clt.build(bench, model, scenario, num_tasks=bench.num_tasks, epochs=kw.pop("epochs", epochs),
                       batch_size=8, num_workers=0, device="cpu", verbose=False, lr=lr, **kw)
        stage = "fit"
        r = tr.fit()
        stage = "check"
        acc = tr.acc[np.tril_indices(tr.scenario.num_tasks)]
        if not np.all(np.isfinite(acc)) or acc.min() < 0 or acc.max() > 100:
            raise FloatingPointError(f"accuracy matrix not finite / out of range: {tr.acc.tolist()}")
        if scenario != "task":
            ok, got, exp = _merged_check(tr)
            if not ok:
                raise AssertionError(f"task-identity leak: merged loader {got:.1f} correct vs per-task {exp:.1f}")
        T = tr.scenario.num_tasks
        n_cls = (len(tr.scenario.seen_classes(T - 1)) if scenario == "class" else
                 len(tr.scenario.get_task(T - 1).classes))
        out["just_trained_acc"] = round(float(tr.acc[T - 1, T - 1]), 2)
        out["at_chance"] = bool(tr.acc[T - 1, T - 1] <= 100.0 / n_cls + 1e-6)
        exempt = LEARN_EXEMPT.get((method, kind, scenario))
        out["learn_exempt"] = exempt if (out["at_chance"] and exempt) else None
        if learn_check and out["at_chance"] and not why and not exempt:
            raise AssertionError(f"did not learn: just-trained task at {tr.acc[T - 1, T - 1]:.1f}% "
                                 f"(chance {100.0 / n_cls:.1f}%)")
        out.update(status="FAIL" if why else "ok", final_acc=round(float(r["avg_accuracy"]), 2),
                   note=f"expected refusal ({why}) but it ran" if why else "")
    except Exception as e:                                       # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        clear = stage in ("build", "fit") and isinstance(e, (ValueError, TypeError, KeyError, NotImplementedError))
        if why and clear:
            out.update(status="refused", note=f"{why} | {msg}")
        else:
            out.update(status="FAIL", note=f"[{stage}] {msg}", trace=traceback.format_exc()[-3000:])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cpu_only_worker():
    """Grid workers run on CPU only. Hide the GPUs: PyTorch's Adam.step() initialises CUDA whenever a
    GPU is visible (a CUDA-graph health check), so 100+ CPU workers would each open a CUDA context."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="runs/interop")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--learn-check", action="store_true",
                    help="FAIL ok cells whose just-trained task is at or below chance")
    a = ap.parse_args(argv)
    models, methods, scens = a.models.split(","), a.methods.split(","), a.scenarios.split(",")
    if any(k in ("bert", "decoder") for k in models):
        _hf_dirs()                                               # create once, before forking
    os.makedirs(a.out, exist_ok=True)
    cells = [(m, k, s) for m in methods for k in models for s in scens]
    print(f"{len(cells)} cells, {a.workers} workers -> {a.out}", flush=True)
    results = []
    with ProcessPoolExecutor(a.workers, initializer=cpu_only_worker) as ex:
        futs = {ex.submit(run_cell, *c, a.epochs, a.learn_check): c for c in cells}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            results.append(r)
            if r["status"] == "FAIL":
                print(f"[{i + 1}/{len(cells)}] FAIL {r['method']} / {r['model']} / {r['scenario']}: {r['note']}",
                      flush=True)
    order = {m: i for i, m in enumerate(methods)}
    results.sort(key=lambda r: (order[r["method"]], models.index(r["model"]), scens.index(r["scenario"])))
    json.dump(results, open(os.path.join(a.out, "grid.json"), "w"), indent=1)
    sym = {"ok": "ok", "refused": "–", "FAIL": "**FAIL**"}
    for r in results:
        if r["status"] == "ok" and r.get("learn_exempt"):
            r["status_sym"] = "ok*"
    lines = ["| method | " + " | ".join(f"{k} {s[0].upper()}" for k in models for s in scens) + " |",
             "|---|" + "---|" * (len(models) * len(scens))]
    by = {(r["method"], r["model"], r["scenario"]): r for r in results}
    for m in methods:
        lines.append(f"| `{m}` | " + " | ".join(by[(m, k, s)].get("status_sym") or sym[by[(m, k, s)]["status"]]
                                   for k in models for s in scens) + " |")
    n = {st: sum(r["status"] == st for r in results) for st in sym}
    refusals = sorted({r["note"].split(" | ")[0] for r in results if r["status"] == "refused"})
    chance = [f"`{r['method']}` / {r['model']} / {r['scenario']} ({r['just_trained_acc']})" for r in results
              if r.get("at_chance")]
    exempt = sorted({f"`{r['method']}` / {r['model']} / {r['scenario']} ({r['just_trained_acc']}%): {r['learn_exempt']}"
                     for r in results if r.get("status_sym") == "ok*"})
    md_exempt = ("\n\nok* = runs correctly, newest task at chance for a known reason:\n\n" +
                 "\n".join(f"- {e}" for e in exempt)) if (exempt and a.learn_check) else ""
    md_chance = md_exempt if a.learn_check else ("\n\nOk cells whose just-trained task is at or below chance (1 epoch, tiny random models; "
                 "review, not necessarily a bug):\n\n" + "\n".join(f"- {c}" for c in chance)) if chance else ""
    mode = (f"{a.epochs} epoch(s) per task" + (", learn check on (just-trained task > chance)" if a.learn_check else ""))
    md = (f"{mode}. {len(results)} cells: {n['ok']} ok, {n['refused']} refused with a clear error, {n['FAIL']} FAIL.\n"
          f"Columns: model, scenario (C = class-IL, T = task-IL, D = domain-IL).\n\n" + "\n".join(lines) +
          "\n\nRefusals (by design):\n\n" + "\n".join(f"- {x}" for x in refusals) + md_chance + "\n")
    open(os.path.join(a.out, "grid.md"), "w").write(md)
    for r in results:
        if r["status"] == "FAIL":
            print(f"\nFAIL {r['method']} / {r['model']} / {r['scenario']}\n{r.get('trace', r['note'])}")
    print(md)
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

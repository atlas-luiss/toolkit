"""Integrity checks: what every reported number relies on.

1. Test data is read only during evaluation (tripwire dataset).
2. No task identity at test time: evaluating all tasks' test sets merged into one
   loader gives exactly the accuracy of evaluating them one by one (a method that used
   *which loader* it was given would differ).
3. Evaluation is read-only: evaluating twice gives the same numbers and leaves weights unchanged.
4. "Frozen backbone" means bit-identical backbone weights.
5. Old tasks' adapters / experts / prompts do not change after their task.
6. Memory budgets are respected.
7. Metrics match hand computation.
8. Reference checks of the maths (independent numpy implementations).
"""

import re
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from conftest import MLP, gauss_data
import cl_toolkit as clt
from cl_toolkit import heads as H, merging
from cl_toolkit import strategies as S
from cl_toolkit.benchmarks import split_by_class
from cl_toolkit.trainers.cl_trainer import compute_cl_metrics

timm = pytest.importorskip("timm")


# ============================================================================ fixtures
class _Tripwire(Dataset):
    """A test set that raises if read while ``armed`` and not inside an evaluation."""
    armed = False
    in_eval = False

    def __init__(self, ds):
        self.ds, self.targets = ds, ds.tensors[1]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if _Tripwire.armed and not _Tripwire.in_eval:
            raise AssertionError("test data read outside evaluation")
        return self.ds[i]


class _EvalGate(clt.callbacks.Callback):
    def on_train_begin(self, trainer, **kw):
        _Tripwire.armed = True

    def on_train_end(self, trainer, results=None, **kw):
        _Tripwire.armed = False

    def on_eval_begin(self, trainer, task_id, eval_task_id, **kw):
        _Tripwire.in_eval = True

    def on_eval_end(self, trainer, task_id, eval_task_id, metrics, **kw):
        _Tripwire.in_eval = False


def vec_bench(num_tasks=3, n_classes=12):
    tr, te = gauss_data(n_classes, n_train=40, n_test=20, sep=1.0)
    return split_by_class(tr, _Tripwire(te), num_tasks, shuffle_classes=False)


def img_bench(num_tasks=2, n_classes=4):
    def mk(seed, n):
        g = torch.Generator().manual_seed(seed)
        y = torch.arange(n_classes).repeat_interleave(n)
        return TensorDataset(torch.randn(len(y), 3, 32, 32, generator=g) * 0.5 + y.view(-1, 1, 1, 1).float() / 2, y)
    return split_by_class(mk(0, 10), _Tripwire(mk(1, 6)), num_tasks, shuffle_classes=False)


def tiny_vit():
    torch.manual_seed(0)
    return clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, freeze_backbone=True,
                               device="cpu", img_size=32)


def mlp():
    torch.manual_seed(0)
    return MLP()


# id -> (model factory, bench factory, clt.build kwargs). Covers every registered strategy,
# every classifier head, every router and every prompt method.
SMALL = {"memory_size": 48}
CASES = {
    "naive": (mlp, vec_bench, dict(strategy="naive")),
    "er": (mlp, vec_bench, dict(strategy="er", strategy_kwargs=SMALL)),
    "er-ace": (mlp, vec_bench, dict(strategy="er-ace", strategy_kwargs=SMALL)),
    "der": (mlp, vec_bench, dict(strategy="der", strategy_kwargs=SMALL)),
    "derpp": (mlp, vec_bench, dict(strategy="derpp", strategy_kwargs=SMALL)),
    "agem": (mlp, vec_bench, dict(strategy="agem", strategy_kwargs=SMALL)),
    "icarl": (mlp, vec_bench, dict(strategy="icarl", strategy_kwargs=SMALL)),
    "gdumb": (mlp, vec_bench, dict(strategy="gdumb", strategy_kwargs={"memory_size": 48, "epochs": 2})),
    "ewc": (mlp, vec_bench, dict(strategy="ewc", strategy_kwargs={"fisher_samples": 32}, freeze_backbone=False)),
    "ewc-online": (mlp, vec_bench, dict(strategy="ewc-online", strategy_kwargs={"fisher_samples": 32},
                                        freeze_backbone=False)),
    "mas": (mlp, vec_bench, dict(strategy="mas", strategy_kwargs={"importance_samples": 32}, freeze_backbone=False)),
    "lwf": (mlp, vec_bench, dict(strategy="lwf")),
    "lora": (mlp, vec_bench, dict(strategy="lora", strategy_kwargs={"rank": 4})),
    "lora-er": (mlp, vec_bench, dict(strategy="lora-er", strategy_kwargs={"rank": 4, **SMALL})),
    "simplecil": (mlp, vec_bench, dict(strategy="simplecil", epochs=0)),
    "ranpac": (mlp, vec_bench, dict(strategy="ranpac", strategy_kwargs={"proj_dim": 256}, epochs=0)),
    "first-session-lora": (mlp, vec_bench, dict(strategy="first-session", strategy_kwargs={"first_session": "lora"},
                                                classifier="ncm")),
    "first-session-adaptformer": (tiny_vit, img_bench, dict(strategy="first-session",
                                                            strategy_kwargs={"first_session": "adaptformer"},
                                                            classifier="lda")),
    "magmax": (mlp, vec_bench, dict(strategy="magmax", freeze_backbone=False)),
    "ties": (mlp, vec_bench, dict(strategy="ties", freeze_backbone=False)),
    **{m: (mlp, vec_bench, dict(strategy=m, freeze_backbone=False))
       for m in ("task-arithmetic", "dare", "breadcrumbs", "consensus", "tsv", "iso-c", "iso-cts",
                 "weight-average", "coma")},
    "cofima": (mlp, vec_bench, dict(strategy="cofima", strategy_kwargs={"fisher_samples": 16}, freeze_backbone=False)),
    **{f"head-{h}": (mlp, vec_bench, dict(strategy="frozen", classifier=h, epochs=0,
                                          **({"classifier_kwargs": {"proj_dim": 256}} if h == "ranpac" else {})))
       for h in ("ncm", "fecam", "lda", "klda", "ranpac")},
    "head-slca": (mlp, vec_bench, dict(strategy="naive", classifier="slca")),
    "head-wa": (mlp, vec_bench, dict(strategy="er", strategy_kwargs=SMALL, classifier="wa")),
    "head-cosine": (mlp, vec_bench, dict(strategy="naive", classifier="cosine")),
    **{f"routed-lora-{r}": (mlp, vec_bench, dict(experts="lora", router=r, strategy_kwargs={"experts_kwargs": {"rank": 4}}))
       for r in ("hide", "lda", "sprompts", "keyquery", "expertgate")},
    "routed-adaptformer-hide": (tiny_vit, img_bench, dict(experts="adaptformer", router="hide")),
    "routed-lora-hide-head-lda": (mlp, vec_bench, dict(experts="lora", router="hide", classifier="lda",
                                                       strategy_kwargs={"experts_kwargs": {"rank": 4}})),
    "routed-prompt-hide-head-ncm": (tiny_vit, img_bench, dict(experts="prompt", router="hide", classifier="ncm")),
    "hide-prompt": (tiny_vit, img_bench, dict(strategy="hide-prompt")),
    "s-prompts": (tiny_vit, img_bench, dict(strategy="s-prompts")),
    "expert-gate": (tiny_vit, img_bench, dict(strategy="expert-gate")),
    "l2p": (tiny_vit, img_bench, dict(strategy="l2p")),
    "dualprompt": (tiny_vit, img_bench, dict(strategy="dualprompt")),
    "coda-prompt": (tiny_vit, img_bench, dict(strategy="coda-prompt", strategy_kwargs={"pool_size": 8})),
}


def build(case, scenario="class", **over):
    model_f, bench_f, kw = CASES[case]
    kw = {**kw, **over}
    bench = bench_f()
    torch.manual_seed(0)
    tr = clt.build(bench, model_f(), scenario, num_tasks=bench.num_tasks, epochs=kw.pop("epochs", 2),
                   batch_size=16, num_workers=0, device="cpu", verbose=False, lr=5e-3,
                   callbacks=[_EvalGate()], **kw)
    return tr


def test_every_registered_method_is_covered():
    names = set(clt.available()["strategy"])
    covered = {c for c in CASES} | {"finetune", "replay", "der++", "first-session", "frozen", "routed"}
    covered |= {"lora-er", "s-prompts", "expert-gate", "hide-prompt"}
    missing = {n for n in names if n not in covered and n not in ("coda", "slda")}
    assert not missing, f"integrity suite misses {missing}"


def _norm(name):
    """Parameter name with peft's wrapping removed (LoraModel.model.<name>, <layer>.base_layer)."""
    name = re.sub(r"\.mlp\.norm\.", ".norm2.", name.replace(".base_layer", "")).replace(".mlp.mlp.", ".mlp.")    # peft / AdaptFormer wrapping
    return name[len("model."):] if name.startswith("model.") else name


# ============================================================================ 1-3: data, task identity, read-only eval
def _eval(tr, loader):
    model, strat = tr.model, tr.strategy
    T = tr.scenario.num_tasks
    _Tripwire.in_eval = True
    try:
        model.eval()
        tr._set_eval_mask(T - 1)
        strat.before_eval(model, None)
        with torch.no_grad(), tr.autocast():
            return (strat.evaluate_task(loader, model, None) or tr.evaluate_task(loader, None))["accuracy"]
    finally:
        _Tripwire.in_eval = False


def _test_sets(tr):
    return [t.test_loader.dataset for t in tr.scenario.tasks]


def check_merged_equals_per_task(tr):
    """No task identity from the loader: one merged loader == the per-task loaders."""
    T = tr.scenario.num_tasks
    sizes = np.array([len(d) for d in _test_sets(tr)])
    expected = float((tr.acc[T - 1, :T] / 100 * sizes).sum())
    got = _eval(tr, DataLoader(ConcatDataset(_test_sets(tr)), batch_size=7)) / 100 * sizes.sum()
    assert abs(got - expected) <= 1.0 + 1e-6, ("merged != per-task", got, expected)


class _Relabel(Dataset):
    def __init__(self, ds, c):
        self.ds, self.c = ds, c

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0], self.c


def check_labels_not_used(tr):
    """Relabel every test sample as class c: accuracy = share predicted as c. If predictions
    do not depend on the labels, these shares over all seen classes sum to exactly 100%."""
    merged = ConcatDataset(_test_sets(tr))
    classes = tr.scenario.seen_classes(tr.scenario.num_tasks - 1)
    total = sum(_eval(tr, DataLoader(_Relabel(merged, c), batch_size=16)) for c in classes)
    assert abs(total - 100.0) <= 100.0 / len(merged) + 1e-6, ("predictions depend on test labels", total)


def check_eval_read_only(tr):
    T = tr.scenario.num_tasks
    state = {k: v.clone() for k, v in tr.model.state_dict().items()}
    final = tr.acc[T - 1, :T].copy()
    armed, _Tripwire.armed = _Tripwire.armed, False
    again = tr.evaluate_seen(T - 1)
    _Tripwire.armed = armed
    assert np.allclose(again, final), ("evaluation is not repeatable", again, final)
    for k, v in tr.model.state_dict().items():
        assert torch.equal(v, state[k]), f"evaluation changed {k}"


@pytest.mark.parametrize("case", list(CASES))
def test_no_leakage(case):
    tr = build(case)
    r = tr.fit()                                  # tripwire: raises if test data is touched in training
    assert np.isfinite(r["avg_accuracy"])
    check_merged_equals_per_task(tr)
    check_labels_not_used(tr)
    check_eval_read_only(tr)


# ---------------------------------------------------------------------------- the checks catch cheaters
class _PeeksAtLabels(S.NaiveStrategy):
    def evaluate_task(self, loader, model, task_id):
        return {"accuracy": 100.0}                           # "predicts" every label correctly


class _UsesLoaderIdentity(S.NaiveStrategy):
    """Counts evaluate_task calls to know which task's loader it is given (task-id leak)."""
    calls = 0

    def evaluate_task(self, loader, model, task_id):
        self.calls += 1
        return {"accuracy": 100.0 if self.calls % self.scenario.num_tasks == 0 else 50.0}


class _LearnsAtTest(S.NaiveStrategy):
    def evaluate_task(self, loader, model, task_id):
        with torch.no_grad():
            model.get_head()._fc.bias.add_(0.1)
        return None


class _TrainsOnTest(S.NaiveStrategy):
    def on_task_end(self, task_id, train_loader, model):
        next(iter(self.scenario.get_task(task_id).test_loader))


@pytest.mark.parametrize("cheat,check", [
    (_PeeksAtLabels, check_labels_not_used), (_UsesLoaderIdentity, check_merged_equals_per_task),
    (_LearnsAtTest, check_eval_read_only), (_TrainsOnTest, None)])
def test_checks_catch_cheating(cheat, check):
    tr = clt.build(vec_bench(), mlp(), "class", cheat(), epochs=1, batch_size=16, num_workers=0, device="cpu",
                   verbose=False, callbacks=[_EvalGate()])
    with pytest.raises(AssertionError):
        tr.fit()
        check(tr)
    _Tripwire.armed = _Tripwire.in_eval = False


# ============================================================================ 4: frozen backbone
FROZEN_CASES = [c for c, (_, _, kw) in CASES.items() if kw.get("freeze_backbone", True)]


@pytest.mark.parametrize("case", FROZEN_CASES)
def test_frozen_backbone_is_bit_identical(case):
    tr = build(case)
    before = {_norm(n): p.detach().clone() for n, p in tr.model._backbone.named_parameters()}
    tr.fit()
    after = {_norm(n): p for n, p in tr.model._backbone.named_parameters() if "lora_" not in n and "ctk_adapter" not in n}
    for n, p in before.items():
        assert n in after, f"{case}: backbone param {n} disappeared"
        assert torch.equal(after[n], p), f"{case}: frozen backbone param {n} changed"


# ============================================================================ 5: old task modules stay fixed
OWNED = {   # case -> pattern of parameter names owned by task k
    "lora": ".task_{k}.", "lora-er": ".task_{k}.",
    "routed-lora-hide": ".expert_{k}.", "routed-lora-lda": ".expert_{k}.",
    "routed-adaptformer-hide": "ctk_adapter.expert_{k}.", "expert-gate": "ctk_adapter.expert_{k}.",
    "hide-prompt": "prompts.{k}", "s-prompts": "prompts.{k}",
    "dualprompt": "e_prompts.{k}", "coda-prompt": ".P.{k}",
}


@pytest.mark.parametrize("case", list(OWNED))
def test_old_task_modules_do_not_change(case):
    snaps = {}

    class Snap(clt.callbacks.Callback):
        def on_task_end(self, trainer, task_id, task_metrics, **kw):
            pat = OWNED[case].format(k=task_id)
            snaps[task_id] = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if pat in n}

    tr = build(case)
    tr.callbacks.callbacks.append(Snap())
    tr.fit()
    params = dict(tr.model.named_parameters())
    for k, snap in snaps.items():
        assert snap, f"{case}: no parameters matched {OWNED[case].format(k=k)}"
        for n, p in snap.items():
            assert torch.equal(params[n], p), f"{case}: task {k}'s {n} changed after task {k}"


# ============================================================================ 6: memory budgets
def _memory(strat):
    s = getattr(strat, "strategy", strat)
    return getattr(s, "buffer", None) if hasattr(s, "buffer") else s.memory


@pytest.mark.parametrize("case", ["er", "er-ace", "der", "derpp", "agem", "icarl", "gdumb", "lora-er"])
def test_memory_budget(case):
    sizes = []

    class Count(clt.callbacks.Callback):
        def on_task_end(self, trainer, task_id, task_metrics, **kw):
            sizes.append(len(_memory(trainer.strategy)))

    tr = build(case)
    tr.callbacks.callbacks.append(Count())
    tr.fit()
    assert sizes and max(sizes) <= SMALL["memory_size"], (case, sizes)
    assert sizes[-1] > 0


# ============================================================================ 7: metrics
def test_metrics_by_hand():
    acc = np.array([[90.0, np.nan, np.nan],
                    [70.0, 80.0, np.nan],
                    [60.0, 75.0, 85.0]])
    m = compute_cl_metrics(acc, test_sizes=[100, 100, 200])
    assert m["avg_accuracy"] == pytest.approx((60 + 75 + 85) / 3)
    assert m["avg_incremental_accuracy"] == pytest.approx((90 + 75 + (60 + 75 + 85) / 3) / 3)
    assert m["avg_forgetting"] == pytest.approx(((90 - 60) + (80 - 75)) / 2)
    assert m["backward_transfer"] == pytest.approx(((60 - 90) + (75 - 80)) / 2)
    assert m["accuracy_all_seen"] == pytest.approx((60 * 100 + 75 * 100 + 85 * 200) / 400)
    assert m["avg_incremental_accuracy_all_seen"] == pytest.approx((90 + 75 + 76.25) / 3)


# ============================================================================ 8: maths vs independent references
class _Scen:
    """Minimal scenario stand-in for head unit tests."""

    def __init__(self, C):
        self.benchmark = type("Bm", (), {"total_classes": C})()


def _gauss(C=5, D=6, n=60, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(C, D)) * 2
    y = np.repeat(np.arange(C), n)
    x = mu[y] + rng.normal(size=(len(y), D)) @ rng.normal(size=(D, D)) * 0.5
    return x, y


def test_lda_matches_numpy_reference():
    x, y = _gauss()
    h = H.LDA(shrinkage=0.0)
    h.bind(None, _Scen(5))
    h.update(torch.tensor(x, dtype=torch.float32), torch.tensor(y), 0)
    mu = np.stack([x[y == c].mean(0) for c in range(5)])
    S = sum(((x[y == c] - mu[c]).T @ (x[y == c] - mu[c])) for c in range(5)) / len(x)
    P = np.linalg.inv(S)
    ref = x @ P @ mu.T - 0.5 * np.einsum("cd,de,ce->c", mu, P, mu)
    ours = h.logits(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.allclose(ours, ref, rtol=1e-3, atol=1e-2)
    assert (ours.argmax(1) == ref.argmax(1)).mean() > 0.999


def test_lda_streaming_equals_batch():
    x, y = _gauss()
    a, b = H.LDA(), H.LDA()
    for h in (a, b):
        h.bind(None, _Scen(5))
    X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    a.update(X, Y, 0)
    b.update(X[Y < 2], Y[Y < 2], 0)
    b.update(X[Y >= 2], Y[Y >= 2], 1)
    assert torch.allclose(a.logits(X), b.logits(X), atol=1e-3)


def test_ncm_matches_reference():
    x, y = _gauss()
    h = H.NCM()
    h.bind(None, _Scen(5))
    h.update(torch.tensor(x, dtype=torch.float32), torch.tensor(y), 0)
    mu = np.stack([x[y == c].mean(0) for c in range(5)])
    xn, mn = x / np.linalg.norm(x, axis=1, keepdims=True), mu / np.linalg.norm(mu, axis=1, keepdims=True)
    assert np.allclose(h.logits(torch.tensor(x, dtype=torch.float32)).numpy(), xn @ mn.T, atol=1e-5)


def test_ranpac_is_exact_ridge_regression():
    x, y = _gauss(D=8)
    h = H.RanPAC(proj_dim=64, lambdas=[0.5])
    h.bind(None, _Scen(5))
    X = torch.tensor(x, dtype=torch.float32)
    h.update(X[torch.tensor(y) < 3], torch.tensor(y)[torch.tensor(y) < 3], 0)   # two tasks: accumulated Gram
    h.update(X[torch.tensor(y) >= 3], torch.tensor(y)[torch.tensor(y) >= 3], 1)
    W = h._W.numpy().astype(np.float64)
    Hm = np.maximum(x @ W, 0)
    Yo = np.eye(5)[y]
    Wout = np.linalg.solve(Hm.T @ Hm + 0.5 * np.eye(64), Hm.T @ Yo)
    assert np.allclose(h.logits(X).numpy(), Hm @ Wout, rtol=1e-3, atol=1e-3)


def test_ranpac_ridge_selection_uses_training_data_only():
    """The ridge is chosen inside ``update`` (training features); nothing else is passed in."""
    import inspect
    assert list(inspect.signature(H.RanPAC.update).parameters) == ["self", "feats", "labels", "task_id"]


def test_klda_random_features_approximate_rbf_kernel():
    h = H.KLDA(rff_dim=20000, gamma=0.1, seed=0)
    x = torch.randn(20, 6)
    z = h.features(x)
    K = torch.exp(-0.1 * torch.cdist(x, x).pow(2))
    assert (z @ z.T - K).abs().max() < 0.05


def test_fecam_matches_reference():
    x, y = _gauss(D=4)
    h = H.FeCAM(shrink2=1.0, tukey=False, normalize=False)
    h.bind(None, _Scen(5))
    h.update(torch.tensor(x, dtype=torch.float32), torch.tensor(y), 0)
    ref = np.zeros((len(x), 5))
    for c in range(5):
        xc = x[y == c]
        mu, cov = xc.mean(0), np.cov(xc.T, bias=True)
        D = 4
        off = (cov.sum() - np.trace(cov)) / (D * D - D)
        cov = cov + np.trace(cov) / D * np.eye(D) + off * (1 - np.eye(D))
        sd = np.sqrt(np.diag(cov))
        cov = cov / np.outer(sd, sd)
        d = x - mu
        ref[:, c] = -np.einsum("nd,de,ne->n", d, np.linalg.pinv(cov), d)
    assert np.allclose(h.logits(torch.tensor(x, dtype=torch.float32)).numpy(), ref, rtol=1e-3, atol=1e-2)


def test_reservoir_is_uniform():
    counts = np.zeros(100)
    for trial in range(400):
        torch.manual_seed(trial)
        b = S.ReservoirBuffer(10)
        for i in range(0, 100, 5):
            b.add(torch.arange(i, i + 5).float().unsqueeze(1), torch.zeros(5, dtype=torch.long), 0)
        counts[b.x[:, 0].long().numpy()] += 1
    p = counts / 400                       # each item kept with prob 10/100
    assert abs(p.mean() - 0.1) < 1e-9 and p[:50].mean() == pytest.approx(p[50:].mean(), abs=0.02)


def test_agem_projected_gradient_does_not_increase_memory_loss():
    torch.manual_seed(0)
    tr = clt.build(vec_bench(num_tasks=2), mlp(), "class", "agem", strategy_kwargs=SMALL, epochs=1,
                   num_workers=0, device="cpu", verbose=False, freeze_backbone=False)
    strat, dots = tr.strategy, []
    orig = strat.before_update

    def spy(model, optimizer):
        orig(model, optimizer)
        if len(strat.buffer):
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            g = torch.cat([p.grad.flatten() for p in params])
            xm, ym, tm, _ = strat.buffer.sample(10 ** 6, model._device)
            with strat.memory_mask(model, strat.task.task_id):
                ref = torch.nn.functional.cross_entropy(model(xm, tm), ym)
            gr = torch.cat([q.flatten() for q in torch.autograd.grad(ref, params)])
            dots.append(float(g @ gr) / (float(g.norm() * gr.norm()) + 1e-12))
    strat.before_update = spy
    strat.ref_sample_size = 10 ** 6                    # use the whole memory so the check is exact
    _Tripwire.armed = False
    tr.fit()
    assert dots and min(dots) > -1e-4


def test_ewc_fisher_is_mean_per_sample_squared_gradient(monkeypatch):
    # deterministic "sampled" labels so both computations use the same ones
    monkeypatch.setattr(torch, "multinomial", lambda p, n: p.argmax(1, keepdim=True))
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    model = clt.models.wrap(nn.Identity(), embed_dim=4, device="cpu")
    model.set_head(clt.models.base.IncrementalClassifier(4, 3))
    model.get_head()._fc.load_state_dict(lin.state_dict())
    x, y = torch.randn(10, 4), torch.randint(0, 3, (10,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    ewc = S.EWC(fisher_samples=10)
    ewc.trainer = type("T", (), {"autocast": staticmethod(lambda: torch.autocast("cpu", enabled=False))})()
    torch.manual_seed(1)
    F_ours = ewc.importance(loader, model, 0)
    torch.manual_seed(1)
    w = model.get_head()._fc.weight
    acc = torch.zeros_like(w)
    for i in range(10):
        logp = torch.log_softmax(model(x[i:i + 1], 0), 1)
        yhat = torch.multinomial(logp.exp(), 1).squeeze(1)
        g, = torch.autograd.grad(torch.nn.functional.nll_loss(logp, yhat), w)
        acc += g.pow(2)
    assert torch.allclose(F_ours[w], acc / 10, atol=1e-6)


def test_ewc_keeps_one_anchor_per_task():
    s = S.EWC()
    s._terms = []
    p = nn.Parameter(torch.zeros(2))
    s.importance = lambda *a: {p: torch.ones(2)}
    s.on_task_end(0, None, None)
    with torch.no_grad():
        p.add_(1.0)
    s.on_task_end(1, None, None)
    with torch.no_grad():
        p.zero_()
    # anchors at 0 and 1: penalty = lam/2 * (|0-0|^2*2 + |0-1|^2*2) = lam/2 * 2
    assert float(s.penalty().detach()) == pytest.approx(s.lam / 2 * 2)


def test_ties_and_magmax_by_hand():
    a = {"w": torch.tensor([1.0, -2.0, 0.5, 3.0])}
    b = {"w": torch.tensor([-1.5, -1.0, 0.4, 2.0])}
    assert torch.equal(merging.magmax([a, b], lam=1.0)["w"], torch.tensor([-1.5, -2.0, 0.5, 3.0]))
    t = merging.ties([a, b], density=1.0)["w"]
    # sign of sums: -, -, +, +  -> disjoint means: -1.5, -1.5, 0.45, 2.5
    assert torch.allclose(t, torch.tensor([-1.5, -1.5, 0.45, 2.5]))


def test_wa_equalises_mean_norms():
    tr = clt.build(vec_bench(num_tasks=2), mlp(), "class", "er", strategy_kwargs=SMALL, classifier="wa",
                   epochs=2, num_workers=0, device="cpu", verbose=False)
    tr.fit()
    fc = tr.model.get_head()._fc
    n = fc.weight.detach().norm(dim=1)
    new, old = tr.scenario.get_task(1).classes, tr.scenario.get_task(0).classes
    assert float(n[new].mean()) == pytest.approx(float(n[old].mean()), rel=1e-4)


@pytest.mark.parametrize("name", ["magmax", "tsv", "coma"])
def test_merging_needs_trainable_backbone(name):
    with pytest.raises(ValueError, match="nothing to merge|no trainable backbone"):
        clt.build(vec_bench(num_tasks=2), mlp(), "class", name, epochs=1, num_workers=0,
                  device="cpu", verbose=False).fit()


def test_preprocess_presets():
    from cl_toolkit.benchmarks.transforms import image_transforms
    from torchvision import transforms as T
    _, te = image_transforms("model", low_res=True, size=224, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert not any(isinstance(t, T.CenterCrop) for t in te.transforms)          # CIFAR: never cropped
    for preset in ("pilot", "coda"):
        tr, te = image_transforms(preset, low_res=True)
        assert not any(isinstance(t, T.Normalize) for t in tr.transforms + te.transforms)   # as in those code bases
    with pytest.raises(ValueError):
        image_transforms("timm")

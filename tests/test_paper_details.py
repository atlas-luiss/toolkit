"""Details checked against the authors' code (see VERIFICATION.md, section 3)."""

import os
import sys
import types

import numpy as np

import pytest
import torch
import torch.nn as nn

from conftest import MLP, gauss_data
import cl_toolkit as clt
from cl_toolkit import adapters, heads as H
from cl_toolkit.benchmarks import split_by_class


# ---------------------------------------------------------------------------- SLCA (GengDavid/SLCA)
class _Scen:
    def __init__(self, tasks):
        self.tasks = tasks

        class Bench:
            total_classes = sum(len(t) for t in tasks)
        self.benchmark = Bench()

    def seen_classes(self, t):
        return [c for k in range(t + 1) for c in self.tasks[k]]

    def get_task(self, t):
        return types.SimpleNamespace(classes=self.tasks[t])


def _slca(**kw):
    h = H.SLCA(**kw)
    h.bind(None, _Scen([[0, 1], [2, 3], [4, 5]]))
    g = torch.Generator().manual_seed(0)
    for t, cls in enumerate([[0, 1], [2, 3], [4, 5]]):
        y = torch.tensor(cls).repeat_interleave(20)
        h.update(torch.randn(len(y), 4, generator=g) + 3.0, y, t)
    return h


def test_slca_mean_decay_matches_authors():
    # authors: cls_mean * (0.9 + (t_id + 1) / (cur_task + 1) * 0.1)
    h = _slca(samples_per_class=4000, shrinkage=0.0)
    x, y = h._sample(torch.Generator().manual_seed(0), task_id=2)
    for c, k in [(0, 0), (2, 1), (4, 2)]:
        want = h._stats[c][0].float() * (0.9 + 0.1 * (k + 1) / 3)
        assert torch.allclose(x[y == c].mean(0), want, atol=0.08), c


def test_slca_skips_alignment_after_first_task():
    h = _slca()
    h.model = None                       # would fail if used
    h.after_task(0, trainer=None)        # authors: CA only when cur_task > 0


def test_slca_logit_norm_is_mean_of_per_task_norms():
    tr = clt.build(gauss_data(6, n_train=20, n_test=5), MLP(), "class", "naive", classifier="slca",
                   classifier_kwargs={"epochs": 1, "samples_per_class": 8}, num_tasks=3, epochs=1,
                   num_workers=0, device="cpu", verbose=False)
    seen = []
    orig = nn.functional.cross_entropy

    def spy(logits, y, *a, **k):
        seen.append(logits.detach())
        return orig(logits, y, *a, **k)

    tr.fit()                                          # full run works
    head = tr.strategy.head
    nn.functional.cross_entropy = spy
    try:
        head.after_task(2, tr)
    finally:
        nn.functional.cross_entropy = orig
    z = seen[0] * head.logit_norm                     # undo temperature
    tasks = [tr.scenario.get_task(k).classes for k in range(3)]
    cols = {c: i for i, c in enumerate(tr.scenario.seen_classes(2))}
    per_task = torch.stack([z[:, [cols[c] for c in t]].norm(dim=1) for t in tasks], 1)
    assert torch.allclose(per_task.mean(1), torch.ones(len(z)), atol=1e-4)


# ---------------------------------------------------------------------------- KLDA-E
def test_klda_ensemble_averages_member_probabilities():
    x, y = torch.randn(60, 8), torch.arange(3).repeat_interleave(20)
    scen = _Scen([[0, 1, 2]])
    e = H.KLDA(rff_dim=64, ensembles=3, seed=5)
    e.bind(None, scen)
    e.update(x, y, 0)
    singles = []
    for i in range(3):
        s = H.KLDA(rff_dim=64, seed=5 + i)
        s.bind(None, scen)
        s.update(x, y, 0)
        singles.append(torch.softmax(s.logits(x), 1))
    assert torch.allclose(e.logits(x).exp(), torch.stack(singles).mean(0), atol=1e-5)


# ---------------------------------------------------------------------------- AdaptFormer (APER / RanPAC code)
timm = pytest.importorskip("timm")


def test_adaptformer_reads_block_input_before_norm2():
    torch.manual_seed(0)
    m = clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, device="cpu", img_size=32).eval()
    x = torch.randn(2, 3, 32, 32)
    before = m.encode(x)
    adapters.add_adaptformer(m, bottleneck=4)
    assert torch.allclose(m.encode(x), before, atol=1e-6)          # zero-init up: the PTM function
    blk = m.vit.blocks[0]
    assert isinstance(blk.norm2, nn.Identity)
    got = {}
    blk.mlp.ctk_adapter["default"].register_forward_hook(lambda mod, inp, out: got.setdefault("x", inp[0]))
    blk.mlp.norm.register_forward_hook(lambda mod, inp, out: got.setdefault("pre", inp[0]))
    m.encode(x)
    assert torch.equal(got["x"], got["pre"])                       # adapter input = input of norm2


# ---------------------------------------------------------------------------- LLM pooling
def test_last_token_pooling_any_padding_side():
    h = torch.arange(2 * 5 * 1, dtype=torch.float32).view(2, 5, 1)
    module = lambda input_ids, attention_mask: types.SimpleNamespace(last_hidden_state=h)
    fake = types.SimpleNamespace(pooling="last")
    right = {"input_ids": None, "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])}
    left = {"input_ids": None, "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])}
    assert clt.models.HFModel._hf_encode(fake, module, right).flatten().tolist() == [2.0, 9.0]
    assert clt.models.HFModel._hf_encode(fake, module, left).flatten().tolist() == [4.0, 9.0]


# ---------------------------------------------------------------------------- clear refusals
def test_prompts_on_a_non_vit_model_are_refused_clearly():
    data = split_by_class(*gauss_data(4, n_train=10, n_test=5), 2, shuffle_classes=False)
    with pytest.raises(TypeError, match="timm ViT"):
        clt.build(data, MLP(), "class", "l2p", num_workers=0, device="cpu", verbose=False)
    tr = clt.build(data, MLP(), "class", "hide-prompt", num_workers=0, device="cpu", verbose=False)
    with pytest.raises(TypeError, match="timm ViT"):
        tr.fit()


# ---------------------------------------------------------------------------- found by experiments/interop_grid.py
def test_agem_projects_in_task_il():
    """Task-IL: the current task's head gets no gradient from the memory loss. A-GEM must still
    project (zero reference gradient there) instead of silently skipping the projection."""
    from cl_toolkit import strategies as S
    data = split_by_class(*gauss_data(6, n_train=20, n_test=5), 3, shuffle_classes=False)
    strat = S.AGEM(memory_size=30, ref_sample_size=16)
    calls = {"projected": 0}
    orig = S.AGEM.before_update

    def spy(self, model, opt):
        before = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]
        orig(self, model, opt)
        after = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        calls["projected"] += any(not torch.equal(a, b) for a, b in zip(after, before))

    strat.before_update = types.MethodType(spy, strat)
    clt.build(data, MLP(), "task", strat, freeze_backbone=False, epochs=2, lr=0.05, num_workers=0, device="cpu",
              verbose=False).fit()
    assert calls["projected"] > 0


def test_freeze_backbone_false_unfreezes_a_prebuilt_model():
    m = MLP()
    m.freeze_backbone()
    tr = clt.build(gauss_data(4, n_train=10, n_test=5), m, "class", "magmax", freeze_backbone=False, num_tasks=2,
                   num_workers=0, device="cpu", verbose=False)
    assert all(p.requires_grad for p in tr.model.backbone.parameters())


def test_merging_svd_survives_degenerate_inputs():
    from cl_toolkit import merging as M
    tvs = [{"w": torch.zeros(6, 6)}, {"w": torch.zeros(6, 6)}, {"w": torch.ones(6, 6) * 1e-30}]
    for m in ("tsv", "iso_c", "iso_cts"):
        out = M.merge(tvs, m)["w"]
        assert torch.isfinite(out).all(), m


def _routed_lda(scenario="class"):
    from conftest import gauss_benchmark, gauss_domains
    bench = gauss_domains() if scenario == "domain" else gauss_benchmark(num_tasks=3, n_classes=9)
    torch.manual_seed(0)
    router = "sprompts" if scenario == "domain" else "hide"
    tr = clt.build(bench, MLP(), scenario, experts="lora", router=router, classifier="lda",
                   strategy_kwargs={"experts_kwargs": {"rank": 4}}, num_tasks=bench.num_tasks, epochs=3,
                   lr=5e-3, num_workers=0, device="cpu", verbose=False)
    tr.fit()
    return tr


def test_per_expert_head_matches_an_independent_computation():
    """Recompute every Class-IL prediction by hand: route with the router on plain features, take the
    routed expert's features, fit a fresh LDA on that task's training data through the same expert,
    predict among that task's classes."""
    from cl_toolkit import batch as Bt
    tr = _routed_lda()
    strat, model, scen = tr.strategy, tr.model, tr.scenario
    assert isinstance(strat.head, H.LDA) and len(strat._heads) == scen.num_tasks
    model.eval()
    ref = {}
    for t in range(scen.num_tasks):
        h = H.LDA()
        h.bind(model, scen)
        xs, ys = zip(*[(x, y) for x, y in scen.feature_loader(t, global_labels=True)])
        x, y = torch.cat(xs), torch.cat(ys)
        with torch.no_grad():
            f = strat.experts.encode_with(model, x, torch.full((len(x),), t))
        h.update(f, y, t)
        ref[t] = h
    correct = total = 0
    with torch.no_grad():
        for task in scen.tasks:
            for x, y in task.test_loader:
                strat.experts.use_none(model)
                ids = strat.router.route(model.encode(x))
                f = strat.experts.encode_with(model, x, ids)
                for i in range(len(x)):
                    t = int(ids[i])
                    s = ref[t].logits(f[i:i + 1])[0]
                    cls = torch.tensor(scen.get_task(t).classes)
                    pred = cls[s[cls].argmax()]
                    correct += int(pred == y[i])
                    total += 1
    T = scen.num_tasks
    sizes = [len(t.test_loader.dataset) for t in scen.tasks]
    got = sum(tr.acc[T - 1, k] / 100 * sizes[k] for k in range(T))
    assert abs(got - correct) < 1e-6, (got, correct)
    assert 100 * correct / total > 100 / 9 * 2                          # learned something


def test_per_expert_heads_only_score_their_own_task():
    tr = _routed_lda()
    x = torch.randn(5, 32)
    for t, h in enumerate(tr.strategy._heads):
        finite = torch.isfinite(h.logits(torch.randn(5, 64))).all(0).nonzero().flatten().tolist()
        assert finite == tr.scenario.get_task(t).classes


@pytest.mark.parametrize("scenario", ["task", "domain"])
def test_per_expert_head_other_scenarios(scenario):
    tr = _routed_lda(scenario)
    T = tr.scenario.num_tasks
    assert np.isfinite(tr.acc[T - 1, :T]).all() and tr.acc[T - 1, :T].mean() > (50 if scenario == "task" else 20)


def test_head_corrections_with_routed_experts_are_refused():
    data = gauss_data(4, n_train=10, n_test=5)
    for h in ("slca", "wa"):
        with pytest.raises(ValueError, match="nothing for it to correct"):
            clt.build(data, MLP(), "class", classifier=h, experts="lora", router="hide", num_tasks=2,
                      num_workers=0, device="cpu", verbose=False)
    from cl_toolkit import strategies as S
    with pytest.raises(ValueError, match="Routed"):            # the wrapper would ignore the experts
        S.WithHead(S.Routed("lora", "hide"), H.LDA())


def test_cpu_runs_do_not_pin_memory():
    tr = clt.build(gauss_data(4, n_train=10, n_test=5), MLP(), "class", "naive", num_tasks=2, num_workers=0,
                   device="cpu", verbose=False)
    assert not tr.scenario.pin_memory and not tr.scenario.tasks[0].train_loader.pin_memory


def test_rerun_into_same_folder_keeps_logs_separate(tmp_path):
    kw = dict(num_tasks=2, epochs=1, num_workers=0, device="cpu", verbose=False, out_dir=str(tmp_path / "r"))
    clt.run(gauss_data(4, n_train=10, n_test=5), MLP(), strategy="naive", **kw)
    clt.run(gauss_data(4, n_train=10, n_test=5), MLP(), strategy="er", **kw)
    log = (tmp_path / "r" / "log.txt").read_text()
    assert log.count("logging to") == 1 and "previous run in this folder moved to" in log
    prev = [p for p in (tmp_path / "r").iterdir() if p.name.startswith("previous_")]
    assert len(prev) == 1 and {f.name for f in prev[0].iterdir()} == {"log.txt", "config.json", "results.json"}


def test_same_seed_same_result_across_processes():
    """Results must not depend on Python's per-process string hashing (found: set order in merging)."""
    import json
    import subprocess
    code = ("import sys, json, torch; sys.path.insert(0, 'tests'); import cl_toolkit as clt; "
            "from conftest import MLP, gauss_data; torch.manual_seed(0); "
            "r = clt.run(gauss_data(6, n_train=20, n_test=5), MLP(), strategy='dare', freeze_backbone=False, "
            "num_tasks=3, epochs=2, num_workers=0, device='cpu', verbose=False, out_dir=None); "
            "print(json.dumps(r['accuracy_matrix'].tolist()))")
    root = os.path.join(os.path.dirname(__file__), "..")
    outs = [subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": h, "PYTHONWARNINGS": "ignore"}).stdout.strip().splitlines()[-1]
            for h in ("1", "2", "3", "4", "5")]
    assert len(set(outs)) == 1, outs


def test_toy_quick_start_defaults():
    tr = clt.build("toy", "mlp", strategy="er")
    assert tr.scenario.num_workers == 0                         # in-memory data: no worker processes
    assert all(p.requires_grad for p in tr.model.backbone.parameters())   # untrained model is trained
    assert tr.scenario.num_tasks == 10 and tr.scenario.benchmark.total_classes == 20
    pre = clt.build("toy", clt.models.mlp(), strategy="er")    # passed-in model: frozen by default
    assert not any(p.requires_grad for p in pre.model.backbone.parameters())
    assert clt.build("toy", "mlp", strategy="er", freeze_backbone=True).model.backbone.parameters().__next__().requires_grad is False

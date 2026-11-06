"""Expert routing: isolation of experts, stable queries, routers, scenarios."""

import pytest
import torch

from conftest import MLP, gauss_benchmark, gauss_domains
import cl_toolkit as clt
from cl_toolkit import ContinualTrainer, Scenario, routing as R
from cl_toolkit.strategies import RoutedExperts, Routed

ROUTERS = {
    "keyquery": lambda: R.KeyQuery(),
    "sprompts": lambda: R.SPrompts(k=3),
    "hide": lambda: R.HiDe(epochs=10),
    "expertgate": lambda: R.ExpertGate(hidden=16, epochs=30),
    "lda": lambda: R.HeadRouter("lda"),
}


def fit(strategy, kind="class", model=None, bench=None, epochs=3, **kw):
    torch.manual_seed(0)
    bench = bench or (gauss_domains() if kind == "domain" else gauss_benchmark())
    scen = Scenario(bench, kind, batch_size=32, num_workers=0)
    tr = ContinualTrainer(model or MLP(), scen, strategy, lr=3e-3, epochs_per_task=epochs, verbose=False, **kw)
    return tr, tr.fit()


@pytest.mark.parametrize("router", list(ROUTERS))
def test_lora_experts_class_il_route_well(router):
    tr, r = fit(RoutedExperts(R.LoRAExperts(rank=4), ROUTERS[router]()))
    assert "final_routing_accuracy" in r
    assert r["final_routing_accuracy"] > 40, (router, r["final_routing_accuracy"])   # chance = 25
    assert r["avg_accuracy"] > 20


def test_task_il_uses_given_task_and_routes_perfectly():
    _, r = fit(Routed("lora", "sprompts", experts_kwargs={"rank": 4}), "task")
    assert r["final_routing_accuracy"] == 100.0 and r["avg_accuracy"] > 50


def test_domain_il_task_level_routers_only():
    _, r = fit(Routed("lora", "sprompts", experts_kwargs={"rank": 4}), "domain")
    assert r["avg_accuracy"] > 40 and "final_routing_accuracy" not in r   # no label->task map in Domain-IL
    with pytest.raises(ValueError, match="supports scenarios"):
        fit(Routed("lora", "hide", experts_kwargs={"rank": 4}), "domain")


def test_query_is_plain_ptm_and_experts_are_isolated():
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(16, 32)
    with torch.no_grad():
        q0 = model.encode(x).clone()
    snaps = {}

    class Snap(clt.callbacks.Callback):
        def on_task_end(self, trainer, task_id, task_metrics, **kw):
            snaps[task_id] = {n: p.detach().clone() for n, p in trainer.model._backbone.named_parameters()
                              if "expert_0" in n}

    strat = RoutedExperts(R.LoRAExperts(rank=4), R.SPrompts(k=2))
    fit(strat, model=model, callbacks=[Snap()])
    for n, v in snaps[0].items():                       # expert 0 never changes after task 0
        assert torch.equal(v, snaps[3][n]), n
    strat.experts.use_none(model)
    with torch.no_grad():
        assert torch.allclose(model.encode(x), q0, atol=1e-6)     # query = untouched PTM


def test_encode_with_matches_single_expert():
    torch.manual_seed(0)
    tr, _ = fit(RoutedExperts(R.LoRAExperts(rank=4), R.SPrompts(k=2)), epochs=1)
    m, ex = tr.model, tr.strategy.experts
    x = torch.randn(10, 32)
    ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    with torch.no_grad():
        mixed = ex.encode_with(m, x, ids)
        for e in range(4):
            ex._activate(m, e)
            assert torch.allclose(mixed[ids == e], m.encode(x[ids == e]), atol=1e-5)


def test_keyquery_keys_trained_only_for_own_task():
    tr, _ = fit(RoutedExperts(R.LoRAExperts(rank=4), R.KeyQuery()))
    keys = torch.stack(list(tr.strategy.router._keys))
    assert keys.shape[0] == 4 and not any(k.requires_grad for k in tr.strategy.router._keys[:3])


def test_unknown_names():
    with pytest.raises(KeyError):
        Routed("nope", "hide")
    with pytest.raises(KeyError):
        Routed("lora", "nope")


# ------------------------------------------------------------------ ViT experts (adaptformer, prompts)
pytest.importorskip("timm")


def tiny_vit():
    torch.manual_seed(0)
    return clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, freeze_backbone=True,
                               device="cpu", img_size=32)


def image_bench(num_tasks=2, n_classes=4):
    from torch.utils.data import TensorDataset
    from cl_toolkit.benchmarks import split_by_class
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n_classes).repeat_interleave(8)
    x = torch.randn(len(y), 3, 32, 32, generator=g) + y.view(-1, 1, 1, 1).float()
    return split_by_class(TensorDataset(x, y), TensorDataset(x, y), num_tasks, shuffle_classes=False)


@pytest.mark.parametrize("experts", ["adaptformer", "prompt"])
def test_vit_experts_backbone_frozen_and_query_stable(experts):
    model = tiny_vit()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        q0 = model.encode(x).clone()
    bb0 = {n: p.clone() for n, p in model._backbone.named_parameters()}
    strat = Routed(experts, "sprompts", router_kwargs={"k": 2})
    tr, r = fit(strat, model=model, bench=image_bench(), epochs=2)
    assert "final_routing_accuracy" in r
    for n, p in model._backbone.named_parameters():
        if n in bb0 and "ctk_adapter" not in n:
            assert torch.equal(p, bb0[n]), n
    strat.experts.use_none(model)
    with torch.no_grad():
        assert torch.allclose(model.encode(x), q0, atol=1e-5)


def test_prompt_experts_single_forward_matches_grouped():
    tr, _ = fit(Routed("prompt", "sprompts", router_kwargs={"k": 2}), model=tiny_vit(), bench=image_bench(),
                epochs=1)
    m, ex = tr.model, tr.strategy.experts
    x = torch.randn(6, 3, 32, 32)
    ids = torch.tensor([0, 1, 0, 1, 1, 0])
    with torch.no_grad():
        once = ex.encode_with(m, x, ids)
        for e in (0, 1):
            ex._tp.ids = torch.full((int((ids == e).sum()),), e)
            assert torch.allclose(once[ids == e], m.encode(x[ids == e]), atol=1e-4)
        ex._tp.ids = None


def test_registry_presets():
    from cl_toolkit.strategies import get_strategy
    for name in ("hide-prompt", "s-prompts", "expert-gate"):
        assert isinstance(get_strategy(name), RoutedExperts)


def test_lora_experts_only_current_expert_trainable():
    """peft's enable_adapter_layers() re-marks every active adapter trainable; old experts must stay frozen."""
    torch.manual_seed(0)
    model = MLP()
    scen = Scenario(gauss_benchmark(num_tasks=3), "class", batch_size=32, num_workers=0)
    strat = Routed("lora", "lda", experts_kwargs={"rank": 4})
    seen = []
    tr = ContinualTrainer(model, scen, strat, epochs_per_task=1, verbose=False)
    orig = strat.on_task_start
    def spy(task_id, loader, m):
        orig(task_id, loader, m)
        seen.append({n.split(".lora_")[1].split(".")[1] for n, p in m._backbone.named_parameters()
                     if "lora_" in n and p.requires_grad})
    strat.on_task_start = spy
    tr.fit()
    assert seen == [{"expert_0"}, {"expert_1"}, {"expert_2"}]

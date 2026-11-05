"""Benchmarks, metrics, merging, adapters, prompts, API."""

import numpy as np
import pytest
import torch

from conftest import MLP, gauss_benchmark, gauss_data
import cl_toolkit as clt
from cl_toolkit import adapters, merging
from cl_toolkit.benchmarks import few_shot, get_targets, joint, split_by_class
from cl_toolkit.trainers import compute_cl_metrics


# --------------------------------------------------------------------- data
def test_split_by_class_partitions():
    b = gauss_benchmark(num_tasks=4, n_classes=20)
    seen = [c for s in b.get_splits() for c in s.classes]
    assert sorted(seen) == list(range(20)) and b.num_tasks == 4 and b.total_classes == 20
    for s in b.get_splits():
        assert set(get_targets(s.train_dataset)) == set(s.classes)


def test_split_initial_classes_and_order():
    tr, te = gauss_data(20)
    b = split_by_class(tr, te, num_tasks=4, initial_classes=11)
    assert [len(s.classes) for s in b.get_splits()] == [11, 3, 3, 3]
    b = split_by_class(tr, te, 2, class_order=list(range(19, -1, -1)))
    assert b.get_splits()[0].classes[0] == 19


def test_few_shot_and_joint():
    b = gauss_benchmark()
    fs = few_shot(b, 3, skip_first=True)
    assert len(fs.get_splits()[0].train_dataset) == len(b.get_splits()[0].train_dataset)
    assert len(fs.get_splits()[1].train_dataset) == 3 * 5
    j = joint(b)
    assert j.num_tasks == 1 and len(j.get_splits()[0].classes) == 20


def test_task_il_labels_are_local():
    scen = clt.Scenario(gauss_benchmark(), "task", num_workers=0)
    for t in scen.tasks:
        ys = torch.cat([y for _, y in t.test_loader])
        assert ys.min() == 0 and ys.max() == 4


# --------------------------------------------------------------------- metrics
def test_metrics():
    acc = np.array([[90, np.nan, np.nan], [70, 80, np.nan], [60, 70, 90.]])
    m = compute_cl_metrics(acc)
    assert m["avg_accuracy"] == pytest.approx(220 / 3)
    assert m["avg_forgetting"] == pytest.approx(((90 - 60) + (80 - 70)) / 2)
    assert m["backward_transfer"] == pytest.approx(((60 - 90) + (70 - 80)) / 2)
    assert m["avg_incremental_accuracy"] == pytest.approx((90 + 75 + 220 / 3) / 3)


# --------------------------------------------------------------------- merging
def test_merging_methods():
    a = {"w": torch.tensor([1.0, -2.0, 0.5])}
    b = {"w": torch.tensor([3.0, 1.0, -0.1])}
    assert torch.allclose(merging.average([a, b])["w"], torch.tensor([2.0, -0.5, 0.2]))
    assert torch.allclose(merging.task_arithmetic([a, b], lam=1)["w"], torch.tensor([4.0, -1.0, 0.4]))
    assert torch.allclose(merging.magmax([a, b], lam=1.0)["w"], torch.tensor([3.0, -2.0, 0.5]))
    t = merging.ties([a, b], density=1.0)["w"]           # sign election: +, -, +
    assert torch.allclose(t, torch.tensor([2.0, -2.0, 0.5]))
    assert torch.allclose(merging.ties([a, a])["w"], merging.ties([a])["w"])
    d = merging.dare([a, b], p=0.0, base=merging.ties, density=1.0)     # p=0: DARE-TIES == TIES
    assert torch.allclose(d["w"], t)
    assert torch.allclose(merging.dare([a, b], p=0.0)["w"], merging.task_arithmetic([a, b])["w"])
    tv = merging.task_vector({"w": torch.ones(3)}, {"w": torch.zeros(3)})
    assert merging.similarity(tv, tv) == pytest.approx(1.0)


# --------------------------------------------------------------------- adapters
def test_lora_add_activate_merge_preserves_function():
    torch.manual_seed(0)
    m = MLP()
    clt.Scenario(gauss_benchmark(), "class").setup(m)
    x = torch.randn(8, 32)
    adapters.add_lora(m, "a", rank=2)
    with torch.no_grad():
        for n, p in m._backbone.named_parameters():
            if "lora_B" in n:
                p.normal_()
    adapters.add_lora(m, "b", rank=2)
    trainable = {n for n, p in m._backbone.named_parameters() if p.requires_grad}
    assert trainable and all(".b." in n for n in trainable)    # only the new adapter
    before = m(x).detach()
    adapters.merge_lora(m, "a")
    assert adapters.lora_names(m) == ["b"]
    assert torch.allclose(m(x), before, atol=1e-5)


# --------------------------------------------------------------------- ViT: prompts & AdaptFormer
timm = pytest.importorskip("timm")


def tiny_vit():
    torch.manual_seed(0)
    return clt.models.ViTModel("vit_tiny_patch16_224", pretrained=False, freeze_backbone=True,
                               device="cpu", img_size=32)


def image_bench(num_tasks=2, n_classes=4):
    from torch.utils.data import TensorDataset
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n_classes).repeat_interleave(8)
    x = torch.randn(len(y), 3, 32, 32, generator=g) + y.view(-1, 1, 1, 1).float()
    return split_by_class(TensorDataset(x, y), TensorDataset(x, y), num_tasks, shuffle_classes=False)


def test_adaptformer_zero_init_is_identity():
    m = tiny_vit()
    x = torch.randn(2, 3, 32, 32)
    before = m.encode(x)
    params = adapters.add_adaptformer(m, bottleneck=8)
    assert params and all(p.requires_grad for p in params)
    assert torch.allclose(m.encode(x), before, atol=1e-5)


@pytest.mark.parametrize("name", ["l2p", "dualprompt", "coda-prompt"])
def test_prompt_methods_train_only_prompts_and_head(name):
    kw = {"coda-prompt": {"pool_size": 8}, "l2p": {}, "dualprompt": {}}[name]
    tr = clt.build(image_bench(), tiny_vit(), "class", name, strategy_kwargs=kw, epochs=1, batch_size=8,
                   num_workers=0, device="cpu", verbose=False)
    backbone0 = {n: p.clone() for n, p in tr.model._backbone.named_parameters()}
    r = tr.fit()
    assert np.isfinite(r["avg_accuracy"])
    for n, p in tr.model._backbone.named_parameters():
        assert torch.equal(p, backbone0[n]), f"backbone param {n} changed"
    assert any(p.requires_grad is False for p in tr.model._prompt.parameters()) or name == "l2p"


def test_dualprompt_freezes_old_task_prompts():
    snaps = {}

    class Snap(clt.callbacks.Callback):
        def on_task_end(self, trainer, task_id, task_metrics, **kw):
            snaps[task_id] = trainer.model._prompt.e_prompts[0].detach().clone()

    tr = clt.build(image_bench(num_tasks=2), tiny_vit(), "class", "dualprompt", epochs=2, batch_size=8,
                   num_workers=0, device="cpu", verbose=False, callbacks=[Snap()])
    tr.fit()
    assert torch.equal(snaps[0], snaps[1])
    assert len(tr.model._prompt.e_prompts) == 2


# --------------------------------------------------------------------- API
def test_run_with_custom_pieces(tmp_path):
    tr, te = gauss_data(10)
    out = tmp_path / "r.json"
    r = clt.run((tr, te), model=MLP(), scenario="class", strategy="er", num_tasks=2, epochs=2,
                num_workers=0, device="cpu", verbose=False, save=str(out), out_dir=None)
    assert out.exists() and r["avg_accuracy"] > 10
    with pytest.raises(KeyError):
        clt.build((tr, te), model=MLP(), strategy="nope", num_tasks=2, num_workers=0, device="cpu")

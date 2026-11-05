"""Classifier heads: correctness of the statistics and behaviour inside the CL loop."""

import numpy as np
import pytest
import torch

from conftest import MLP, gauss_benchmark, gauss_data, gauss_domains
import cl_toolkit as clt
from cl_toolkit import ContinualTrainer, Scenario, heads as H
from cl_toolkit.models.base import CosineLinear, IncrementalClassifier
from cl_toolkit.strategies import ExperienceReplay, FirstSessionAdaptation, NaiveStrategy, WithHead

STAT_HEADS = {
    "ncm": lambda: H.NCM(),
    "fecam": lambda: H.FeCAM(),
    "lda": lambda: H.LDA(),
    "klda": lambda: H.KLDA(rff_dim=512),
    "ranpac": lambda: H.RanPAC(proj_dim=256),
}


class _Scen:   # minimal scenario stand-in for unit tests of the statistics
    class benchmark:
        total_classes = 5


def fit(strategy, kind="class", epochs=3, lr=3e-3):
    torch.manual_seed(0)
    bench = gauss_domains() if kind == "domain" else gauss_benchmark()
    scen = Scenario(bench, kind, batch_size=32, num_workers=0)
    return ContinualTrainer(MLP(), scen, strategy, lr=lr, epochs_per_task=epochs, verbose=False).fit()


# ------------------------------------------------------------------ in the loop
@pytest.mark.parametrize("kind", ["class", "task", "domain"])
@pytest.mark.parametrize("name", list(STAT_HEADS))
def test_statistical_heads_on_frozen_backbone(name, kind):
    r = fit(WithHead(FirstSessionAdaptation(), STAT_HEADS[name]()), kind, epochs=0)
    chance = 100 / (20 if kind == "class" else 5)
    assert r["avg_accuracy"] > 2 * chance, (name, kind, r["avg_accuracy"])
    assert r["avg_forgetting"] < 15, (name, kind, r["avg_forgetting"])   # frozen features: little forgetting


def test_heads_compose_with_replay():
    r = fit(WithHead(ExperienceReplay(200), H.FeCAM()), "class")
    assert np.isfinite(r["avg_accuracy"]) and r["avg_accuracy"] > 20


def test_slca_reduces_recency_bias():
    plain = fit(NaiveStrategy(), "class")
    slca = fit(WithHead(NaiveStrategy(), H.SLCA(epochs=5)), "class")
    assert slca["avg_accuracy"] > plain["avg_accuracy"] + 10


def test_wa_equalises_weight_norms():
    torch.manual_seed(0)
    scen = Scenario(gauss_benchmark(), "class", batch_size=32, num_workers=0)
    wa = H.WA()
    tr = ContinualTrainer(MLP(), scen, WithHead(NaiveStrategy(), wa), lr=3e-3, epochs_per_task=3, verbose=False)
    tr.fit()
    norms = tr.model.get_head()._fc.weight.detach().norm(dim=1)
    last = scen.get_task(3).classes
    old = scen.seen_classes(2)
    assert norms[last].mean().item() == pytest.approx(norms[old].mean().item(), rel=1e-4)


def test_cosine_classifier_layer():
    r = clt.run(gauss_data(10), model=MLP(), strategy="er", classifier="cosine", num_tasks=2, epochs=2,
                num_workers=0, device="cpu", verbose=False, out_dir=None)
    assert r["avg_accuracy"] > 20
    inc = IncrementalClassifier(8, 3, layer="cosine")
    inc._fc.scale.data.fill_(7.0)
    w = inc._fc.weight.detach().clone()
    inc.grow(2)
    assert isinstance(inc._fc, CosineLinear) and inc.num_classes == 5
    assert torch.equal(inc._fc.weight[:3], w) and inc._fc.scale.item() == 7.0


def test_api_classifier_argument():
    tr = clt.build(gauss_data(10), model=MLP(), strategy="naive", classifier="fecam", num_tasks=2,
                   num_workers=0, device="cpu", verbose=False)
    assert isinstance(tr.strategy, WithHead) and isinstance(tr.strategy.head, H.FeCAM)
    with pytest.raises(KeyError):
        clt.build(gauss_data(10), model=MLP(), classifier="nope", num_tasks=2, num_workers=0, device="cpu")


# ------------------------------------------------------------------ statistics
def _bound(head, C=5):
    s = _Scen()
    s.benchmark.total_classes = C
    head.bind(None, s)
    return head


def test_ncm_means_exact():
    x, y = torch.randn(50, 4), torch.arange(5).repeat(10)
    h = _bound(H.NCM())
    h.update(x[:20], y[:20], 0)
    h.update(x[20:], y[20:], 1)
    for c in range(5):
        assert torch.allclose(h.means[c], x[y == c].mean(0), atol=1e-6)


def test_lda_is_streaming_exact():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(300, 6, generator=g), torch.arange(3).repeat(100)
    x = x + y.unsqueeze(1).float()
    a, b = _bound(H.LDA(shrinkage=0.0), 3), _bound(H.LDA(shrinkage=0.0), 3)
    a.update(x, y, 0)
    for i in range(0, 300, 70):               # chunks mixing classes, like Domain-IL
        b.update(x[i:i + 70], y[i:i + 70], 0)
    q = torch.randn(10, 6)
    assert torch.allclose(a.logits(q), b.logits(q), atol=1e-3)
    # reference: pooled within-class covariance
    mu = torch.stack([x[y == c].mean(0) for c in range(3)]).double()
    S = sum(((x[y == c].double() - mu[c]).T @ (x[y == c].double() - mu[c])) for c in range(3)) / 300
    P = torch.linalg.inv(S)
    ref = q.double() @ P @ mu.T - 0.5 * ((mu @ P) * mu).sum(1)
    assert torch.allclose(a.logits(q).double(), ref, atol=1e-3)


def test_fecam_tukey_only_for_nonnegative_features():
    h = _bound(H.FeCAM())
    h.update(torch.randn(40, 4), torch.arange(5).repeat(8), 0)
    assert h.tukey is False
    h2 = _bound(H.FeCAM())
    h2.update(torch.rand(40, 4), torch.arange(5).repeat(8), 0)
    assert h2.tukey is True
    assert torch.isfinite(h2.logits(torch.rand(3, 4))).all()


def test_unseen_classes_are_never_predicted():
    for make in STAT_HEADS.values():
        h = _bound(make(), C=5)
        h.update(torch.randn(30, 4), torch.tensor([0, 1, 2]).repeat(10), 0)
        s = h.logits(torch.randn(8, 4))
        assert torch.isinf(s[:, 3:]).all() and set(s.argmax(1).tolist()) <= {0, 1, 2}, type(h).__name__


def test_head_retraining_unaffected_by_mixed_precision():
    """Regression: on_task_end ran inside one long autocast region, so SLCA/GDumb head
    retraining used stale cached bf16 weights and corrupted the head (CIFAR-100: 97.8% -> 23%)."""
    from cl_toolkit.strategies import GDumb

    def run(strategy, precision):
        torch.manual_seed(0)
        scen = Scenario(gauss_benchmark(), "class", batch_size=32, num_workers=0)
        return ContinualTrainer(MLP(), scen, strategy, lr=3e-3, epochs_per_task=3, verbose=False,
                                precision=precision).fit()["avg_accuracy"]

    for make in (lambda: WithHead(NaiveStrategy(), H.SLCA(epochs=5)), lambda: GDumb(200, epochs=5)):
        fp32, bf16 = run(make(), "fp32"), run(make(), "bf16")
        assert abs(fp32 - bf16) < 8, (fp32, bf16)


def test_on_task_end_not_in_autocast():
    seen = []

    class Spy(NaiveStrategy):
        def on_task_end(self, task_id, train_loader, model):
            seen.append(torch.is_autocast_enabled("cpu"))

    scen = Scenario(gauss_benchmark(num_tasks=2), "class", batch_size=32, num_workers=0)
    ContinualTrainer(MLP(), scen, Spy(), epochs_per_task=1, verbose=False, precision="bf16").fit()
    assert seen == [False, False]

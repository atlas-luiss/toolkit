"""Merging methods: each checked against a hand computation or a property its paper relies on."""

import pytest
import torch

from cl_toolkit import merging as M
from cl_toolkit import strategies as S


def tv(**kw):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items()}


# ---------------------------------------------------------------------------- element-wise
def test_task_arithmetic_and_average():
    a, b = tv(w=[1.0, 2.0]), tv(w=[3.0, -2.0])
    assert torch.allclose(M.task_arithmetic([a, b], lam=0.5)["w"], torch.tensor([2.0, 0.0]))
    assert torch.allclose(M.average([a, b])["w"], torch.tensor([2.0, 0.0]))


def test_magmax_default_lambda_is_papers():
    a, b = tv(w=[1.0, -2.0, 0.5]), tv(w=[-1.5, -1.0, 0.4])
    assert torch.allclose(M.magmax([a, b])["w"], 0.5 * torch.tensor([-1.5, -2.0, 0.5]))


def test_ties_trims_over_the_whole_task_vector():
    # task vector with two tensors; global top-50% keeps the 2 largest of 4 entries overall
    a = tv(x=[10.0, 9.0], y=[0.1, 0.2])
    out = M.ties([a], density=0.5, lam=1.0)
    assert torch.equal(out["x"], torch.tensor([10.0, 9.0])) and torch.equal(out["y"], torch.zeros(2))
    per_tensor = M.ties([a], density=0.5, lam=1.0, scope="tensor")         # keeps one of each tensor
    assert torch.equal(per_tensor["y"], torch.tensor([0.0, 0.2]))


def test_ties_sign_election_and_disjoint_mean():
    a, b = tv(w=[1.0, -2.0, 0.5, 3.0]), tv(w=[-1.5, -1.0, 0.4, 2.0])
    t = M.ties([a, b], density=1.0)["w"]
    # elected signs (sum): -, -, +, + -> mean over agreeing entries
    assert torch.allclose(t, torch.tensor([-1.5, -1.5, 0.45, 2.5]))


def test_ties_zero_sign_goes_to_majority():
    a, b = tv(w=[1.0, 2.0, 3.0]), tv(w=[-1.0, 1.0, 1.0])     # first coordinate sums to 0
    t = M.ties([a, b], density=1.0)["w"]
    assert t[0] == pytest.approx(1.0)                         # majority sign is +


def test_dare_is_unbiased_and_drops_p():
    a = {"w": torch.ones(20000)}
    out = M.dare([a], p=0.9, base=M.average, seed=0)["w"]
    assert abs(float((out == 0).float().mean()) - 0.9) < 0.01
    assert abs(float(out.mean()) - 1.0) < 0.05                 # E[rescaled] = original


def test_breadcrumbs_drops_both_tails_per_layer():
    w = torch.arange(1, 1001, dtype=torch.float32)             # magnitudes 1..1000
    out = M.breadcrumbs([{"w": w}], beta=0.9, gamma=0.99, lam=1.0)["w"]
    kept = out.nonzero().flatten() + 1
    assert kept.min() == 901 and kept.max() == 990             # bottom 90% and top 1% removed


def test_consensus_keeps_weights_selected_by_k_tasks():
    a, b, c = tv(w=[1.0, 1.0, 0.0]), tv(w=[1.0, 0.0, 0.0]), tv(w=[1.0, 0.0, 5.0])
    masks = M.tall_masks([a, b, c], lam=0.4)
    # coordinate 0: every task contributes equally -> all masks on; 1 and 2: one task "selfish"
    assert [bool(m["w"][0]) for m in masks] == [True, True, True]
    out = M.consensus([a, b, c], k=2, lam=1.0)["w"]
    assert torch.allclose(out, torch.tensor([3.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------- SVD-based
def _orthogonal_tasks(T=3, d=12, rank=4, seed=0):
    """T task matrices of rank r/T living in mutually orthogonal subspaces (no interference)."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    tvs = []
    for t in range(T):
        U, V = Q[:, t * rank:(t + 1) * rank], R[:, t * rank:(t + 1) * rank]
        S = torch.rand(rank, generator=g) + 1
        tvs.append({"w": (U * S) @ V.T, "b": torch.randn(d, generator=g)})
    return tvs


def test_tsv_is_exact_without_interference():
    tvs = _orthogonal_tasks()
    out = M.tsv(tvs, alpha=1.0)
    assert torch.allclose(out["w"], sum(t["w"] for t in tvs), atol=1e-4)    # disjoint subspaces kept exactly
    assert torch.allclose(out["b"], torch.stack([t["b"] for t in tvs]).mean(0))   # 1-D: averaged


def test_tsv_keeps_r_over_T_components_per_task():
    g = torch.Generator().manual_seed(0)
    tvs = [{"w": torch.randn(12, 12, generator=g)} for _ in range(3)]
    assert torch.linalg.matrix_rank(M.tsv(tvs)["w"]) <= 12               # 3 tasks x 4 components


def test_iso_c_has_flat_spectrum_equal_to_mean():
    g = torch.Generator().manual_seed(0)
    tvs = [{"w": torch.randn(10, 6, generator=g)} for _ in range(3)]
    S_sum = torch.linalg.svdvals(sum(t["w"] for t in tvs))
    S_out = torch.linalg.svdvals(M.iso_c(tvs, alpha=1.0)["w"])
    assert torch.allclose(S_out, torch.full_like(S_out, float(S_sum.mean())), atol=1e-4)


def test_iso_cts_with_full_common_subspace_is_iso_c():
    g = torch.Generator().manual_seed(0)
    tvs = [{"w": torch.randn(10, 8, generator=g)} for _ in range(4)]
    assert torch.allclose(M.iso_cts(tvs, common_fraction=1.0)["w"], M.iso_c(tvs)["w"], atol=1e-4)


def test_iso_cts_spectrum_is_isotropic():
    g = torch.Generator().manual_seed(0)
    tvs = [{"w": torch.randn(16, 16, generator=g)} for _ in range(4)]
    S = torch.linalg.svdvals(M.iso_cts(tvs, common_fraction=0.5)["w"])
    assert torch.allclose(S, S[0].expand_as(S), atol=1e-3)


def test_unknown_method_is_an_error():
    with pytest.raises(KeyError):
        M.merge([tv(w=[1.0])], "fisher")


# ---------------------------------------------------------------------------- CoMA / CoFiMA
class _Param:
    """Tiny stand-in model with one backbone parameter."""

    def __init__(self, value):
        import torch.nn as nn
        self._backbone = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self._backbone.weight.fill_(value)


def _end_task(strat, model, value, t, fisher=None):
    with torch.no_grad():
        model._backbone.weight.fill_(value)
    if fisher is not None:
        strat._fisher = lambda loader, m, task_id: {"weight": torch.full((1, 1), fisher.pop(0))}
    strat.on_task_end(t, None, model)
    return float(model._backbone.weight)


def test_coma_recursion_by_hand():
    s, m = S.ModelAveraging(S.NaiveStrategy(), lam=0.4), _Param(0.0)
    assert _end_task(s, m, 1.0, 0) == pytest.approx(1.0)                    # first task: theta*_0 = theta_0
    assert _end_task(s, m, 2.0, 1) == pytest.approx(0.4 * 2 + 0.6 * 1)      # 1.4
    assert _end_task(s, m, 0.0, 2) == pytest.approx(0.6 * 1.4)


def test_cofima_weights_by_fisher():
    s, m = S.ModelAveraging(S.NaiveStrategy(), lam=0.4, fisher=True), _Param(0.0)
    fishers = [3.0, 1.0, 2.0]                     # F*_0, then F_1 (theta_1), then F*_1 (averaged)
    _end_task(s, m, 1.0, 0, fishers)
    got = _end_task(s, m, 2.0, 1, fishers)
    w = 0.4 * 1.0 / (0.4 * 1.0 + 0.6 * 3.0)
    assert got == pytest.approx(w * 2.0 + (1 - w) * 1.0)


def test_clip_zeroshot_head_is_frozen_text_embeddings():
    import cl_toolkit as clt
    from conftest import MLP, gauss_data
    from cl_toolkit.benchmarks import split_by_class

    class FakeCLIP(MLP):
        def text_classifier(self, names, templates=None):
            self.names = names
            g = torch.Generator().manual_seed(0)
            return torch.nn.functional.normalize(torch.randn(len(names), self.embed_dim, generator=g), dim=1)

    tr_ds, te_ds = gauss_data(4, n_train=20, n_test=10)
    bench = split_by_class(tr_ds, te_ds, 2, shuffle_classes=False,
                           class_names={0: "cat", 1: "dog", 2: "maple_tree", 3: "car"})
    model = FakeCLIP()
    tr = clt.build(bench, model, "class", "naive", classifier="clip-zeroshot", epochs=1, num_workers=0,
                   device="cpu", verbose=False, freeze_backbone=False)
    tr.fit()
    fc = model.get_head()._fc
    assert model.names == ["cat", "dog", "maple tree", "car"]
    assert not fc.weight.requires_grad and float(fc.scale) == 100.0
    assert torch.allclose(fc.weight, model.text_classifier(model.names))          # never trained


def test_dare_masks_do_not_depend_on_key_order():
    g = torch.Generator().manual_seed(0)
    a = {"x": torch.randn(50, generator=g), "y": torch.randn(50, generator=g)}
    b = {"y": a["y"], "x": a["x"]}                                 # same task vector, other order
    oa, ob = M.dare([a], p=0.5, base=M.average), M.dare([b], p=0.5, base=M.average)
    assert all(torch.equal(oa[k], ob[k]) for k in a)

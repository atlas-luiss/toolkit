"""Every strategy x every scenario it claims to support must run and learn."""

import numpy as np
import pytest
import torch

from conftest import MLP, gauss_benchmark, gauss_domains
from cl_toolkit import ContinualTrainer, Scenario
from cl_toolkit import strategies as S

FACTORIES = {
    "naive": lambda: S.NaiveStrategy(),
    "ewc": lambda: S.EWC(lam=100),
    "ewc_online": lambda: S.EWCOnline(lam=100),
    "mas": lambda: S.MAS(lam=0.1),
    "lwf": lambda: S.LwF(),
    "er": lambda: S.ExperienceReplay(200),
    "der": lambda: S.DER(200),
    "derpp": lambda: S.DERPP(200),
    "agem": lambda: S.AGEM(200),
    "icarl": lambda: S.iCaRL(200),
    "gdumb": lambda: S.GDumb(200, epochs=5),
    "lora": lambda: S.LoRAStrategy(rank=4),
    "lora_merge": lambda: S.LoRAStrategy(rank=4, merge_after_task=True),
    "lora_er": lambda: S.LoRAReplay(200, rank=4),
    "simplecil": lambda: S.SimpleCIL(),
    "ranpac": lambda: S.RanPAC(proj_dim=256),
    "magmax": lambda: S.MergeAfterTask(S.NaiveStrategy(), "magmax"),
    "ties": lambda: S.MergeAfterTask(S.NaiveStrategy(), "ties", density=0.5),
}
KINDS = ["class", "task", "domain"]


def fit(strategy, kind, epochs=5, lr=3e-3, **kw):
    torch.manual_seed(0)
    bench = gauss_domains() if kind == "domain" else gauss_benchmark()
    scen = Scenario(bench, kind, batch_size=32, num_workers=0)
    return ContinualTrainer(MLP(), scen, strategy, lr=lr, epochs_per_task=epochs, verbose=False, **kw).fit()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(FACTORIES))
def test_runs(name, kind):
    strat = FACTORIES[name]()
    if kind not in strat.scenarios:
        with pytest.raises(ValueError, match="supports scenarios"):
            fit(strat, kind)
        return
    r = fit(strat, kind)
    acc = r["accuracy_matrix"]
    assert np.isfinite(acc[np.tril_indices(len(acc))]).all()
    assert 0 <= r["avg_accuracy"] <= 100
    chance = 100 / (5 if kind != "class" else 20)
    assert r["avg_accuracy"] > chance, f"{name}/{kind} did not learn: {r['avg_accuracy']:.1f}"


def test_replay_beats_naive_class_il():
    naive = fit(S.NaiveStrategy(), "class")
    er = fit(S.ExperienceReplay(400), "class")
    assert er["avg_accuracy"] > naive["avg_accuracy"] + 10
    assert er["avg_forgetting"] < naive["avg_forgetting"]


def test_replay_does_not_hurt_task_il():
    """Regression: replayed samples used to go through the *current* task's head."""
    naive = fit(S.NaiveStrategy(), "task")
    er = fit(S.ExperienceReplay(400), "task")
    assert er["avg_forgetting"] <= naive["avg_forgetting"] + 1.0


def test_closed_form_methods_do_not_forget():
    for strat in (S.SimpleCIL(), S.RanPAC(proj_dim=512)):
        r = fit(strat, "class", epochs=1)
        assert r["avg_forgetting"] < 15, (strat.name, r["avg_forgetting"])


class _Spy(S.NaiveStrategy):
    def __init__(self):
        super().__init__()
        self.eval_ids = []

    def before_eval(self, model, task_id):
        self.eval_ids.append(task_id)


@pytest.mark.parametrize("kind,expect_ids", [("class", False), ("domain", False), ("task", True)])
def test_task_id_only_visible_in_task_il(kind, expect_ids):
    spy = _Spy()
    fit(spy, kind, epochs=1)
    if expect_ids:
        assert all(t is not None for t in spy.eval_ids)
    else:
        assert all(t is None for t in spy.eval_ids)


def test_lora_class_il_has_no_task_oracle():
    """Evaluating task j must not depend on which test set it is: all adapters active."""
    from cl_toolkit import adapters
    torch.manual_seed(0)
    model = MLP()
    scen = Scenario(gauss_benchmark(), "class", batch_size=32, num_workers=0)
    tr = ContinualTrainer(model, scen, S.LoRAStrategy(rank=4), lr=3e-3, epochs_per_task=3, verbose=False)
    r = tr.fit()
    assert sorted(adapters.lora_names(model)) == [f"task_{i}" for i in range(4)]
    x = torch.cat([t.test_loader.dataset[i][0][None] for t in scen.tasks for i in range(3)])
    model.eval()
    with torch.no_grad():
        out_a = model(x, None)
        tr.strategy.before_eval(model, None)
        out_b = model(x, None)
    assert torch.allclose(out_a, out_b)
    assert r["avg_accuracy"] < 95  # sanity: plain LoRA in class-IL should forget something


def test_train_accuracy_reported_without_second_forward():
    calls = {"n": 0}
    model = MLP()
    orig = model.forward

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    model.forward = counting
    scen = Scenario(gauss_benchmark(num_tasks=2), "class", batch_size=40, num_workers=0)
    ContinualTrainer(model, scen, S.NaiveStrategy(), epochs_per_task=1, verbose=False,
                     eval_every_task=False).fit()
    n_train_batches = sum(len(t.train_loader) for t in scen.tasks)
    n_eval_batches = sum(len(t.test_loader) for t in scen.tasks)
    assert calls["n"] == n_train_batches + n_eval_batches

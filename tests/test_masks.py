"""Class-IL logit masking: what the model may predict, during training and evaluation."""

import pytest
import torch

from conftest import MLP, gauss_benchmark
from cl_toolkit import ContinualTrainer, Scenario
from cl_toolkit.callbacks import Callback
from cl_toolkit.strategies import NaiveStrategy


class RecordPreds(Callback):
    """Collect argmax predictions on every test set after each task."""

    def __init__(self):
        self.preds = {}

    def on_task_end(self, trainer, task_id, task_metrics, **kw):
        trainer._set_eval_mask(task_id)
        out = []
        with torch.no_grad():
            for t in trainer.scenario.tasks[: task_id + 1]:
                for x, _ in t.test_loader:
                    out.append(trainer.model(x, None).argmax(1))
        self.preds[task_id] = torch.cat(out)


def run(train_mask, eval_mask, kind="class"):
    torch.manual_seed(0)
    rec = RecordPreds()
    scen = Scenario(gauss_benchmark(), kind, batch_size=32, num_workers=0)
    tr = ContinualTrainer(MLP(), scen, NaiveStrategy(train_mask, eval_mask), lr=3e-3, epochs_per_task=3,
                          verbose=False, callbacks=[rec] if kind == "class" else [])
    return tr, tr.fit(), rec


def test_seen_mask_never_predicts_unseen_classes():
    tr, _, rec = run("seen", "seen")
    for t, p in rec.preds.items():
        assert set(p.tolist()) <= set(tr.scenario.seen_classes(t)), f"after task {t}"


def test_eval_mask_none_exposes_all_outputs():
    tr, _, _ = run("seen", "none")
    tr._set_eval_mask(0)
    assert tr.model._class_mask is None
    x = tr.scenario.tasks[0].test_loader.dataset[0][0][None]
    assert tr.model(x, None).shape[1] == 20 and torch.isfinite(tr.model(x, None)).all()


def test_current_mask_trains_only_current_logits():
    seen = {}

    class Grab(Callback):
        def on_epoch_begin(self, trainer, task_id, epoch, total_epochs, **kw):
            m = trainer.model._class_mask
            seen[task_id] = set(torch.nonzero(m).flatten().tolist())

    torch.manual_seed(0)
    scen = Scenario(gauss_benchmark(), "class", batch_size=32, num_workers=0)
    ContinualTrainer(MLP(), scen, NaiveStrategy("current"), epochs_per_task=1, verbose=False,
                     callbacks=[Grab()]).fit()
    for t, cls in seen.items():
        assert cls == set(scen.get_task(t).classes)


def test_masks_ignored_in_task_il():
    tr, r, _ = run("current", "none", kind="task")
    assert tr.model._class_mask is None and r["avg_accuracy"] > 20


def test_bad_mask_name_rejected():
    with pytest.raises(ValueError, match="train_mask"):
        run("everything", "seen")


def test_current_mask_reduces_forgetting():
    """Known effect: masking training logits to the current task removes most of the
    new-class bias of naive fine-tuning in Class-IL."""
    _, none, _ = run("none", "none")
    _, cur, _ = run("current", "seen")
    assert cur["avg_forgetting"] < none["avg_forgetting"]


@pytest.mark.parametrize("make", [
    lambda: __import__("cl_toolkit").strategies.ExperienceReplay(400),
    lambda: __import__("cl_toolkit").strategies.DERPP(400),
    lambda: __import__("cl_toolkit").strategies.AGEM(400),
    lambda: __import__("cl_toolkit").strategies.LoRAReplay(400, rank=4),
], ids=["er", "derpp", "agem", "lora_er"])
def test_replay_with_current_mask_is_finite_and_remembers(make):
    """Regression: with train_mask='current', replayed labels got -inf logits -> inf loss -> NaN head."""
    torch.manual_seed(0)
    strat = make()
    strat.train_mask = "current"
    scen = Scenario(gauss_benchmark(), "class", batch_size=32, num_workers=0)
    r = ContinualTrainer(MLP(), scen, strat, lr=3e-3, epochs_per_task=5, verbose=False).fit()
    old_tasks = r["accuracy_matrix"][-1, :-1]
    assert old_tasks.mean() > 20, old_tasks        # was ~1% when broken


def test_non_finite_loss_stops_training():
    class Bad(NaiveStrategy):
        def compute_loss(self, batch, model, task_id):
            loss, logits = super().compute_loss(batch, model, task_id)
            return loss * float("inf"), logits

    scen = Scenario(gauss_benchmark(num_tasks=2), "class", batch_size=32, num_workers=0)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        ContinualTrainer(MLP(), scen, Bad(), epochs_per_task=1, verbose=False).fit()

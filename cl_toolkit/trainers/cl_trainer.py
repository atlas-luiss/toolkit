"""ContinualTrainer: the one training loop. Every step is an overridable method."""

from __future__ import annotations

import contextlib
import json
import time
from typing import Dict, List, Optional, Type

import numpy as np
import torch

from cl_toolkit import batch as B
import torch.nn.functional as F

from ..models.base import CLModel
from ..scenarios import Scenario, TaskInfo
from ..strategies.base import Strategy


TRAIN_MASKS = ("none", "seen", "current")
EVAL_MASKS = ("none", "seen")


class ContinualTrainer:
    """Runs a strategy over a scenario and records the accuracy matrix.

    ``fit()`` runs everything; ``start``, ``train_task``, ``evaluate`` and ``finish`` are its
    steps, to write the loop yourself. Override points (coarse → fine): ``fit``, ``run_task``,
    ``run_epoch``, ``train_step``, ``evaluate_task``, ``make_optimizer``.

    Args:
        precision: "bf16", "fp16" or "fp32". Default: bf16 on CUDA, fp32 on CPU.
        eval_every_task: evaluate on all seen tasks after each task (needed for
            the full accuracy matrix / forgetting). Set False to only evaluate at the end.
    """

    def __init__(
        self,
        model: CLModel,
        scenario: Scenario,
        strategy: Optional[Strategy] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs_per_task: int = 1,
        scheduler: Optional[str] = None,
        callbacks=None,
        device=None,
        precision: Optional[str] = None,
        eval_every_task: bool = True,
        verbose: bool = True,
    ):
        from ..callbacks import CallbackList
        from ..strategies import NaiveStrategy

        self.model, self.scenario = model, scenario
        self.strategy = strategy or NaiveStrategy()
        self.optimizer_cls, self.lr, self.weight_decay = optimizer_cls, lr, weight_decay
        self.epochs_per_task, self.scheduler_name = epochs_per_task, scheduler
        self.device = torch.device(device) if device is not None else model._device
        if hasattr(scenario, "use_device"):
            scenario.use_device(self.device)
        self.precision = precision or ("bf16" if self.device.type == "cuda" else "fp32")
        self.eval_every_task, self.verbose = eval_every_task, verbose
        self.callbacks = callbacks if isinstance(callbacks, CallbackList) else CallbackList(list(callbacks or []))
        self.stop_training = self.stop_epoch = False
        self.optimizer = self.scheduler = None
        self.results: Dict = {}

        if self.strategy.train_mask not in TRAIN_MASKS or self.strategy.eval_mask not in EVAL_MASKS:
            raise ValueError(f"train_mask must be in {TRAIN_MASKS}, eval_mask in {EVAL_MASKS}; got "
                             f"{self.strategy.train_mask!r}, {self.strategy.eval_mask!r}")
        if scenario.kind not in self.strategy.scenarios:
            raise ValueError(f"{self.strategy.name} supports scenarios {self.strategy.scenarios}, "
                             f"not {scenario.kind!r}")

    # ------------------------------------------------------------------ utils
    def autocast(self):
        """Mixed-precision context for *forward passes only*.

        Keep each region short (one forward / loss): autocast caches the low-precision
        copy of every weight until the outermost region exits, so updating parameters
        inside one region makes later forwards use stale weights.
        """
        if self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ main loop
    def fit(self) -> dict:
        """The whole loop. Written out, it is::

            trainer.start()
            for task in scenario:
                trainer.train_task(task)
                trainer.evaluate(task.task_id)
            return trainer.finish()

        Write that loop yourself to run your own code between the steps.
        """
        self.start()
        for task in self.scenario:
            self.run_task(task)
        return self.finish()

    def start(self) -> None:
        """Prepare the model, the scenario, the strategy and the accuracy matrix."""
        model, scen = self.model, self.scenario
        model.to(self.device)
        scen.setup(model)
        self.strategy.bind(self)
        T = scen.num_tasks
        self.acc = np.full((T, T), np.nan)
        self.results = {"accuracy_matrix": self.acc, "task_results": [], "config": self.config()}
        self._last_task, self._train_time = None, float("nan")
        self.callbacks.on_train_begin(self)
        bb = [p for p in model._backbone.parameters()]
        n_train = sum(p.numel() for p in bb if p.requires_grad)
        self.results["config"]["backbone_trainable_params_at_start"] = n_train
        self.results["config"]["backbone_params"] = sum(p.numel() for p in bb)
        self.log(f"\n{'=' * 70}\n{self.strategy!r} | {scen.kind}-IL | {T} tasks | "
                 f"{self.epochs_per_task} ep/task | lr={self.lr} | {self.precision}\n"
                 f"model {self.results['config']['model']} | backbone "
                 f"{'frozen' if n_train == 0 else f'trainable ({n_train:,} params)'} at start\n{'=' * 70}")
        self._t_start = time.time()

    def finish(self) -> dict:
        """Compute the metrics from the accuracy matrix and return the results."""
        if not hasattr(self, "_t_start"):
            raise RuntimeError("call trainer.start() before the loop")
        last = self._last_task
        if last is not None and np.isnan(self.acc[last]).all():
            self.evaluate_seen(last)                      # eval_every_task=False: evaluate once at the end
        sizes = [len(t.test_loader.dataset) for t in self.scenario.tasks]
        self.results["test_sizes"] = sizes
        self.results.update(compute_cl_metrics(self.acc, sizes))
        self.results["total_time"] = time.time() - self._t_start
        self.callbacks.on_train_end(self, self.results)
        if self.verbose:
            print_results(self.acc, self.results)
        return self.results

    def run_task(self, task: TaskInfo) -> None:
        """Train on one task, then evaluate every task seen so far (if ``eval_every_task``)."""
        self.train_task(task)
        if self.eval_every_task:
            self.evaluate(task.task_id)
        else:
            self._record_task(task.task_id, [], 0.0)

    def train_task(self, task: TaskInfo) -> None:
        """Train on one task: the strategy's ``on_task_start``, the epochs, its ``on_task_end``."""
        if not hasattr(self, "_t_start"):
            raise RuntimeError("call trainer.start() before the loop")
        t, model, strat = task.task_id, self.model, self.strategy
        strat.task = task
        self._last_task = t
        self.log(f"\n--- Task {t}/{self.scenario.num_tasks - 1}  classes={len(task.classes)}  "
                 f"train={len(task.train_loader.dataset)}")
        self.callbacks.on_task_begin(self, t)
        model.begin_task(t, self.scenario.num_tasks)
        strat.on_task_start(t, task.train_loader, model)
        self._set_train_mask(t)

        t0 = time.time()
        if strat.wants_training(t) and self.epochs_per_task > 0:
            params = list({id(p): p for p in model.trainable_params() + strat.extra_params()}.values())
            if not params:
                raise RuntimeError(f"Task {t}: nothing to train (all parameters frozen)")
            self.optimizer = self.make_optimizer(params)
            self.scheduler = self.make_scheduler(self.optimizer, len(task.train_loader) * self.epochs_per_task)
            strat.optimizer = self.optimizer
            self.log(f"  trainable params: {sum(p.numel() for p in params):,}")
            for epoch in range(self.epochs_per_task):
                model.train()
                self.callbacks.on_epoch_begin(self, t, epoch, self.epochs_per_task)
                m = self.run_epoch(task, epoch)
                self.callbacks.on_epoch_end(self, t, epoch, self.epochs_per_task, m)
                self.log(f"  epoch {epoch + 1}/{self.epochs_per_task}  loss={m['loss']:.4f}  "
                         f"acc={m['accuracy']:.1f}%  ({m['time']:.0f}s)")
                if self.stop_training:
                    self.stop_training = False
                    break
        model.eval()
        # NOT inside autocast: strategies may update parameters here (SLCA, GDumb), and a
        # long-lived autocast region would keep serving stale cached bf16 weight casts.
        strat.on_task_end(t, task.train_loader, model)
        model.end_task(t)
        self._train_time = time.time() - t0

    def evaluate(self, after_task: int) -> List[float]:
        """Test on every task seen so far, fill row ``after_task`` of the accuracy matrix and
        return those accuracies (one per task)."""
        t1 = time.time()
        accs = self.evaluate_seen(after_task)
        self._record_task(after_task, accs, time.time() - t1)
        return accs

    def _record_task(self, t: int, accs: List[float], eval_time: float) -> None:
        tm = {"task_id": t, "accuracy": float(np.mean(accs)) if accs else float("nan"),
              "task_accuracies": accs, "train_time": self._train_time, "eval_time": eval_time}
        self.results["task_results"].append(tm)
        self.callbacks.on_task_end(self, t, tm)

    def run_epoch(self, task: TaskInfo, epoch: int) -> dict:
        t0, loss_sum, correct, total, n = time.time(), 0.0, 0, 0, 0
        for i, batch in enumerate(task.train_loader):
            self.callbacks.on_batch_begin(self, task.task_id, epoch, i)
            loss, c, k = self.train_step(batch, task.task_id, epoch, i)
            loss_sum, correct, total, n = loss_sum + loss, correct + c, total + k, n + 1
            self.callbacks.on_batch_end(self, task.task_id, epoch, i, loss)
            if self.stop_epoch:
                self.stop_epoch = False
                break
        return {"loss": loss_sum / max(n, 1), "accuracy": 100.0 * correct / max(total, 1),
                "lr": self.optimizer.param_groups[0]["lr"], "time": time.time() - t0}

    def train_step(self, batch, task_id: int, epoch: int = 0, batch_idx: int = 0):
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            out = self.strategy.compute_loss(batch, self.model, task_id)
        loss, logits = out if isinstance(out, tuple) else (out, None)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss ({loss.item()}) at task {task_id}, epoch {epoch}, batch {batch_idx} "
                f"with {self.strategy!r}, train_mask={self.strategy.train_mask!r}. A common cause is a "
                "logit mask that excludes a sample's label (see Strategy.memory_mask).")
        loss.backward()
        self.callbacks.on_before_step(self, task_id, epoch, batch_idx, loss)
        self.strategy.before_update(self.model, self.optimizer)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if logits is None:
            return loss.item(), 0, 0
        y = batch[1].to(logits.device)
        return loss.item(), (logits.detach().argmax(1) == y).sum().item(), y.size(0)

    # ------------------------------------------------------------------ evaluation
    def evaluate_seen(self, after_task: int) -> List[float]:
        """Evaluate every task seen so far; fills row ``after_task`` of the accuracy matrix."""
        self.model.eval()
        self._set_eval_mask(after_task)
        accs, extras = [], {}
        for j in range(after_task + 1):
            info = self.scenario.get_task(j)
            tid = j if self.scenario.task_aware else None      # no task oracle unless Task-IL
            self.callbacks.on_eval_begin(self, after_task, j)
            self.strategy.before_eval(self.model, tid)
            with torch.no_grad(), self.autocast():
                r = self.strategy.evaluate_task(info.test_loader, self.model, tid)
                if r is None:
                    r = self.evaluate_task(info.test_loader, tid)
            self.acc[after_task, j] = r["accuracy"]
            accs.append(r["accuracy"])
            # any extra numeric metric a strategy returns (e.g. routing_accuracy) is kept
            for k, v in r.items():
                if k not in ("accuracy", "loss") and isinstance(v, (int, float)):
                    extras.setdefault(k, []).append(float(v))
                    self.results.setdefault("eval_extras", {}).setdefault(k, {})[f"{after_task},{j}"] = float(v)
            self.callbacks.on_eval_end(self, after_task, j, r)
        line = "  eval: " + "  ".join(f"T{j}={a:.1f}" for j, a in enumerate(accs)) + f"  | avg={np.mean(accs):.2f}"
        for k, vs in extras.items():
            line += f"  | {k}: " + " ".join(f"{v:.0f}" for v in vs) + f" (avg {np.mean(vs):.1f})"
            self.results[f"final_{k}"] = float(np.mean(vs))      # overwritten each task -> final row
        self.log(line)
        return accs

    @torch.no_grad()
    def evaluate_task(self, loader, task_id: Optional[int]) -> dict:
        correct = total = 0
        loss = 0.0
        for batch in loader:
            x, y = B.to_device(batch[0], self.device), batch[1].to(self.device, non_blocking=True)
            logits = self.model(x, task_id).float()
            loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return {"accuracy": 100.0 * correct / max(total, 1), "loss": loss / max(total, 1)}

    # ------------------------------------------------------------------ factories
    def _set_train_mask(self, t: int) -> None:
        mode = self.strategy.train_mask
        if not self.scenario.shared_head or mode == "none":
            self.model.set_class_mask(None)
        elif mode == "current":
            self.model.set_class_mask(self.scenario.get_task(t).classes)
        else:
            self.model.set_class_mask(self.scenario.seen_classes(t))

    def _set_eval_mask(self, t: int) -> None:
        if not self.scenario.shared_head or self.strategy.eval_mask == "none":
            self.model.set_class_mask(None)
        else:
            self.model.set_class_mask(self.scenario.seen_classes(t))

    def make_optimizer(self, params) -> torch.optim.Optimizer:
        return self.optimizer_cls(params, lr=self.lr, weight_decay=self.weight_decay)

    def make_scheduler(self, opt, total_steps: int):
        if self.scheduler_name in (None, "none", "constant"):
            return None
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(total_steps, 1))
        raise ValueError(f"unknown scheduler {self.scheduler_name!r}")

    def config(self) -> dict:
        return {"strategy": repr(self.strategy), "scenario": self.scenario.kind,
                "benchmark": getattr(self.scenario.benchmark, "name", "?"),
                "preprocess": getattr(self.scenario.benchmark, "preprocess", "custom"),
                "num_tasks": self.scenario.num_tasks,
                "model": getattr(self.model, "model_name", type(self.model).__name__),
                "epochs_per_task": self.epochs_per_task, "lr": self.lr, "precision": self.precision,
                "optimizer": self.optimizer_cls.__name__, "weight_decay": self.weight_decay,
                "scheduler": self.scheduler_name or "constant", "batch_size": self.scenario.batch_size,
                "train_mask": self.strategy.train_mask, "eval_mask": self.strategy.eval_mask}

    def save_results(self, path: str) -> None:
        save_results(self.results, path)

    # backwards compat
    @property
    def task_loaders(self):
        return self.scenario.tasks


# ---------------------------------------------------------------------- metrics
def compute_cl_metrics(acc: np.ndarray, test_sizes=None) -> dict:
    """Standard metrics from the (T x T) accuracy matrix (row = after task i).

    * avg_accuracy: mean over tasks of the final accuracies (A_T, task-averaged)
    * avg_incremental_accuracy: mean of A_t over t
    * avg_forgetting: mean over old tasks of (best past acc - final acc)
    * backward_transfer: mean over old tasks of (final acc - acc right after learning it)

    With ``test_sizes`` (test samples per task) also the *sample-weighted* versions,
    i.e. accuracy over all test samples seen so far, which PTM-CL papers report
    (SimpleCIL/APER "A_B" and "Ā", RanPAC, ...):
    * accuracy_all_seen: final accuracy over all test samples
    * avg_incremental_accuracy_all_seen: mean over t of accuracy on all samples seen at t
    They equal the task-averaged ones when tasks have equal test-set sizes (e.g. CIFAR).
    """
    T = acc.shape[0]
    last = int(np.max(np.where(~np.isnan(acc).all(1))[0])) if not np.isnan(acc).all() else 0
    row_means = [float(np.nanmean(acc[i, : i + 1])) for i in range(last + 1) if not np.isnan(acc[i, 0])]
    out = {"avg_accuracy": float(np.nanmean(acc[last, : last + 1])),
           "avg_incremental_accuracy": float(np.mean(row_means)) if row_means else float("nan"),
           "avg_forgetting": 0.0, "backward_transfer": 0.0}
    if last >= 1 and not np.isnan(acc[: last + 1, : last + 1].diagonal()).any():
        f = [np.nanmax(acc[j:last, j]) - acc[last, j] for j in range(last)]
        b = [acc[last, j] - acc[j, j] for j in range(last)]
        out["avg_forgetting"], out["backward_transfer"] = float(np.mean(f)), float(np.mean(b))
    if test_sizes is not None:
        n = np.asarray(test_sizes, dtype=float)
        def weighted(i):
            return float(np.sum(acc[i, : i + 1] * n[: i + 1]) / np.sum(n[: i + 1]))
        rows = [i for i in range(last + 1) if not np.isnan(acc[i, : i + 1]).any()]
        if last in rows:
            out["accuracy_all_seen"] = weighted(last)
        if rows:
            out["avg_incremental_accuracy_all_seen"] = float(np.mean([weighted(i) for i in rows]))
    return out


def print_results(acc: np.ndarray, m: dict) -> None:
    T = acc.shape[0]
    print(f"\n{'=' * 70}\nAccuracy matrix (row: after task i, col: task j)\n" + " " * 9
          + "".join(f"{'T' + str(j):>7s}" for j in range(T)))
    for i in range(T):
        if np.isnan(acc[i]).all():
            continue
        print(f"after T{i:<2d}" + "".join(f"{acc[i, j]:7.1f}" if j <= i else f"{'':>7s}" for j in range(T)))
    print(f"\n  Final avg accuracy  : {m['avg_accuracy']:.2f}%\n"
          f"  Avg incremental acc : {m['avg_incremental_accuracy']:.2f}%\n"
          f"  Avg forgetting      : {m['avg_forgetting']:.2f}%\n"
          f"  Backward transfer   : {m['backward_transfer']:.2f}%"
          + (f"\n  Acc. on all seen test samples (sample-weighted): final {m['accuracy_all_seen']:.2f}%, "
             f"avg incremental {m['avg_incremental_accuracy_all_seen']:.2f}%"
             if "accuracy_all_seen" in m and "avg_incremental_accuracy_all_seen" in m else "")
          + f"\n{'=' * 70}")


def save_results(results: dict, path: str) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return np.where(np.isnan(o), None, np.round(o, 3)).tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=conv)


Trainer = ContinualTrainer

"""ContinualTrainer — the single training loop for all CTK cl experiments."""

import time
import torch
import numpy as np
from typing import List, Optional, Type, Dict
from torch.optim import AdamW

from ..models.base import CLModel
from ..scenarios.base import CLScenario, TaskInfo
from ..strategies.base import Strategy


class ContinualTrainer:
    """Orchestrates the full multi-task training loop and computes standard CL metrics.

    Every logical step is a separate overridable method so subclasses can
    customise behaviour without rewriting the whole loop::

        class AmpTrainer(ContinualTrainer):
            def forward(self, x, task_id):
                with torch.autocast("cuda"):
                    return self.model(x, task_id)

        class MixupTrainer(ContinualTrainer):
            def train_step(self, batch, task_id):
                batch = self.mixup(batch)
                return super().train_step(batch, task_id)

    Override points (coarsest → finest):
        fit()          — full training run
        run_task()     — one task (epochs + evaluation)
        run_epoch()    — one epoch over the task's DataLoader
        train_step()   — one batch: forward, loss, backward, step
        forward()      — model forward pass only
        evaluate_task()— evaluation on one task's test loader
    """

    def __init__(
        self,
        model: CLModel,
        scenario: CLScenario,
        strategy: Optional[Strategy] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = AdamW,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs_per_task: int = 10,
        callbacks=None,
        device=None,
        verbose: bool = True,
    ):
        if strategy is None:
            from ..strategies import NaiveStrategy
            strategy = NaiveStrategy()

        self.model           = model
        self.scenario        = scenario
        self.strategy        = strategy
        self.optimizer_cls   = optimizer_cls
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.epochs_per_task = epochs_per_task
        self.verbose         = verbose

        self.device = (
            torch.device(device) if isinstance(device, str)
            else device if device is not None
            else model._device
        )

        from ..callbacks import CallbackList
        self.callbacks = (
            callbacks if isinstance(callbacks, CallbackList)
            else CallbackList(list(callbacks) if callbacks else [])
        )

        self.stop_training = False
        self.stop_epoch    = False

        self.results: Dict = {
            'task_results':    [],
            'accuracy_matrix': None,
        }

        # Set during run_task so train_step can access the optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run the full continual learning loop and return the results dict."""
        model    = self.model
        scenario = self.scenario

        model.to(self.device)
        scenario.setup(model)

        num_tasks  = scenario.num_tasks
        acc_matrix = np.zeros((num_tasks, num_tasks))
        self.results['accuracy_matrix'] = acc_matrix
        self.results['task_results']    = []

        self.callbacks.on_train_begin(self)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ContinualTrainer — {num_tasks} tasks, "
                  f"{self.epochs_per_task} epochs/task, lr={self.lr}")
            print(f"{'='*70}")

        for task_info in scenario:
            self.run_task(task_info, acc_matrix)

        cl_metrics = self._compute_cl_metrics(acc_matrix)
        self.results.update(cl_metrics)
        self.callbacks.on_train_end(self, self.results)

        if self.verbose:
            self._print_results(acc_matrix, cl_metrics)

        return self.results

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def run_task(self, task_info: TaskInfo, acc_matrix: np.ndarray) -> None:
        """Train on one task then evaluate all tasks seen so far.

        Override to add multi-phase training, task-level scheduling, etc.
        """
        task_id  = task_info.task_id
        model    = self.model
        strategy = self.strategy
        num_tasks = self.scenario.num_tasks

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Task {task_id}/{num_tasks - 1}")
            print(f"{'='*70}")

        self.callbacks.on_task_begin(self, task_id)
        strategy.on_task_start(task_id, task_info.train_loader, model)

        self.optimizer = self._make_optimizer(
            model.trainable_params() + strategy.extra_params()
            or [p for p in model.get_head().parameters() if p.requires_grad]
        )
        if not self.optimizer.param_groups[0]['params']:
            raise RuntimeError(
                f"Task {task_id}: no trainable parameters found. "
                "Unfreeze the backbone or head before training."
            )
        strategy.optimizer = self.optimizer

        # ---- epoch loop ------------------------------------------------
        t0 = time.time()
        for epoch in range(self.epochs_per_task):
            model.train()
            self.callbacks.on_epoch_begin(self, task_id, epoch, self.epochs_per_task)

            metrics = self.run_epoch(task_info, epoch)

            self.callbacks.on_epoch_end(self, task_id, epoch, self.epochs_per_task, metrics)

            if self.verbose:
                print(f"  Epoch {epoch + 1}/{self.epochs_per_task}  "
                      f"loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.1f}%")

            if self.stop_training:
                self.stop_training = False
                break

        train_time = time.time() - t0
        strategy.on_task_end(task_id, task_info.train_loader, model)

        # ---- evaluation ------------------------------------------------
        model.eval()
        if self.verbose:
            print(f"\n  Evaluating tasks 0-{task_id}:")

        t1 = time.time()
        task_accs: List[float] = []
        for eval_id in range(task_id + 1):
            eval_info = self.scenario.get_task(eval_id)
            self.callbacks.on_eval_begin(self, task_id, eval_id)

            result = strategy.evaluate_task(eval_info.test_loader, model, eval_id)
            if result is None:
                result = self.evaluate_task(eval_info.test_loader, eval_id)

            acc_matrix[task_id][eval_id] = result['accuracy']
            task_accs.append(result['accuracy'])
            self.callbacks.on_eval_end(self, task_id, eval_id, result)

            if self.verbose:
                print(f"    Task {eval_id}: {result['accuracy']:.1f}%")

        eval_time = time.time() - t1

        task_metrics = {
            'accuracy':        float(np.mean(task_accs)),
            'task_accuracies': task_accs,
            'accuracy_matrix': acc_matrix.copy(),
            'train_time':      train_time,
            'eval_time':       eval_time,
            'total_time':      train_time + eval_time,
        }
        self.results['accuracy_matrix'] = acc_matrix
        self.results['task_results'].append({'task_id': task_id, **task_metrics})
        self.callbacks.on_task_end(self, task_id, task_metrics)

    def run_epoch(self, task_info: TaskInfo, epoch: int) -> dict:
        """Run one epoch over task_info.train_loader and return metrics dict.

        Override to add custom data augmentation, sampling, or logging.
        """
        task_id    = task_info.task_id
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for batch_idx, batch in enumerate(task_info.train_loader):
            self.callbacks.on_batch_begin(self, task_id, epoch, batch_idx)

            loss, n_correct, n_total = self.train_step(batch, task_id)

            epoch_loss += loss
            correct    += n_correct
            total      += n_total

            self.callbacks.on_batch_end(self, task_id, epoch, batch_idx,
                                        torch.tensor(loss))

            if self.stop_epoch:
                self.stop_epoch = False
                break

        n = max(len(task_info.train_loader), 1)
        return {
            'loss':     epoch_loss / n,
            'accuracy': 100.0 * correct / max(total, 1),
            'lr':       self.optimizer.param_groups[0]['lr'],
        }

    def train_step(self, batch, task_id: int):
        """Forward, loss, backward, optimizer step for one batch.

        Returns (loss_value, n_correct, n_total).
        Override for custom loss scaling, gradient accumulation, AMP, etc.
        """
        self.optimizer.zero_grad()
        loss = self.strategy.compute_loss(batch, self.model, task_id)
        loss.backward()

        self.callbacks.on_before_step(self, task_id, 0, 0, loss)
        self.strategy.before_update(self.model, self.optimizer)
        self.optimizer.step()

        with torch.no_grad():
            x      = batch[0].to(self.device)
            y      = batch[1].to(self.device)
            logits = self.forward(x, task_id)
            n_correct = (logits.argmax(1) == y).sum().item()
            n_total   = y.size(0)

        return loss.item(), n_correct, n_total

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Model forward pass.

        Override for mixed precision, gradient checkpointing, etc.::

            def forward(self, x, task_id):
                with torch.autocast("cuda"):
                    return self.model(x, task_id)
        """
        return self.model(x, task_id)

    def evaluate_task(
        self,
        test_loader: torch.utils.data.DataLoader,
        task_id: int,
    ) -> dict:
        """Evaluate the model on one task's test set.

        Returns {'accuracy': float, 'loss': float}.
        Override to add custom metrics (e.g. F1, per-class accuracy).
        """
        import torch.nn.functional as F

        total_loss = 0.0
        correct    = 0
        total      = 0

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                logits      = self.forward(x, task_id)
                total_loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        n = max(total, 1)
        return {'accuracy': 100.0 * correct / n, 'loss': total_loss / n}

    # ------------------------------------------------------------------
    # CL metrics
    # ------------------------------------------------------------------

    def _compute_cl_metrics(self, acc_matrix: np.ndarray) -> dict:
        n            = acc_matrix.shape[0]
        avg_accuracy = float(np.mean(acc_matrix[n - 1, :n]))

        if n < 2:
            return {'avg_accuracy': avg_accuracy,
                    'avg_forgetting': 0.0, 'backward_transfer': 0.0}

        forgetting, bwt = [], []
        for j in range(n - 1):
            diag_acc  = acc_matrix[j, j]
            final_acc = acc_matrix[n - 1, j]
            max_seen  = float(np.max(acc_matrix[j:n, j]))
            forgetting.append(max_seen - final_acc)
            bwt.append(final_acc - diag_acc)

        return {
            'avg_accuracy':      avg_accuracy,
            'avg_forgetting':    float(np.mean(forgetting)),
            'backward_transfer': float(np.mean(bwt)),
        }

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_results(self, acc_matrix: np.ndarray, metrics: dict) -> None:
        n     = acc_matrix.shape[0]
        col_w = 7

        print(f"\n{'='*70}")
        print("Final Accuracy Matrix  (row = after task N, col = eval on task M)")
        print(f"{'='*70}")
        print(f"{'':8s}" + "".join(f"T{j:<{col_w - 1}}" for j in range(n)))
        print("-" * (8 + col_w * n))
        for i in range(n):
            row = f"After T{i} "
            for j in range(n):
                row += f"{acc_matrix[i,j]:6.1f} " if j <= i else f"{'---':>6s} "
            print(row)

        print(f"\n{'='*70}")
        print(f"  Average Accuracy   : {metrics['avg_accuracy']:.2f}%")
        print(f"  Average Forgetting : {metrics['avg_forgetting']:.2f}%")
        print(f"  Backward Transfer  : {metrics['backward_transfer']:.2f}%")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    def _make_optimizer(self, params: list) -> torch.optim.Optimizer:
        return self.optimizer_cls(params, lr=self.lr, weight_decay=self.weight_decay)

    # ------------------------------------------------------------------
    # Properties for callback compatibility
    # ------------------------------------------------------------------

    @property
    def task_loaders(self):
        if self.scenario._task_infos is None:
            self.scenario._task_infos = self.scenario._build_loaders()
        return self.scenario._task_infos


# Convenience alias
Trainer = ContinualTrainer

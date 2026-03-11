"""ContinualTrainer — the single training loop for all CTK experiments."""

import torch
import numpy as np
from typing import List, Optional, Type
from torch.optim import AdamW

from ctk.models.base import CLModel
from ctk.scenarios.base import CLScenario
from ctk.strategies.base import Strategy


class ContinualTrainer:
    """Orchestrates the full multi-task training loop and computes standard CL metrics."""

    def __init__(
        self,
        model: CLModel,
        scenario: CLScenario,
        strategy: Strategy,
        optimizer_cls: Type[torch.optim.Optimizer] = AdamW,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs_per_task: int = 10,
        callbacks=None,
        device=None,
        verbose: bool = True,
    ):
        self.model          = model
        self.scenario       = scenario
        self.strategy       = strategy
        self.optimizer_cls  = optimizer_cls
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.epochs_per_task = epochs_per_task
        self.verbose        = verbose

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = model._device

        from ctk.callbacks import CallbackList
        if callbacks is None:
            self.callbacks = CallbackList([])
        elif isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(list(callbacks))

        self.stop_training = False
        self.stop_epoch    = False

        self.results = {
            'task_results': [],
            'accuracy_matrix': None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run the full continual learning training loop and return results dict."""
        model    = self.model
        scenario = self.scenario
        strategy = self.strategy

        model.to(self.device)
        scenario.setup(model)

        num_tasks = scenario.num_tasks
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
            task_id = task_info.task_id

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Task {task_id}/{num_tasks - 1}")
                print(f"{'='*70}")

            self.callbacks.on_task_begin(self, task_id)
            strategy.on_task_start(task_id, task_info.train_loader, model)

            optimizer = self._make_optimizer(
                model.trainable_params() + strategy.extra_params()
            )
            strategy.optimizer = optimizer

            # ---- Epoch loop -----------------------------------------------
            for epoch in range(self.epochs_per_task):
                model.train()
                self.callbacks.on_epoch_begin(
                    self, task_id, epoch, self.epochs_per_task)

                epoch_loss = 0.0
                correct    = 0
                total      = 0

                for batch_idx, batch in enumerate(task_info.train_loader):
                    self.callbacks.on_batch_begin(
                        self, task_id, epoch, batch_idx)

                    optimizer.zero_grad()
                    loss = strategy.compute_loss(batch, model, task_id)
                    loss.backward()

                    self.callbacks.on_before_step(
                        self, task_id, epoch, batch_idx, loss)
                    strategy.before_update(model, optimizer)
                    optimizer.step()

                    with torch.no_grad():
                        x = batch[0].to(self.device)
                        y = batch[1].to(self.device)
                        logits = model(x, task_id)
                        correct    += (logits.argmax(1) == y).sum().item()
                        total      += y.size(0)
                        epoch_loss += loss.item()

                    self.callbacks.on_batch_end(
                        self, task_id, epoch, batch_idx, loss)

                    if self.stop_epoch:
                        self.stop_epoch = False
                        break
                num_batches = max(len(task_info.train_loader), 1)
                metrics = {
                    'loss':     epoch_loss / num_batches,
                    'accuracy': 100.0 * correct / max(total, 1),
                }
                self.callbacks.on_epoch_end(
                    self, task_id, epoch, self.epochs_per_task, metrics)

                if self.verbose:
                    print(f"  Epoch {epoch + 1}/{self.epochs_per_task}  "
                          f"loss={metrics['loss']:.4f}  "
                          f"acc={metrics['accuracy']:.1f}%")

                if self.stop_training:
                    self.stop_training = False
                    break
            strategy.on_task_end(task_id, task_info.train_loader, model)

            model.eval()
            task_accs: List[float] = []

            if self.verbose:
                print(f"\n  Evaluating tasks 0-{task_id}:")

            for eval_id in range(task_id + 1):
                eval_info = scenario.get_task(eval_id)
                self.callbacks.on_eval_begin(self, task_id, eval_id)

                result = strategy.evaluate_task(
                    eval_info.test_loader, model, eval_id)
                if result is None:
                    result = self._evaluate(
                        eval_info.test_loader, model, eval_id)

                acc_matrix[task_id][eval_id] = result['accuracy']
                task_accs.append(result['accuracy'])
                self.callbacks.on_eval_end(self, task_id, eval_id, result)

                if self.verbose:
                    print(f"    Task {eval_id}: {result['accuracy']:.1f}%")

            task_metrics = {
                'accuracy':          float(np.mean(task_accs)),
                'task_accuracies':   task_accs,
                'accuracy_matrix':   acc_matrix.copy(),
            }
            self.results['accuracy_matrix'] = acc_matrix
            self.results['task_results'].append(
                {'task_id': task_id, **task_metrics})
            self.callbacks.on_task_end(self, task_id, task_metrics)
        cl_metrics = self._compute_cl_metrics(acc_matrix)
        self.results.update(cl_metrics)
        self.callbacks.on_train_end(self, self.results)

        if self.verbose:
            self._print_results(acc_matrix, cl_metrics)

        return self.results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: CLModel,
        task_id: int,
    ) -> dict:
        """Standard accuracy evaluation. Returns {'accuracy': float, 'loss': float}."""
        import torch.nn.functional as F

        total_loss    = 0.0
        correct       = 0
        total         = 0

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                logits = model(x, task_id)
                total_loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        n = max(total, 1)
        return {
            'accuracy': 100.0 * correct / n,
            'loss':     total_loss / n,
        }

    # ------------------------------------------------------------------
    # CL metrics - forgetting, avg-accuracy, backward transfer
    # ------------------------------------------------------------------

    def _compute_cl_metrics(self, acc_matrix: np.ndarray) -> dict:
        """Compute avg_accuracy, avg_forgetting, and backward_transfer from the accuracy matrix."""
        n = acc_matrix.shape[0]

        avg_accuracy = float(np.mean(acc_matrix[n - 1, :n]))

        if n < 2:
            return {
                'avg_accuracy':      avg_accuracy,
                'avg_forgetting':    0.0,
                'backward_transfer': 0.0,
            }

        forgetting_vals      = []
        backward_transfer_vals = []
        for j in range(n - 1):
            diag_acc  = acc_matrix[j, j]
            final_acc = acc_matrix[n - 1, j]
            max_seen  = float(np.max(acc_matrix[j:n, j]))

            forgetting_vals.append(max_seen - final_acc)
            backward_transfer_vals.append(final_acc - diag_acc)

        return {
            'avg_accuracy':      avg_accuracy,
            'avg_forgetting':    float(np.mean(forgetting_vals)),
            'backward_transfer': float(np.mean(backward_transfer_vals)),
        }

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_results(self, acc_matrix: np.ndarray, metrics: dict) -> None:
        """Print accuracy matrix table and summary CL metrics."""
        n = acc_matrix.shape[0]

        print(f"\n{'='*70}")
        print("Final Accuracy Matrix (rows=after training task N, "
              "cols=eval on task M)")
        print(f"{'='*70}")

        col_w = 7
        header = f"{'':8s}" + "".join(f"T{j:<{col_w - 1}}" for j in range(n))
        print(header)
        print("-" * (8 + col_w * n))

        for i in range(n):
            row_str = f"After T{i} "
            for j in range(n):
                if j <= i:
                    row_str += f"{acc_matrix[i, j]:6.1f} "
                else:
                    row_str += f"{'---':>6s} "
            print(row_str)

        print(f"\n{'='*70}")
        print(f"  Average Accuracy:       {metrics['avg_accuracy']:.2f}%")
        print(f"  Average Forgetting:     {metrics['avg_forgetting']:.2f}%")
        print(f"  Backward Transfer:      {metrics['backward_transfer']:.2f}%")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    def _make_optimizer(
        self, params: list
    ) -> torch.optim.Optimizer:
        """Build an optimizer from stored hyperparameters."""
        return self.optimizer_cls(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # Properties for callback compatibility
    # ------------------------------------------------------------------

    @property
    def task_loaders(self):
        """List of TaskInfo objects (lazy-built)."""
        if self.scenario._task_infos is None:
            self.scenario._task_infos = self.scenario._build_loaders()
        return self.scenario._task_infos


# Convenience alias
Trainer = ContinualTrainer


# - add forgetting alleviation as well
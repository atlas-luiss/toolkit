"""ContinualEvaluator — post-hoc evaluation of a trained model on a CLScenario."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from ..models.base import CLModel
from ..scenarios.base import CLScenario
from ..strategies.base import Strategy


class ContinualEvaluator:
    """Post-hoc evaluation of a trained model across all scenario tasks."""

    def __init__(
        self,
        model: CLModel,
        scenario: CLScenario,
        strategy: Optional[Strategy] = None,
        device=None,
    ):
        self.model    = model
        self.scenario = scenario
        self.strategy = strategy

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = model._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(self) -> dict:
        """Evaluate on all tasks and return accuracy/metric summary."""
        num_tasks = self.scenario.num_tasks
        task_accs: List[float] = []
        task_results: List[dict] = []

        self.model.eval()
        for task_id in range(num_tasks):
            task_info = self.scenario.get_task(task_id)
            result    = self._eval_one(task_info.test_loader, task_id)
            task_accs.append(result['accuracy'])
            task_results.append({'task_id': task_id, **result})

        acc_matrix = np.array(task_accs, dtype=float).reshape(1, num_tasks)

        return {
            'task_accuracies':   task_accs,
            'accuracy_matrix':   acc_matrix,
            'avg_accuracy':      float(np.mean(task_accs)),
            'avg_forgetting':    0.0,
            'backward_transfer': 0.0,
            'task_results':      task_results,
        }

    def evaluate_task(self, task_id: int) -> dict:
        """Evaluate on a single task; returns {'accuracy': float, 'loss': float}."""
        task_info = self.scenario.get_task(task_id)
        self.model.eval()
        return self._eval_one(task_info.test_loader, task_id)

    def print_report(self, results: Optional[dict] = None) -> None:
        """Print per-task accuracy table and CL metrics; calls evaluate_all() if results is None."""
        if results is None:
            results = self.evaluate_all()

        task_accs = results.get('task_accuracies', [])
        num_tasks = len(task_accs)

        print(f"\n{'='*60}")
        print("ContinualEvaluator — Evaluation Report")
        print(f"{'='*60}")
        print(f"  {'Task':<8}  {'Accuracy':>10}  {'Loss':>10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

        for r in results.get('task_results', []):
            tid  = r['task_id']
            acc  = r.get('accuracy', 0.0)
            loss = r.get('loss', float('nan'))
            print(f"  {tid:<8}  {acc:>9.2f}%  {loss:>10.4f}")

        print(f"  {'='*8}  {'='*10}  {'='*10}")
        print(f"  {'Average':<8}  {results['avg_accuracy']:>9.2f}%")
        print(f"\n  avg_forgetting:    {results['avg_forgetting']:.2f}%")
        print(f"  backward_transfer: {results['backward_transfer']:.2f}%")
        print(f"{'='*60}\n")

    def compute_cl_metrics(self, acc_matrix: np.ndarray) -> dict:
        """Compute avg_accuracy, avg_forgetting, and backward_transfer from acc_matrix[i,j]."""
        n = acc_matrix.shape[0]
        avg_accuracy = float(np.mean(acc_matrix[n - 1, :n]))

        if n < 2:
            return {
                'avg_accuracy':      avg_accuracy,
                'avg_forgetting':    0.0,
                'backward_transfer': 0.0,
            }

        forgetting_vals:       List[float] = []
        backward_transfer_vals: List[float] = []
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_one(
        self,
        test_loader: torch.utils.data.DataLoader,
        task_id: int,
    ) -> dict:
        """Evaluate one task, delegating to strategy hook or falling back to CE/accuracy."""
        if self.strategy is not None:
            result = self.strategy.evaluate_task(test_loader, self.model, task_id)
            if result is not None:
                return result

        total_loss = 0.0
        correct    = 0
        total      = 0

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                logits     = self.model(x, task_id)
                total_loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        n = max(total, 1)
        return {
            'accuracy': 100.0 * correct / n,
            'loss':     total_loss / n,
        }

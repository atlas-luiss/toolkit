"""
Progressive results callback — displays accuracy matrix and metrics after each task.
"""

import numpy as np
from tabulate import tabulate
from ctk.utils import format_time

    
def format_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Callback:
    """Base class for all callbacks. Override only the hooks you need."""

    def on_train_begin(self, trainer, **kwargs): pass
    def on_train_end(self, trainer, *args, **kwargs): pass

    def on_task_begin(self, trainer, task_id, **kwargs): pass
    def on_task_end(self, trainer, task_id, task_metrics, **kwargs): pass

    def on_epoch_begin(self, trainer, task_id, epoch, total_epochs, **kwargs): pass
    def on_epoch_end(self, trainer, task_id, epoch, total_epochs, metrics, **kwargs): pass

    def on_batch_begin(self, trainer, task_id, epoch, batch_idx, **kwargs): pass
    def on_forward_end(self, trainer, task_id, epoch, batch_idx, logits, **kwargs): pass
    def on_before_step(self, trainer, task_id, epoch, batch_idx, loss, **kwargs): pass
    def on_batch_end(self, trainer, task_id, epoch, batch_idx, loss, **kwargs): pass

    def on_eval_begin(self, trainer, task_id, eval_task_id, **kwargs): pass
    def on_eval_end(self, trainer, task_id, eval_task_id, metrics, **kwargs): pass


class ProgressiveResultsCallback(Callback):
    """Print accuracy matrix, metrics, and timing after each task."""

    def __init__(self, show_after_each_task: bool = True, verbose: bool = True,
                 # kept for backwards compat with existing configs — ignored here,
                 # use AdapterMergeCallback instead
                 test_merge_interval: int = 0, merge_strategies: list = None):
        super().__init__()
        self.show_after_each_task = show_after_each_task
        self.verbose = verbose

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        if not self.show_after_each_task or not self.verbose:
            return

        print(f"\n{'='*70}")
        print(f"TASK {task_id} COMPLETED - Progressive Results")
        print(f"{'='*70}")

        acc_matrix = trainer.results.get('accuracy_matrix')
        if acc_matrix is None:
            return

        print(f"\nAccuracy Matrix (Tasks 0-{task_id}):")
        headers = ["T\\E"] + [f"T{i}" for i in range(task_id + 1)]
        rows = []
        for i in range(task_id + 1):
            row = [f"T{i}"]
            for j in range(task_id + 1):
                row.append(f"{acc_matrix[i][j]:.2f}" if j <= i else "-")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        completed = task_id + 1
        current_acc = acc_matrix[task_id][task_id]
        # Average accuracy = mean of the current row (retention across all seen tasks)
        avg_acc = np.mean(acc_matrix[task_id][:completed])

        print(f"\nMetrics after Task {task_id}:")
        print(f"  Current task accuracy:  {current_acc:.2f}%")
        print(f"  Average accuracy:       {avg_acc:.2f}%")
        if completed >= 2:
            forgetting = np.mean([
                acc_matrix[i][i] - acc_matrix[task_id][i]
                for i in range(completed - 1)
            ])
            print(f"  Average forgetting:     {forgetting:.2f}%")

        task_results = trainer.results.get('task_results', [])
        if task_results and len(task_results) > task_id:
            r = task_results[task_id]
            train_t = r.get('train_time', 0.0)
            eval_t  = r.get('eval_time', 0.0)
            total_t = r.get('total_time', 0.0)
            print(f"\nTiming:")
            print(f"  Training:    {format_time(train_t)}")
            print(f"  Evaluation:  {format_time(eval_t)}")
            print(f"  Total:       {format_time(total_t)}")

        strategy = trainer.strategy
        if hasattr(strategy, 'groups') and strategy.groups:
            print(f"\nGroups: {len(strategy.groups)}")
            for g_id, group in enumerate(strategy.groups):
                tasks = group.get('tasks', [])
                alpha = group.get('alpha', 0)
                size  = group.get('size', 0)
                print(f"  Group {g_id}: {size} tasks, alpha={alpha:.4f}, tasks={tasks}")

    def on_train_end(self, trainer, *args, **kwargs):
        if not self.verbose:
            return

        print(f"\n{'='*70}")
        print("FINAL RESULTS - All Tasks Complete")
        print(f"{'='*70}")

        acc_matrix = trainer.results.get('accuracy_matrix')
        if acc_matrix is None:
            return
        num_tasks = acc_matrix.shape[0]

        print("\nFinal Accuracy Matrix:")
        headers = ["T\\E"] + [f"T{i}" for i in range(num_tasks)]
        rows = []
        for i in range(num_tasks):
            row = [f"T{i}"]
            for j in range(num_tasks):
                row.append(f"{acc_matrix[i][j]:.2f}" if j <= i else "-")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

"""Continual-learning-specific callbacks: merge comparison and per-task metrics."""

import sys
import json
from pathlib import Path
import numpy as np
from tabulate import tabulate
from .progressive_results import Callback


class AdapterMergeCallback(Callback):
    """Test adapter merge strategies at task intervals and compare against the individual-adapter baseline."""

    def __init__(self, strategies=None, interval=5):
        self.strategies = strategies or ['linear', 'ties']
        self.interval   = interval
        self.history    = []   # list of {task_id, baseline, results}

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        completed = task_id + 1
        run_now = (
            self.interval > 0 and completed % self.interval == 0 and completed >= 2
        ) or (
            completed == len(getattr(trainer, 'task_loaders', [None]))
        )
        if not run_now or not self.strategies:
            return
        self._run_comparison(trainer, task_id)

    def _run_comparison(self, trainer, task_id):
        print(f"\n{'='*70}")
        print(f"MERGE COMPARISON after Task {task_id}")
        print(f"{'='*70}")
        sys.stdout.flush()

        strategy     = trainer.strategy
        task_loaders = getattr(trainer, 'task_loaders', [])

        if not task_loaders or not hasattr(strategy, 'perform_final_merge'):
            print("  [SKIP] strategy does not support merging or task_loaders unavailable")
            sys.stdout.flush()
            return

        baseline_acc = float(
            trainer.results['accuracy_matrix'][task_id, :task_id + 1].mean()
        )
        print(f"\n  Individual adapters (baseline): {baseline_acc:.2f}%")
        sys.stdout.flush()

        record = {'task_id': task_id, 'baseline': baseline_acc, 'results': {}}

        for merge_type in self.strategies:
            density = 0.2 if merge_type in ('ties', 'dare_ties') else 1.0
            try:
                strategy.perform_final_merge(combination_type=merge_type, density=density)
                accs = []
                for eval_id in range(task_id + 1):
                    if eval_id < len(task_loaders):
                        _, test_loader = task_loaders[eval_id]
                        m = strategy.evaluate_with_merged(test_loader=test_loader, verbose=False)
                        accs.append(m['accuracy'])
                avg  = float(np.mean(accs))
                diff = avg - baseline_acc
                sign = '+' if diff >= 0 else ''
                mark = '+' if diff >= 0 else '-'
                print(f"  {mark} {merge_type.upper():12s}: {avg:.2f}% ({sign}{diff:.2f}%)")
                record['results'][merge_type] = {'avg': avg, 'accs': accs}
            except Exception as e:
                print(f"  x {merge_type.upper():12s}: FAILED — {e}")
            sys.stdout.flush()

        self.history.append(record)

    def on_train_end(self, trainer, *args, **kwargs):
        if not self.history:
            return
        print(f"\n{'='*70}")
        print("MERGE STRATEGY SUMMARY")
        print(f"{'='*70}")
        for record in self.history:
            t        = record['task_id']
            baseline = record['baseline']
            print(f"\nAfter Task {t}  (baseline: {baseline:.2f}%)")
            for name, res in record['results'].items():
                avg  = res['avg']
                diff = avg - baseline
                sign = '+' if diff >= 0 else ''
                print(f"  {name.upper():12s}: {avg:.2f}% ({sign}{diff:.2f}%)")


class TaskMetricsCallback(Callback):
    """Collect per-task accuracy, forgetting, and timing; print summary and optionally dump JSON."""

    def __init__(self, save_json=None):
        self.save_json = save_json
        self.records   = []   # one entry per task

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        acc_matrix   = trainer.results.get('accuracy_matrix')
        task_results = trainer.results.get('task_results', [])

        current_acc = float(acc_matrix[task_id][task_id]) if acc_matrix is not None else 0.0

        timing = {}
        if task_results and len(task_results) > task_id:
            r      = task_results[task_id]
            timing = {
                'train_s': r.get('train_time', 0.0),
                'eval_s':  r.get('eval_time',  0.0),
                'total_s': r.get('total_time', 0.0),
            }

        forgetting_per_prev = []
        if acc_matrix is not None and task_id > 0:
            for prev in range(task_id):
                initial  = float(acc_matrix[prev][prev])
                current  = float(acc_matrix[task_id][prev])
                forgetting_per_prev.append(initial - current)

        self.records.append({
            'task_id':          task_id,
            'accuracy':         current_acc,
            'forgetting_prev':  forgetting_per_prev,
            'avg_forgetting':   float(np.mean(forgetting_per_prev)) if forgetting_per_prev else 0.0,
            **timing,
        })

    def on_train_end(self, trainer, *args, **kwargs):
        if not self.records:
            return

        acc_matrix = trainer.results.get('accuracy_matrix')
        num_tasks  = len(self.records)

        avg_accuracy = np.mean([r['accuracy'] for r in self.records])
        avg_forgetting = np.mean([r['avg_forgetting'] for r in self.records[1:]]) if num_tasks > 1 else 0.0

        bwt = -avg_forgetting

        total_train  = sum(r.get('train_s', 0.0) for r in self.records)
        total_eval   = sum(r.get('eval_s', 0.0) for r in self.records)
        total_time   = sum(r.get('total_s', 0.0) for r in self.records)

        print(f"\n{'='*70}")
        print("TASK METRICS SUMMARY")
        print(f"{'='*70}")

        rows = []
        for r in self.records:
            rows.append([
                f"T{r['task_id']}",
                f"{r['accuracy']:.2f}%",
                f"{r['avg_forgetting']:.2f}%" if r['task_id'] > 0 else "-",
                f"{r.get('train_s', 0.0):.1f}s",
                f"{r.get('eval_s', 0.0):.1f}s",
            ])
        print(tabulate(rows,
                       headers=["Task", "Accuracy", "Forgetting", "Train time", "Eval time"],
                       tablefmt="simple"))

        print(f"\nAggregate:")
        print(f"  Average Accuracy:   {avg_accuracy:.2f}%")
        print(f"  Avg Forgetting:     {avg_forgetting:.2f}%")
        print(f"  Backward Transfer:  {bwt:.2f}%")
        print(f"  Total train time:   {total_train:.1f}s")
        print(f"  Total eval time:    {total_eval:.1f}s")
        print(f"  Total time:         {total_time:.1f}s")

        if self.save_json:
            Path(self.save_json).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'tasks':           self.records,
                'avg_accuracy':    avg_accuracy,
                'avg_forgetting':  avg_forgetting,
                'bwt':             bwt,
                'total_train_s':   total_train,
                'total_eval_s':    total_eval,
                'total_s':         total_time,
            }
            with open(self.save_json, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"\n  Metrics saved to {self.save_json}")

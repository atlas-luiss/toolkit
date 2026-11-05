"""Standard callbacks for training loop monitoring and control."""

import os
import csv
import time
import math
from pathlib import Path
import torch
from . import Callback


class EarlyStopping(Callback):
    """Stop training within a task when a monitored metric stops improving; resets per task."""

    def __init__(self, monitor='loss', patience=3, min_delta=0.001, mode='min'):
        self.monitor   = monitor
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self._reset()

    def _reset(self):
        self.wait = 0
        self.best = None

    def _is_improvement(self, value):
        if self.best is None:
            return True
        if self.mode == 'min':
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_task_begin(self, trainer, task_id, **kwargs):
        self._reset()

    def on_epoch_end(self, trainer, task_id, epoch, total_epochs, metrics, **kwargs):
        value = metrics.get(self.monitor)
        if value is None:
            return
        if self._is_improvement(value):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"  [EarlyStopping] No improvement in '{self.monitor}' "
                      f"for {self.patience} epochs — stopping task early.")
                trainer.stop_training = True


class ModelCheckpointing(Callback):
    """Save model checkpoints; optionally save best and per-task checkpoints."""

    def __init__(self, checkpoint_dir, monitor='accuracy', mode='max',
                 save_best_only=True, save_task_checkpoints=True):
        self.checkpoint_dir       = Path(checkpoint_dir)
        self.monitor              = monitor
        self.mode                 = mode
        self.save_best_only       = save_best_only
        self.save_task_checkpoints = save_task_checkpoints
        self.best_value           = None

    def _is_improvement(self, value):
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == 'max' else value < self.best_value

    def _save(self, trainer, filename):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        state = {'model_state': trainer.model.state_dict(), 'results': trainer.results}
        torch.save(state, path)
        return path

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        acc_matrix = trainer.results.get('accuracy_matrix')
        if acc_matrix is None:
            return

        if self.monitor == 'accuracy':
            value = float(task_metrics['accuracy'])
        else:
            value = task_metrics.get(self.monitor, 0.0)

        if not self.save_best_only or self._is_improvement(value):
            self.best_value = value
            path = self._save(trainer, 'best_checkpoint.pt')
            print(f"  [Checkpoint] Saved best ({self.monitor}={value:.2f}) -> {path}")

        if self.save_task_checkpoints:
            self._save(trainer, f'checkpoint_task_{task_id}.pt')


class GradientClipping(Callback):
    """Clip gradient norms on on_before_step (after backward, before step)."""

    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm  = max_norm
        self.norm_type = norm_type

    def on_before_step(self, trainer, task_id, epoch, batch_idx, loss, **kwargs):
        params = [p for p in trainer.model.parameters() if p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.max_norm, self.norm_type)


class MetricLogger(Callback):
    """Log epoch-level metrics to console, CSV, W&B, and/or TensorBoard."""

    def __init__(self, log_to='console', csv_path=None, wandb_project=None,
                 tensorboard_dir=None, log_every_n_epochs=1):
        if isinstance(log_to, str):
            log_to = [log_to]
        self.log_to               = log_to
        self.csv_path             = csv_path
        self.wandb_project        = wandb_project
        self.tensorboard_dir      = tensorboard_dir
        self.log_every_n_epochs   = log_every_n_epochs

        self._csv_file   = None
        self._csv_writer = None
        self._tb_writer  = None
        self._wandb      = None
        self._step       = 0

    def on_train_begin(self, trainer, **kwargs):
        if 'csv' in self.log_to and self.csv_path:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            self._csv_file   = open(self.csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['task_id', 'epoch', 'loss', 'accuracy', 'lr'])

        if 'tensorboard' in self.log_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(self.tensorboard_dir or 'runs/cl_toolkit')
            except ImportError:
                print("  [MetricLogger] TensorBoard not available (pip install tensorboard)")

        if 'wandb' in self.log_to:
            try:
                import wandb
                wandb.init(project=self.wandb_project or 'cl_toolkit')
                self._wandb = wandb
            except ImportError:
                print("  [MetricLogger] W&B not available (pip install wandb)")

    def on_epoch_end(self, trainer, task_id, epoch, total_epochs, metrics, **kwargs):
        if epoch % self.log_every_n_epochs != 0:
            return

        loss     = metrics.get('loss', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        lr       = metrics.get('lr', 0.0)
        self._step += 1

        if 'console' in self.log_to:
            print(f"  Task {task_id} | Epoch {epoch+1}/{total_epochs} | "
                  f"loss={loss:.4f}  acc={accuracy:.2f}%  lr={lr:.2e}")

        if self._csv_writer:
            self._csv_writer.writerow([task_id, epoch, f"{loss:.4f}",
                                       f"{accuracy:.2f}", f"{lr:.2e}"])
            self._csv_file.flush()

        if self._tb_writer:
            self._tb_writer.add_scalar(f'task_{task_id}/loss',     loss,     self._step)
            self._tb_writer.add_scalar(f'task_{task_id}/accuracy', accuracy, self._step)

        if self._wandb:
            self._wandb.log({'task_id': task_id, 'epoch': epoch,
                             'loss': loss, 'accuracy': accuracy}, step=self._step)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._csv_file:
            self._csv_file.close()
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()


class ProgressBar(Callback):
    """Task-level tqdm progress bar showing accuracy after each task."""

    def __init__(self):
        self._pbar      = None
        self._num_tasks = None

    def on_train_begin(self, trainer, **kwargs):
        try:
            from tqdm import tqdm as _tqdm
            self._tqdm = _tqdm
        except ImportError:
            print("  [ProgressBar] tqdm not available (pip install tqdm)")
            self._tqdm = None

    def on_task_begin(self, trainer, task_id, **kwargs):
        if self._tqdm is None:
            return
        if self._pbar is None:
            num_tasks = trainer.scenario.num_tasks
            self._pbar = self._tqdm(total=num_tasks, desc='Tasks', unit='task',
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        if self._pbar is None:
            return
        acc = task_metrics['accuracy']
        self._pbar.set_postfix({'acc': f'{acc:.1f}%'})
        self._pbar.update(1)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

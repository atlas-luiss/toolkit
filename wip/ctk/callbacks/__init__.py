"""CTK callback system; all callbacks inherit from Callback."""

from .progressive_results import Callback, ProgressiveResultsCallback
from .builtin import (
    EarlyStopping,
    ModelCheckpointing,
    LRSchedulerCallback,
    WarmupScheduler,
    GradientClipping,
    MetricLogger,
    ProgressBar,
)
from .continual import AdapterMergeCallback, TaskMetricsCallback


class CallbackList:
    """Dispatches hook calls to a list of callbacks."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def _fire(self, hook, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(*args, **kwargs)

    def on_train_begin(self, trainer, **kwargs):
        self._fire('on_train_begin', trainer, **kwargs)

    def on_train_end(self, trainer, *args, **kwargs):
        self._fire('on_train_end', trainer, *args, **kwargs)

    def on_task_begin(self, trainer, task_id, **kwargs):
        self._fire('on_task_begin', trainer, task_id, **kwargs)

    def on_task_end(self, trainer, task_id, task_metrics, **kwargs):
        self._fire('on_task_end', trainer, task_id, task_metrics, **kwargs)

    def on_epoch_begin(self, trainer, task_id, epoch, total_epochs, **kwargs):
        self._fire('on_epoch_begin', trainer, task_id, epoch, total_epochs, **kwargs)

    def on_epoch_end(self, trainer, task_id, epoch, total_epochs, metrics, **kwargs):
        self._fire('on_epoch_end', trainer, task_id, epoch, total_epochs, metrics, **kwargs)

    def on_batch_begin(self, trainer, task_id, epoch, batch_idx, **kwargs):
        self._fire('on_batch_begin', trainer, task_id, epoch, batch_idx, **kwargs)

    def on_forward_end(self, trainer, task_id, epoch, batch_idx, logits, **kwargs):
        self._fire('on_forward_end', trainer, task_id, epoch, batch_idx, logits, **kwargs)

    def on_before_step(self, trainer, task_id, epoch, batch_idx, loss, **kwargs):
        self._fire('on_before_step', trainer, task_id, epoch, batch_idx, loss, **kwargs)

    def on_batch_end(self, trainer, task_id, epoch, batch_idx, loss, **kwargs):
        self._fire('on_batch_end', trainer, task_id, epoch, batch_idx, loss, **kwargs)

    def on_eval_begin(self, trainer, task_id, eval_task_id, **kwargs):
        self._fire('on_eval_begin', trainer, task_id, eval_task_id, **kwargs)

    def on_eval_end(self, trainer, task_id, eval_task_id, metrics, **kwargs):
        self._fire('on_eval_end', trainer, task_id, eval_task_id, metrics, **kwargs)


__all__ = [
    # Base
    'Callback',
    'CallbackList',
    # Display
    'ProgressiveResultsCallback',
    # Standard
    'EarlyStopping',
    'ModelCheckpointing',
    'LRSchedulerCallback',
    'WarmupScheduler',
    'GradientClipping',
    'MetricLogger',
    'ProgressBar',
    # Continual-learning specific
    'AdapterMergeCallback',
    'TaskMetricsCallback',
]

"""Callbacks: observe or steer the training loop without touching it.

Subclass :class:`Callback` and override any hook; every hook receives the
trainer, so you can read ``trainer.model``, ``trainer.optimizer``,
``trainer.results`` or set ``trainer.stop_training = True``.
"""


class Callback:
    def on_train_begin(self, trainer, **kw): ...
    def on_train_end(self, trainer, results=None, **kw): ...
    def on_task_begin(self, trainer, task_id, **kw): ...
    def on_task_end(self, trainer, task_id, task_metrics, **kw): ...
    def on_epoch_begin(self, trainer, task_id, epoch, total_epochs, **kw): ...
    def on_epoch_end(self, trainer, task_id, epoch, total_epochs, metrics, **kw): ...
    def on_batch_begin(self, trainer, task_id, epoch, batch_idx, **kw): ...
    def on_before_step(self, trainer, task_id, epoch, batch_idx, loss, **kw): ...
    def on_batch_end(self, trainer, task_id, epoch, batch_idx, loss, **kw): ...
    def on_eval_begin(self, trainer, task_id, eval_task_id, **kw): ...
    def on_eval_end(self, trainer, task_id, eval_task_id, metrics, **kw): ...


class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])

    def __getattr__(self, hook):
        if not hook.startswith("on_"):
            raise AttributeError(hook)

        def fire(*a, **kw):
            for cb in self.callbacks:
                fn = getattr(cb, hook, None)
                if fn is not None:
                    fn(*a, **kw)
        return fire


from .builtin import EarlyStopping, ModelCheckpointing, GradientClipping, MetricLogger, ProgressBar  # noqa: E402

__all__ = ["Callback", "CallbackList", "EarlyStopping", "ModelCheckpointing", "GradientClipping",
           "MetricLogger", "ProgressBar"]

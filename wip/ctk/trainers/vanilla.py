"""StandardTrainer — single-dataset supervised training, no CL required."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type

from ctk.callbacks import CallbackList


class StandardTrainer:
    """Standard trainer for a single dataset — no continual learning required.


    Usage::

        trainer = VanillaTrainer(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=20,
            lr=1e-3,
            callbacks=[EarlyStopping(patience=3), MetricLogger()],
        )
        results = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_classes: Optional[int] = None,
        epochs: int = 10,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn=None,
        callbacks=None,
        device=None,
        verbose: bool = True,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.epochs       = epochs
        self.optimizer_cls = optimizer_cls
        self.lr           = lr
        self.weight_decay = weight_decay
        self.loss_fn      = loss_fn or F.cross_entropy
        self.verbose      = verbose

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        if callbacks is None:
            self.callbacks = CallbackList([])
        elif isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(list(callbacks))

        # Detect CLModel so forward can be called with task_id=0
        from ctk.models.base import CLModel
        self._is_cl_model = isinstance(model, CLModel)

        # Auto-attach a head when num_classes is given and model has no head yet
        if self._is_cl_model and num_classes is not None and model._head is None:
            from ctk.models.base import IncrementalClassifier
            head = IncrementalClassifier(model.embed_dim)
            head.grow(num_classes)
            model.set_head(head)

        # Mirrors ContinualTrainer so callbacks that inspect trainer.strategy work
        self.strategy      = self
        self.stop_training = False
        self.stop_epoch    = False

        self.results = {
            'train_loss':     [],
            'train_accuracy': [],
            'val_loss':       [],
            'val_accuracy':   [],
        }

        # Build optimizer now — all model params, no extra_params()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(trainable, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run the training loop and return the results dict."""
        model = self.model.to(self.device)

        self.callbacks.on_train_begin(self)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"VanillaTrainer — {self.epochs} epochs, lr={self.lr}")
            print(f"{'='*60}")

        # Reuse task_id=0 so all CL callbacks (which expect a task_id) still work
        TASK_ID = 0
        self.callbacks.on_task_begin(self, TASK_ID)

        for epoch in range(self.epochs):
            self.callbacks.on_epoch_begin(self, TASK_ID, epoch, self.epochs)

            train_metrics = self._run_epoch(epoch, TASK_ID)
            self.results['train_loss'].append(train_metrics['loss'])
            self.results['train_accuracy'].append(train_metrics['accuracy'])

            metrics = dict(train_metrics)

            if self.val_loader is not None:
                val_metrics = self._evaluate(self.val_loader)
                self.results['val_loss'].append(val_metrics['loss'])
                self.results['val_accuracy'].append(val_metrics['accuracy'])
                metrics['val_loss']     = val_metrics['loss']
                metrics['val_accuracy'] = val_metrics['accuracy']

            self.callbacks.on_epoch_end(self, TASK_ID, epoch, self.epochs, metrics)

            if self.verbose:
                val_str = ''
                if self.val_loader is not None:
                    val_str = (f"  |  val_loss={metrics['val_loss']:.4f}"
                               f"  val_acc={metrics['val_accuracy']:.1f}%")
                print(f"  Epoch {epoch + 1}/{self.epochs}"
                      f"  loss={metrics['loss']:.4f}"
                      f"  acc={metrics['accuracy']:.1f}%"
                      f"{val_str}")

            if self.stop_training:
                self.stop_training = False
                if self.verbose:
                    print(f"  Early stop at epoch {epoch + 1}.")
                break

        task_metrics = {
            'accuracy': self.results['train_accuracy'][-1] if self.results['train_accuracy'] else 0.0,
            'loss':     self.results['train_loss'][-1]     if self.results['train_loss']     else 0.0,
        }
        self.callbacks.on_task_end(self, TASK_ID, task_metrics)
        self.callbacks.on_train_end(self, self.results)

        if self.verbose:
            best_acc = max(self.results['train_accuracy'], default=0.0)
            print(f"\n  Best train accuracy: {best_acc:.2f}%")
            if self.val_loader is not None:
                best_val = max(self.results['val_accuracy'], default=0.0)
                print(f"  Best val accuracy  : {best_val:.2f}%")

        return self.results

    # ------------------------------------------------------------------
    # Compatibility shims for callbacks that inspect trainer.strategy
    # ------------------------------------------------------------------

    def parameters(self):
        """Let GradientClipping and similar callbacks call trainer.strategy.parameters()."""
        return self.model.parameters()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, task_id: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            self.callbacks.on_batch_begin(self, task_id, epoch, batch_idx)

            self.optimizer.zero_grad()
            logits = self.model(x, task_id) if self._is_cl_model else self.model(x)
            loss   = self.loss_fn(logits, y)
            loss.backward()

            self.callbacks.on_before_step(self, task_id, epoch, batch_idx, loss)
            self.optimizer.step()

            total_loss += loss.item()
            correct    += (logits.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

            self.callbacks.on_batch_end(self, task_id, epoch, batch_idx, loss)

            if self.stop_epoch:
                self.stop_epoch = False
                break

        n = max(len(self.train_loader), 1)
        return {
            'loss':     total_loss / n,
            'accuracy': 100.0 * correct / max(total, 1),
            'lr':       self.optimizer.param_groups[0]['lr'],
        }

    def _evaluate(self, loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits      = self.model(x, 0) if self._is_cl_model else self.model(x)
                total_loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        n = max(total, 1)
        return {
            'loss':     total_loss / n,
            'accuracy': 100.0 * correct / n,
        }


# Backward-compatible alias
VanillaTrainer = StandardTrainer

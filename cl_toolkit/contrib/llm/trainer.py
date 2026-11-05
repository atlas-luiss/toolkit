"""LLMTrainer — single-task LoRA fine-tuning loop for language models."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LLMTrainingConfig:
    """Hyperparameters for LLMTrainer."""

    output_dir: str                     = "output"
    num_epochs: int                     = 3
    batch_size: int                     = 8
    eval_batch_size: int                = 16
    gradient_accumulation_steps: int    = 1

    # Optimizer
    lr: float               = 2e-4
    weight_decay: float     = 0.01
    max_grad_norm: float    = 1.0
    warmup_ratio: float     = 0.05     # fraction of total steps for linear warmup

    # Precision
    bf16: bool  = True
    fp16: bool  = False

    # Regularization
    gradient_checkpointing: bool = True

    # Logging / saving
    logging_steps: int  = 50
    eval_steps: int     = 500
    save_steps: int     = 500

    # Data
    max_seq_len: int            = 512
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


class LLMTrainer:
    """Fine-tunes an LLMModel on a single downstream task.

    Supports causal LM, seq2seq, and classification; handles mixed precision,
    gradient accumulation, LR warmup, periodic eval, and checkpoint saving.
    """

    def __init__(
        self,
        model,                              # LLMModel
        config: LLMTrainingConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader]   = None,
        compute_metrics: Optional[Callable] = None,  # fn(eval_loader, model) -> dict
    ):
        self.model          = model
        self.config         = config
        self.train_loader   = train_loader
        self.eval_loader    = eval_loader
        self.compute_metrics = compute_metrics

        self.device = model.device
        self._use_amp = (config.bf16 or config.fp16) and torch.cuda.is_available()
        self._amp_dtype = torch.bfloat16 if config.bf16 else torch.float16

        if config.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Run the full training loop. Returns a results dict."""
        cfg     = self.config
        model   = self.model
        model.to(self.device)

        optimizer = AdamW(
            model.trainable_params(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = math.ceil(len(self.train_loader) / cfg.gradient_accumulation_steps)
        total_steps     = steps_per_epoch * cfg.num_epochs
        warmup_steps    = int(total_steps * cfg.warmup_ratio)
        scheduler       = self._make_scheduler(optimizer, warmup_steps, total_steps)

        scaler = torch.cuda.amp.GradScaler() if (self._use_amp and cfg.fp16) else None

        global_step = 0
        log_losses: List[float] = []
        history: List[Dict] = []
        t0 = time.time()

        logger.info(
            f"LLMTrainer: {cfg.num_epochs} epochs  "
            f"{total_steps} total steps  warmup={warmup_steps}"
        )
        print(f"\n{'='*70}")
        print(f"LLMTrainer — {cfg.num_epochs} epochs, {total_steps} steps, lr={cfg.lr}")
        print(f"{'='*70}")

        for epoch in range(cfg.num_epochs):
            model.train()
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self._use_amp:
                    with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                        loss = model(**batch).loss / cfg.gradient_accumulation_steps
                else:
                    loss = model(**batch).loss / cfg.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                log_losses.append(loss.item() * cfg.gradient_accumulation_steps)

                is_accum_step = (step + 1) % cfg.gradient_accumulation_steps == 0
                is_last_step  = (step + 1) == len(self.train_loader)

                if is_accum_step or is_last_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_params(), cfg.max_grad_norm
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % cfg.logging_steps == 0:
                        avg_loss = float(np.mean(log_losses[-cfg.logging_steps:]))
                        lr_now   = scheduler.get_last_lr()[0]
                        elapsed  = time.time() - t0
                        print(
                            f"  step {global_step:5d}/{total_steps}"
                            f"  loss={avg_loss:.4f}"
                            f"  lr={lr_now:.2e}"
                            f"  [{elapsed:.0f}s]"
                        )
                        history.append({"step": global_step, "loss": avg_loss, "lr": lr_now})

                    # Eval
                    if self.eval_loader and global_step % cfg.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._print_eval(global_step, eval_metrics)
                        history[-1].update(eval_metrics)
                        model.train()

                    # Save
                    if global_step % cfg.save_steps == 0:
                        self._save_checkpoint(global_step)

            print(f"  Epoch {epoch + 1}/{cfg.num_epochs} done")

        # Final eval
        final_metrics = {}
        if self.eval_loader:
            final_metrics = self.evaluate()
            self._print_eval("final", final_metrics)

        self._save_checkpoint("final")

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.0f}s")
        print(f"{'='*70}")

        return {
            "history":       history,
            "final_metrics": final_metrics,
            "total_steps":   global_step,
            "elapsed":       elapsed,
        }

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict:
        """Evaluate on eval_loader (or given loader). Returns metrics dict."""
        loader = loader or self.eval_loader
        if loader is None:
            return {}

        # Custom metrics function takes priority
        if self.compute_metrics is not None:
            return self.compute_metrics(loader, self.model)

        return self._default_evaluate(loader)

    def save(self, path: str) -> None:
        """Save model and tokenizer to path (no-op if model has no save_pretrained)."""
        os.makedirs(path, exist_ok=True)
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "save_pretrained"):
            inner.save_pretrained(path)
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(path)
        logger.info(f"Saved to {path}")

    # ------------------------------------------------------------------
    # Default evaluation
    # ------------------------------------------------------------------

    def _default_evaluate(self, loader: DataLoader) -> Dict:
        """Compute loss (and accuracy for classification) over loader."""
        model  = self.model
        device = self.device
        model.eval()

        total_loss  = 0.0
        n_batches   = 0
        correct     = 0
        total       = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if self._use_amp:
                    with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    n_batches  += 1

                # Accuracy for classification
                if model.task_type == "classification" and hasattr(outputs, "logits"):
                    preds    = outputs.logits.argmax(-1)
                    labels   = batch.get("labels")
                    if labels is not None:
                        correct += (preds == labels).sum().item()
                        total   += labels.size(0)

        metrics: Dict = {}
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            metrics["eval_loss"] = avg_loss
            # Perplexity for language model tasks
            if model.task_type in ("causal_lm", "seq2seq_lm"):
                try:
                    metrics["perplexity"] = math.exp(avg_loss)
                except OverflowError:
                    metrics["perplexity"] = float("inf")

        if total > 0:
            metrics["accuracy"] = 100.0 * correct / total

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
        """Linear warmup → cosine decay scheduler."""
        try:
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        except ImportError:
            pass

        # Fallback: linear decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, step) -> None:
        path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        self.save(path)

    def _print_eval(self, step, metrics: Dict) -> None:
        parts = [f"step={step}"]
        parts += [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(f"  [eval] {' | '.join(parts)}")

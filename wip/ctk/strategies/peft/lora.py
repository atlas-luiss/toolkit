"""LoRAAdapter — wraps any Strategy with per-task LoRA adapter lifecycle."""

from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctk.strategies.base import Strategy


class LoRAAdapter(Strategy):
    """
    Wraps any CL strategy with a per-task LoRA adapter lifecycle.

    The inner strategy handles the loss (what to optimise and how to prevent
    forgetting).  LoRAAdapter handles the adapter lifecycle (when to attach,
    freeze, or merge adapters) independently.

    Usage::

        # LoRA + plain cross-entropy (equivalent to old LoRAStrategy)
        strategy = LoRAAdapter(NaiveStrategy(), rank=8)

        # LoRA + experience replay
        strategy = LoRAAdapter(ExperienceReplay(memory_size=500), rank=8)

        # LoRA + EWC
        strategy = LoRAAdapter(EWC(lam=500), rank=8)

    Customising the adapter lifecycle::

        class SharedAdapterStrategy(LoRAAdapter):
            \"\"\"One shared adapter across all tasks, never frozen.\"\"\"

            def on_lora_task_start(self, task_id, model):
                if task_id == 0:
                    model.freeze_backbone()
                    model.add_lora(0, rank=self.rank,
                                   lora_alpha=self.lora_alpha,
                                   lora_dropout=self.lora_dropout,
                                   target_modules=self.target_modules)

            def on_lora_task_end(self, task_id, model):
                pass   # keep adapter trainable across tasks
    """

    def __init__(
        self,
        strategy: Strategy,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        merge_after_task: bool = False,
    ):
        self.strategy         = strategy
        self.rank             = rank
        self.lora_alpha       = lora_alpha
        self.lora_dropout     = lora_dropout
        self.target_modules   = target_modules
        self.merge_after_task = merge_after_task

    # ------------------------------------------------------------------
    # Overridable lifecycle hooks
    # ------------------------------------------------------------------

    def on_lora_task_start(self, task_id: int, model) -> None:
        """Attach a new LoRA adapter for task_id and freeze the backbone.
        Override to change when/how adapters are created."""
        model.freeze_backbone()
        model.add_lora(
            task_id,
            rank           = self.rank,
            lora_alpha     = self.lora_alpha,
            lora_dropout   = self.lora_dropout,
            target_modules = self.target_modules,
        )

    def on_lora_task_end(self, task_id: int, model) -> None:
        """Freeze or merge the adapter after training on task_id.
        Override to change post-task adapter behaviour."""
        if self.merge_after_task:
            model.merge_lora(task_id)
        else:
            model.freeze_lora(task_id)

    # ------------------------------------------------------------------
    # Strategy interface — delegates everything to inner strategy
    # ------------------------------------------------------------------

    def on_task_start(self, task_id, train_loader, model) -> None:
        self.on_lora_task_start(task_id, model)
        self.strategy.on_task_start(task_id, train_loader, model)

    def compute_loss(self, batch, model, task_id) -> torch.Tensor:
        return self.strategy.compute_loss(batch, model, task_id)

    def before_update(self, model, optimizer) -> None:
        self.strategy.before_update(model, optimizer)

    def on_task_end(self, task_id, train_loader, model) -> None:
        self.strategy.on_task_end(task_id, train_loader, model)
        self.on_lora_task_end(task_id, model)

    def extra_params(self) -> List[nn.Parameter]:
        return self.strategy.extra_params()

    def evaluate_task(self, test_loader, model, task_id) -> Optional[Dict]:
        return self.strategy.evaluate_task(test_loader, model, task_id)


# ---------------------------------------------------------------------------
# Convenience shorthands — keep old names working
# ---------------------------------------------------------------------------

def LoRAStrategy(
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules=None,
    merge_after_task: bool = False,
) -> LoRAAdapter:
    """One LoRA adapter per task with plain cross-entropy loss.
    Shorthand for LoRAAdapter(NaiveStrategy(), ...)."""
    from ctk.strategies.naive import NaiveStrategy
    return LoRAAdapter(
        NaiveStrategy(),
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        merge_after_task=merge_after_task,
    )


def LoRAReplay(
    memory_size: int = 200,
    replay_batch_ratio: float = 0.5,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules=None,
    merge_after_task: bool = False,
) -> LoRAAdapter:
    """LoRA adapter per task + experience replay buffer.
    Shorthand for LoRAAdapter(ExperienceReplay(...), ...)."""
    from ctk.strategies.replay.er import ExperienceReplay
    return LoRAAdapter(
        ExperienceReplay(memory_size=memory_size,
                         replay_batch_ratio=replay_batch_ratio),
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        merge_after_task=merge_after_task,
    )

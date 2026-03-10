"""LoRA-per-task strategy: one frozen LoRA adapter per task, selected at inference."""

import torch.nn.functional as F
from typing import Optional, List

from ctk.strategies.base import Strategy


class LoRAStrategy(Strategy):
    """One LoRA adapter per task; freeze backbone, train adapter + head, freeze after task."""

    def __init__(
        self,
        rank:             int            = 8,
        target_modules:   Optional[List] = None,
        lora_alpha:       float          = 16.0,
        lora_dropout:     float          = 0.0,
        merge_after_task: bool           = False,
    ):
        self.rank             = rank
        self.target_modules   = target_modules
        self.lora_alpha       = lora_alpha
        self.lora_dropout     = lora_dropout
        self.merge_after_task = merge_after_task

    def on_task_start(self, task_id, train_loader, model):
        model.freeze_backbone()
        model.add_lora(
            task_id,
            rank           = self.rank,
            target_modules = self.target_modules,
            lora_alpha     = self.lora_alpha,
            lora_dropout   = self.lora_dropout,
        )

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        return F.cross_entropy(model(x, task_id), y)

    def on_task_end(self, task_id, train_loader, model):
        if self.merge_after_task:
            model.merge_lora(task_id)
        else:
            model.freeze_lora(task_id)

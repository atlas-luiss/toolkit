"""LoRA + Experience Replay: LoRA adapter per task with reservoir replay."""

import torch.nn.functional as F

from ctk.strategies.peft.lora import LoRAStrategy
from ctk.strategies.replay.buffer import ReservoirBuffer


class LoRAReplay(LoRAStrategy):
    """One LoRA adapter per task (frozen backbone) + reservoir replay between tasks."""

    def __init__(
        self,
        rank:             int   = 8,
        lora_alpha:       float = 16.0,
        lora_dropout:     float = 0.0,
        memory_size:      int   = 200,
        replay_ratio:     float = 0.5,
        merge_after_task: bool  = False,
    ):
        super().__init__(
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_after_task=merge_after_task,
        )
        self.memory_size  = memory_size
        self.replay_ratio = replay_ratio
        self.buffer: ReservoirBuffer | None = None

    def on_task_start(self, task_id, train_loader, model):
        super().on_task_start(task_id, train_loader, model)
        if self.buffer is None:
            self.buffer = ReservoirBuffer(self.memory_size, model._device)

    def compute_loss(self, batch, model, task_id):
        x, y = batch[0].to(model._device), batch[1].to(model._device)
        self.buffer.add_batch(x.detach(), y.detach(), task_id)
        loss = F.cross_entropy(model(x, task_id), y)

        n_replay = max(1, int(len(x) * self.replay_ratio))
        x_m, y_m, t_m, _ = self.buffer.sample(n_replay)
        if x_m is not None:
            x_m = x_m.to(model._device)
            y_m = y_m.to(model._device)
            loss = loss + F.cross_entropy(model(x_m, task_id), y_m)

        return loss

"""LoRAAdapter: give any strategy a per-task LoRA adapter lifecycle."""

from typing import List, Optional

from cl_toolkit import adapters
from cl_toolkit.strategies.base import Strategy


class LoRAAdapter(Strategy):
    """Wraps an inner strategy (the loss) with one LoRA adapter per task.

    At task t a new adapter ``task_t`` is added and trained; older adapters stay
    *active but frozen*, so the backbone is base + sum of all task deltas.
    Evaluation is identical in every scenario: all adapters active. (There is no
    adapter selection by task id, which would leak task identity in Class-IL.)

    ``merge_after_task=True`` folds each adapter into the base weights after its
    task (sequential merging: constant memory, same function).

    ``targets`` / ``layers`` choose where LoRA goes (see
    :func:`cl_toolkit.adapters.resolve_lora_targets`), e.g.
    ``targets=["qkv", "fc1"], layers=[9, 10, 11]``. Default: the model's
    ``lora_targets`` / ``lora_layers`` (ViT: ``qkv`` in every block).

    Examples::

        LoRAAdapter(NaiveStrategy(), rank=8)
        LoRAAdapter(ExperienceReplay(500), rank=8)
    """

    def __init__(self, strategy: Strategy, rank: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 targets=None, layers: Optional[List[int]] = None, merge_after_task: bool = False,
                 target_modules=None):
        self.strategy, self.rank, self.lora_alpha, self.lora_dropout = strategy, rank, lora_alpha, lora_dropout
        self.targets = targets if targets is not None else target_modules   # None -> model default
        self.layers, self.merge_after_task = layers, merge_after_task
        self.scenarios, self.train_mask, self.eval_mask = strategy.scenarios, strategy.train_mask, strategy.eval_mask

    def bind(self, trainer):
        super().bind(trainer)
        self.strategy.bind(trainer)

    def on_task_start(self, task_id, train_loader, model):
        model.freeze_backbone()
        mods = adapters.add_lora(model, f"task_{task_id}", self.rank, self.lora_alpha, self.lora_dropout,
                                 targets=self.targets, layers=self.layers)
        if task_id == 0 and self.trainer is not None:
            self.trainer.log(f"  LoRA on {len(mods)} modules, e.g. {mods[:3]}")
        self.strategy.task = self.task
        self.strategy.on_task_start(task_id, train_loader, model)

    def wants_training(self, task_id):
        return self.strategy.wants_training(task_id)

    def compute_loss(self, batch, model, task_id):
        return self.strategy.compute_loss(batch, model, task_id)

    def before_update(self, model, optimizer):
        self.strategy.before_update(model, optimizer)

    def on_task_end(self, task_id, train_loader, model):
        self.strategy.on_task_end(task_id, train_loader, model)
        if self.merge_after_task:
            adapters.merge_lora(model, f"task_{task_id}")
        else:
            adapters.freeze_lora(model, f"task_{task_id}")

    def before_eval(self, model, task_id):
        self.strategy.before_eval(model, task_id)

    def evaluate_task(self, test_loader, model, task_id):
        return self.strategy.evaluate_task(test_loader, model, task_id)

    def extra_params(self):
        return self.strategy.extra_params()

    def __repr__(self):
        return (f"LoRA(r={self.rank}, targets={self.targets or 'default'}, layers={self.layers or 'all'}, "
                f"merge={self.merge_after_task}) + {self.strategy!r}")


def LoRAStrategy(rank=8, lora_alpha=16.0, lora_dropout=0.0, targets=None, layers=None, merge_after_task=False,
                 target_modules=None):
    from cl_toolkit.strategies.naive import NaiveStrategy
    return LoRAAdapter(NaiveStrategy(), rank, lora_alpha, lora_dropout, targets, layers, merge_after_task,
                       target_modules)


def LoRAReplay(memory_size=200, replay_batch_ratio=1.0, rank=8, lora_alpha=16.0, lora_dropout=0.0,
               targets=None, layers=None, merge_after_task=False, target_modules=None):
    from cl_toolkit.strategies.replay.er import ExperienceReplay
    return LoRAAdapter(ExperienceReplay(memory_size, replay_batch_ratio), rank, lora_alpha, lora_dropout,
                       targets, layers, merge_after_task, target_modules)

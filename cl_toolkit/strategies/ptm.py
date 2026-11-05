"""Pre-trained-model baselines built from a backbone policy + a classifier head.

* :class:`FirstSessionAdaptation` — the backbone is frozen; optionally a PEFT
  module (AdaptFormer or LoRA) is trained on task 0 only, then frozen too
  (APER, Zhou et al. IJCV 2024; RanPAC, McDonnell et al. NeurIPS 2023).
* :func:`SimpleCIL` = FirstSessionAdaptation + :class:`~cl_toolkit.heads.NCM`
* :func:`RanPAC`    = FirstSessionAdaptation + :class:`~cl_toolkit.heads.RanPAC`

Any other head works the same way: ``WithHead(FirstSessionAdaptation(), FeCAM())``.
"""

from __future__ import annotations

from typing import Optional

from cl_toolkit import adapters, heads
from cl_toolkit.strategies.base import Strategy
from cl_toolkit.strategies.with_head import WithHead


class FirstSessionAdaptation(Strategy):
    """Train only on task 0 (with an optional PEFT module); everything is frozen afterwards.

    ``first_session``: None (no training at all), "adaptformer" or "lora".
    """

    def __init__(self, first_session: Optional[str] = None, bottleneck: int = 64, lora_rank: int = 8,
                 lora_targets=None, lora_layers=None):
        if first_session not in (None, "adaptformer", "lora"):
            raise ValueError("first_session must be None, 'adaptformer' or 'lora'")
        self.first_session, self.bottleneck, self.lora_rank = first_session, bottleneck, lora_rank
        self.lora_targets, self.lora_layers = lora_targets, lora_layers

    def on_task_start(self, task_id, train_loader, model):
        model.freeze_backbone()
        if task_id == 0 and self.first_session == "adaptformer":
            adapters.add_adaptformer(model, self.bottleneck)
        elif task_id == 0 and self.first_session == "lora":
            adapters.add_lora(model, "first_session", rank=self.lora_rank, targets=self.lora_targets,
                              layers=self.lora_layers)
        if task_id > 0:
            for p in model.parameters():
                p.requires_grad_(False)

    def wants_training(self, task_id):
        return task_id == 0 and self.first_session is not None

    def __repr__(self):
        if self.first_session is None:
            return "Frozen()"
        extra = f"bottleneck={self.bottleneck}" if self.first_session == "adaptformer" else \
            f"lora_rank={self.lora_rank}, targets={self.lora_targets or 'default'}"
        return f"FirstSessionAdaptation({self.first_session}, {extra})"

    def on_task_end(self, task_id, train_loader, model):
        if task_id == 0:
            for p in model.parameters():
                p.requires_grad_(False)


def SimpleCIL(first_session: Optional[str] = None, metric: str = "cosine", **kw) -> WithHead:
    """Prototype classifier on a frozen PTM (Zhou et al., IJCV 2024). With
    ``first_session="adaptformer"`` this is APER's adapted-model variant."""
    return WithHead(FirstSessionAdaptation(first_session, **kw), heads.NCM(metric))


def RanPAC(proj_dim: int = 10000, first_session: Optional[str] = None, lambdas=None, seed: int = 0,
           **kw) -> WithHead:
    """RanPAC (McDonnell et al., NeurIPS 2023): random projection + ridge on (adapted) PTM features.

    The default (``first_session=None``) is RanPAC *without* its Phase 1 (first-session
    PETL adaptation), i.e. the paper's "no PETL" ablation. ``first_session="adaptformer"``
    adds Phase 1 (the paper's full method uses AdaptFormer / SSF / VPT).
    """
    return WithHead(FirstSessionAdaptation(first_session, **kw), heads.RanPAC(proj_dim, lambdas, seed))


ProtoNet = SimpleCIL  # old name

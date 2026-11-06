"""Naive sequential fine-tuning: the lower bound."""

from cl_toolkit.strategies.base import Strategy


class NaiveStrategy(Strategy):
    """Plain cross-entropy (+ model aux loss). No forgetting prevention.

    ``train_mask="current"`` restricts training logits to the current task's
    classes (used by prompt methods; also a strong cheap baseline with PTMs).
    """

    def __init__(self, train_mask: str = "seen", eval_mask: str = "seen"):
        self.train_mask, self.eval_mask = train_mask, eval_mask


Naive = NaiveStrategy

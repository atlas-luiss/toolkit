"""Class-Incremental Learning scenario: growing output space, no task oracle at inference."""

from __future__ import annotations

from ctk.models.base import CLModel

from .base import CLScenario


class ClassIncremental(CLScenario):
    """Class-IL: single IncrementalClassifier pre-allocated for all tasks; global labels."""

    def setup(self, model: CLModel) -> None:
        """Attach and pre-grow IncrementalClassifier to cover all task classes."""
        from ctk.models.base import IncrementalClassifier

        head = IncrementalClassifier(model.embed_dim, initial_classes=0)
        model.set_head(head)

        for split in self.benchmark.get_splits():
            model.grow_head(len(split.classes))

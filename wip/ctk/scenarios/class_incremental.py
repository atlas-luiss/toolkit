"""Class-Incremental Learning scenario: growing output space, no task oracle at inference."""

from __future__ import annotations

from ctk.models.base import CLModel

from .base import CLScenario


class ClassIncremental(CLScenario):
    """Class-IL: single IncrementalClassifier pre-allocated for all tasks; global labels."""

    def setup(self, model: CLModel) -> None:
        """Attach an IncrementalClassifier pre-grown to cover all benchmark classes.

        Growing to total_classes (not classes_per_task × num_tasks) ensures that
        original class IDs — which can be any value in [0, total_classes-1] — are
        always valid head indices regardless of how classes are shuffled across tasks.
        """
        from ctk.models.base import IncrementalClassifier

        head = IncrementalClassifier(model.embed_dim, initial_classes=0)
        model.set_head(head)
        model.grow_head(self.benchmark.total_classes)

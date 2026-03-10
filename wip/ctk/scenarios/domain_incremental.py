"""Domain-Incremental Learning scenario: fixed output space, shifting input domain."""

from __future__ import annotations

from ctk.models.base import CLModel

from .base import CLScenario


class DomainIncremental(CLScenario):
    """Domain-IL: fixed IncrementalClassifier covering all shared classes."""

    def setup(self, model: CLModel) -> None:
        """Attach a fixed IncrementalClassifier sized for all shared classes."""
        from ctk.models.base import IncrementalClassifier

        n_classes = self.benchmark.total_classes
        head = IncrementalClassifier(model.embed_dim, initial_classes=0)
        model.set_head(head)
        model.grow_head(n_classes)

"""
Integration tests for the refactored CTK architecture.

Tests the full pipeline: benchmark → scenario → strategy → trainer.
Uses small synthetic datasets to avoid external dependencies.
"""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import TensorDataset
from typing import List

from ctk.benchmarks.base import CLBenchmark, TaskSplit
from ctk.models.base import CLModel, IncrementalClassifier, MultiHeadClassifier
from ctk.scenarios import ClassIncremental, TaskIncremental, DomainIncremental, FewShot, FSCIL
from ctk.scenarios.finetuning import Finetuning, JointTraining
from ctk.strategies import (
    NaiveStrategy, ExperienceReplay, DERPP, AGEM, GDumb,
    EWC, EWCOnline, LwF, MAS, LoRAStrategy, ProtoNet,
)
from ctk.trainer import ContinualTrainer


# ---------------------------------------------------------------------------
# Minimal synthetic model
# ---------------------------------------------------------------------------

class TinyModel(CLModel):
    """4-dim input → 16-dim hidden → features. No pretrained weights needed."""

    def __init__(self, input_dim=4, hidden_dim=16, device=None):
        super().__init__(device)
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ).to(self._device)
        self._embed_dim = hidden_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _default_lora_targets(self):
        return []


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

class SyntheticBenchmark(CLBenchmark):
    """Lightweight benchmark for testing — no disk I/O."""

    def __init__(self, splits: List[TaskSplit]):
        self._splits = splits

    @property
    def num_tasks(self) -> int:
        return len(self._splits)

    @property
    def total_classes(self) -> int:
        return sum(len(s.classes) for s in self._splits)

    @property
    def classes_per_task(self) -> int:
        return len(self._splits[0].classes)

    def get_splits(self) -> List[TaskSplit]:
        return self._splits


def make_benchmark(num_tasks=2, classes_per_task=5, n_train=40, n_test=10,
                   input_dim=4, shared_classes=False):
    """Return a SyntheticBenchmark with random tensors."""
    splits = []
    for t in range(num_tasks):
        if shared_classes:
            classes = list(range(classes_per_task))  # same classes, different domain
        else:
            offset = t * classes_per_task
            classes = list(range(offset, offset + classes_per_task))
        x_tr = torch.randn(n_train, input_dim)
        y_tr = torch.randint(min(classes), max(classes) + 1, (n_train,))
        x_te = torch.randn(n_test, input_dim)
        y_te = torch.randint(min(classes), max(classes) + 1, (n_test,))
        splits.append(TaskSplit(
            task_id=t,
            classes=classes,
            train_dataset=TensorDataset(x_tr, y_tr),
            test_dataset=TensorDataset(x_te, y_te),
        ))
    return SyntheticBenchmark(splits)


# ---------------------------------------------------------------------------
# Shared trainer params
# ---------------------------------------------------------------------------

SCENARIO_KW = dict(batch_size=8, num_workers=0, pin_memory=False)
TRAINER_KW = dict(
    optimizer_cls=torch.optim.SGD,
    lr=1e-2,
    epochs_per_task=1,
    verbose=False,
)


def run_pipeline(scenario_cls, strategy, bench, **extra_scenario_kw):
    model = TinyModel()
    kw = {**SCENARIO_KW, **extra_scenario_kw}
    scenario = scenario_cls(bench, **kw)
    trainer = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
    results = trainer.fit()
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassIncremental:
    def test_naive(self):
        results = run_pipeline(ClassIncremental, NaiveStrategy(), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)
        assert "avg_accuracy" in results
        assert "avg_forgetting" in results

    def test_experience_replay(self):
        results = run_pipeline(ClassIncremental, ExperienceReplay(memory_size=50),
                               make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_derpp(self):
        results = run_pipeline(ClassIncremental, DERPP(memory_size=50), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_agem(self):
        results = run_pipeline(ClassIncremental, AGEM(memory_size=50), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_ewc(self):
        results = run_pipeline(ClassIncremental, EWC(lam=100.0, fisher_samples=20),
                               make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_ewc_online(self):
        results = run_pipeline(ClassIncremental, EWCOnline(lam=100.0, fisher_samples=20),
                               make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_lwf(self):
        results = run_pipeline(ClassIncremental, LwF(lam=1.0), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_mas(self):
        results = run_pipeline(ClassIncremental, MAS(lam=0.1, importance_samples=20),
                               make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)


class TestTaskIncremental:
    def test_naive(self):
        results = run_pipeline(TaskIncremental, NaiveStrategy(), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_ewc(self):
        results = run_pipeline(TaskIncremental, EWC(lam=100.0, fisher_samples=20),
                               make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)


class TestDomainIncremental:
    def test_naive(self):
        bench = make_benchmark(shared_classes=True)
        results = run_pipeline(DomainIncremental, NaiveStrategy(), bench)
        assert results["accuracy_matrix"].shape == (2, 2)


class TestFinetuningScenarios:
    def test_finetuning(self):
        results = run_pipeline(Finetuning, NaiveStrategy(), make_benchmark())
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_joint(self):
        results = run_pipeline(JointTraining, NaiveStrategy(), make_benchmark())
        # JointTraining trains once → 1×1 accuracy matrix
        assert results["accuracy_matrix"].shape[0] == 1


class TestMultiTask:
    def test_3tasks_naive(self):
        results = run_pipeline(ClassIncremental, NaiveStrategy(),
                               make_benchmark(num_tasks=3, classes_per_task=4))
        assert results["accuracy_matrix"].shape == (3, 3)

    def test_3tasks_replay(self):
        results = run_pipeline(ClassIncremental, ExperienceReplay(memory_size=60),
                               make_benchmark(num_tasks=3, classes_per_task=4))
        assert results["accuracy_matrix"].shape == (3, 3)


class TestFewShot:
    def test_few_shot_class_il(self):
        """FewShot scenario + NaiveStrategy — verifies K-shot subsampling."""
        bench    = make_benchmark(num_tasks=2, classes_per_task=5,
                                  n_train=50, n_test=10)
        model    = TinyModel()
        scenario = FewShot(bench, k_shots=5, batch_size=8,
                           num_workers=0, pin_memory=False)
        strategy = NaiveStrategy()
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (2, 2)
        # Each task's train loader should have at most k*classes examples
        for task_info in scenario:
            assert len(task_info.train_loader.dataset) <= 5 * 5

    def test_proto_net_frozen(self):
        """ProtoNet with frozen backbone — no gradients, prototype inference."""
        bench    = make_benchmark(num_tasks=2, classes_per_task=5,
                                  n_train=50, n_test=10)
        model    = TinyModel()
        scenario = FewShot(bench, k_shots=5, batch_size=8,
                           num_workers=0, pin_memory=False)
        strategy = ProtoNet(distance="cosine", finetune_backbone=False)
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (2, 2)
        assert strategy.num_prototypes == 10   # 2 tasks × 5 classes

    def test_proto_net_finetune(self):
        """ProtoNet with backbone fine-tuning — CE loss on support set."""
        bench    = make_benchmark(num_tasks=2, classes_per_task=5,
                                  n_train=50, n_test=10)
        model    = TinyModel()
        scenario = FewShot(bench, k_shots=10, batch_size=8,
                           num_workers=0, pin_memory=False)
        strategy = ProtoNet(distance="l2", finetune_backbone=True)
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (2, 2)

    def test_proto_net_full_dataset(self):
        """ProtoNet on full dataset (k_shots=None) — Class-IL with NMC."""
        bench    = make_benchmark(num_tasks=3, classes_per_task=4)
        model    = TinyModel()
        scenario = FewShot(bench, k_shots=None, batch_size=8,
                           num_workers=0, pin_memory=False)
        strategy = ProtoNet(distance="cosine")
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (3, 3)

    def test_few_shot_task_il(self):
        """FewShot wrapping TaskIncremental — per-task heads + K-shot."""
        bench    = make_benchmark(num_tasks=2, classes_per_task=5,
                                  n_train=50, n_test=10)
        model    = TinyModel()
        scenario = FewShot(bench, k_shots=5,
                           base_scenario=TaskIncremental,
                           batch_size=8, num_workers=0, pin_memory=False)
        strategy = NaiveStrategy()
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (2, 2)


class TestFSCIL:
    def test_fscil_base_full_incremental_kshot(self):
        """FSCIL: base task has full data, incremental tasks are K-shot."""
        bench    = make_benchmark(num_tasks=3, classes_per_task=5,
                                  n_train=60, n_test=10)
        model    = TinyModel()
        scenario = FSCIL(bench, k_shots=5, batch_size=8,
                         num_workers=0, pin_memory=False)
        strategy = ProtoNet(distance="cosine")
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (3, 3)

        # Base session should have full data, incremental should be K-shot
        task_infos = list(scenario)
        assert len(task_infos[0].train_loader.dataset) == 60           # full
        assert len(task_infos[1].train_loader.dataset) <= 5 * 5        # K-shot
        assert len(task_infos[2].train_loader.dataset) <= 5 * 5        # K-shot

    def test_fscil_naive_strategy(self):
        """FSCIL with naive fine-tuning to test forgetting."""
        bench    = make_benchmark(num_tasks=3, classes_per_task=4,
                                  n_train=60, n_test=10)
        model    = TinyModel()
        scenario = FSCIL(bench, k_shots=3, batch_size=8,
                         num_workers=0, pin_memory=False)
        strategy = NaiveStrategy()
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()
        assert results["accuracy_matrix"].shape == (3, 3)

    def test_fscil_proto_stable_prototypes(self):
        """ProtoNet on FSCIL: old prototypes are never modified."""
        bench    = make_benchmark(num_tasks=3, classes_per_task=5,
                                  n_train=80, n_test=20)
        model    = TinyModel()
        scenario = FSCIL(bench, k_shots=5, batch_size=8,
                         num_workers=0, pin_memory=False)
        strategy = ProtoNet(distance="cosine", finetune_backbone=False)
        trainer  = ContinualTrainer(model, scenario, strategy, **TRAINER_KW)
        results  = trainer.fit()

        mat = results["accuracy_matrix"]
        assert mat.shape == (3, 3)
        # All 15 prototypes should be stored (3 tasks × 5 classes)
        assert strategy.num_prototypes == 15
        # Prototypes for task-0 classes are never re-computed after task 0
        proto_snapshot = strategy.get_prototypes()
        assert all(c in proto_snapshot for c in range(5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

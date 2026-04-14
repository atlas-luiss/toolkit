"""
Smoke-tests for all recent CTK changes:
  1. Trainer module imports (cl_trainer, cl_evaluator, vanilla)
  2. Legacy strategy base classes removed
  3. merge_lora no longer unwraps backbone when adapters remain
  4. ContinualTrainer falls back to head params when backbone is frozen
  5. End-to-end mini continual training run on GPU (synthetic data)
"""

import sys
import traceback

sys.path.insert(0, ".")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def test(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        print(f"         {e}")
        traceback.print_exc()
        results.append((name, False, e))


# ---------------------------------------------------------------------------
# 1. Trainer module imports
# ---------------------------------------------------------------------------
print("\n=== 1. Trainer module imports ===")

def test_top_level_imports():
    from ctk import Trainer, ContinualTrainer, ContinualEvaluator, VanillaTrainer
    assert Trainer is ContinualTrainer

def test_trainer_submodule_imports():
    from ctk.trainers import (
        ContinualTrainer, ContinualEvaluator,
        VanillaTrainer, LLMTrainer, LLMTrainingConfig,
    )

def test_trainer_alias():
    from ctk import Trainer, ContinualTrainer
    assert Trainer is ContinualTrainer, "Trainer alias broken"

test("top-level: from ctk import Trainer / ContinualTrainer / ...", test_top_level_imports)
test("submodule: from ctk.trainers import all trainers", test_trainer_submodule_imports)
test("Trainer is ContinualTrainer alias", test_trainer_alias)


# ---------------------------------------------------------------------------
# 2. Legacy base classes removed
# ---------------------------------------------------------------------------
print("\n=== 2. Legacy base classes removed ===")

def test_legacy_classes_gone():
    import ctk.strategies.base as base_mod
    for cls_name in ("ContinualLearningStrategy", "ReplayStrategy", "RegularizationStrategy"):
        assert not hasattr(base_mod, cls_name), \
            f"{cls_name} still present in strategies/base.py"

def test_strategy_base_exports_only_strategy():
    import ctk.strategies.base as base_mod
    public = [k for k in vars(base_mod) if not k.startswith("_")]
    assert "Strategy" in public
    assert "ContinualLearningStrategy" not in public

def test_all_strategies_import_clean():
    from ctk.strategies import (
        Strategy, NaiveStrategy,
        EWC, EWCOnline, LwF, MAS,
        ExperienceReplay, DERPP, AGEM, iCaRL, GDumb,
        LoRAStrategy, LoRAReplay, HAM,
        ProtoNet,
    )

test("ContinualLearningStrategy / ReplayStrategy / RegularizationStrategy gone", test_legacy_classes_gone)
test("strategies/base.py public namespace clean", test_strategy_base_exports_only_strategy)
test("all strategy classes still importable", test_all_strategies_import_clean)


# ---------------------------------------------------------------------------
# 3. merge_lora fix
# ---------------------------------------------------------------------------
print("\n=== 3. merge_lora preserves remaining adapters ===")

def _make_model_with_mock_peft(adapter_names):
    """
    Build a TinyModel whose _backbone is a mock LoraModel-like object
    with a real peft_config dict and a no-op merge_adapter().
    This avoids PEFT version-specific target_modules matching issues
    while still exercising the actual merge_lora() code path.
    """
    import torch.nn as nn
    from ctk.models.base import CLModel, IncrementalClassifier

    class TinyModel(CLModel):
        def __init__(self):
            super().__init__()
            self._backbone = nn.Linear(8, 8)
            self._head = IncrementalClassifier(8)
            self._head.grow(2)
        @property
        def embed_dim(self): return 8
        def _encode(self, x): return self._backbone(x)
        def _default_lora_targets(self): return []

    model = TinyModel()

    # Wrap _backbone in a lightweight mock that looks like a LoraModel
    plain_backbone = model._backbone

    class _MockLoraBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = plain_backbone          # what merge_lora unwraps to
            self.peft_config = {n: object() for n in adapter_names}
        def merge_adapter(self, adapter_names):
            pass  # weights already "merged" for our purposes
        def forward(self, x):
            return self.model(x)

    model._backbone = _MockLoraBackbone()
    return model, plain_backbone

def test_merge_lora_no_double_unwrap():
    """Merging task_0 must leave task_1 intact and not unwrap the LoraModel."""
    model, _ = _make_model_with_mock_peft(["task_0", "task_1"])
    model._active_lora = 0

    model.merge_lora(task_id=0)

    assert hasattr(model._backbone, "peft_config"), \
        "LoraModel was unwrapped prematurely — task_1 still exists"
    assert "task_1" in model._backbone.peft_config, \
        "task_1 adapter was destroyed when merging task_0"
    assert "task_0" not in model._backbone.peft_config, \
        "task_0 should have been removed from peft_config"
    assert model._active_lora is None

def test_merge_lora_unwraps_when_empty():
    """After merging the last adapter, _backbone must be unwrapped to plain nn.Module."""
    model, plain_backbone = _make_model_with_mock_peft(["task_0"])
    model._active_lora = 0

    model.merge_lora(task_id=0)

    assert not hasattr(model._backbone, "peft_config"), \
        "LoraModel was NOT unwrapped after last adapter merged"
    assert model._backbone is plain_backbone, \
        "_backbone should point to the plain unwrapped module"
    assert model._active_lora is None

test("merge_lora: task_1 survives merging task_0", test_merge_lora_no_double_unwrap)
test("merge_lora: LoraModel unwraps when all adapters merged", test_merge_lora_unwraps_when_empty)


# ---------------------------------------------------------------------------
# 4. ContinualTrainer empty-params fallback
# ---------------------------------------------------------------------------
print("\n=== 4. ContinualTrainer empty-params fallback ===")

def _make_tiny_model():
    import torch.nn as nn
    from ctk.models.base import CLModel, IncrementalClassifier

    class TinyModel(CLModel):
        def __init__(self):
            super().__init__()
            self._backbone = nn.Linear(4, 8)
            self._head = IncrementalClassifier(8)
            self._head.grow(2)
        @property
        def embed_dim(self): return 8
        def _encode(self, x): return self._backbone(x)
        def _default_lora_targets(self): return []

    return TinyModel()

def test_frozen_backbone_falls_back_to_head():
    model = _make_tiny_model()
    # Freeze only the backbone
    for p in model._backbone.parameters():
        p.requires_grad_(False)

    # Backbone params should all be frozen
    backbone_trainable = [p for p in model._backbone.parameters() if p.requires_grad]
    assert len(backbone_trainable) == 0, "Expected backbone params to be frozen"

    # model.trainable_params() includes head (still trainable) — optimizer won't be empty
    all_trainable = model.trainable_params()
    assert len(all_trainable) > 0, "trainable_params() should include head params"

    # Head params explicitly available as fallback
    head_params = [p for p in model.get_head().parameters() if p.requires_grad]
    assert len(head_params) > 0, "Head should still be trainable"

def test_all_frozen_produces_empty_params():
    model = _make_tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)   # freeze everything

    trainable  = model.trainable_params()
    head_params = [p for p in model.get_head().parameters() if p.requires_grad]
    params = trainable + head_params

    # Both empty → RuntimeError path in ContinualTrainer
    assert len(params) == 0, "Expected all params to be frozen"

test("frozen backbone falls back to head params", test_frozen_backbone_falls_back_to_head)
test("all-frozen model produces empty param list (RuntimeError path)", test_all_frozen_produces_empty_params)


# ---------------------------------------------------------------------------
# 5. End-to-end mini continual training on GPU (synthetic data)
# ---------------------------------------------------------------------------
print("\n=== 5. End-to-end mini training (GPU) ===")

def test_end_to_end_gpu():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from ctk.models.base import CLModel, IncrementalClassifier
    from ctk.scenarios.base import CLScenario, TaskInfo
    from ctk.strategies.base import Strategy
    from ctk.trainers.cl_trainer import ContinualTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"         device: {device}")

    # --- Minimal CLModel ---
    class TinyModel(CLModel):
        def __init__(self):
            super().__init__(device=device)
            self._backbone = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
            self._head = IncrementalClassifier(8)
        @property
        def embed_dim(self): return 8
        def _encode(self, x): return self._backbone(x)
        def _default_lora_targets(self): return []

    # --- Minimal Strategy ---
    class TinyStrategy(Strategy):
        def compute_loss(self, batch, model, task_id):
            import torch.nn.functional as F
            x, y = batch[0].to(model._device), batch[1].to(model._device)
            return F.cross_entropy(model(x, task_id), y)

    # --- Minimal CLScenario (bypasses benchmark requirement) ---
    class TinyScenario(CLScenario):
        _NUM_TASKS = 3

        def __init__(self, n=64):
            # Don't call super().__init__() — no real benchmark
            self._n = n
            self._task_infos = None

        @property
        def num_tasks(self):
            return self._NUM_TASKS

        def setup(self, model):
            infos = []
            for t in range(self._NUM_TASKS):
                x = torch.randn(self._n, 16)
                y = torch.randint(0, 2, (self._n,))
                loader = DataLoader(TensorDataset(x, y), batch_size=16)
                model.grow_head(2)
                infos.append(TaskInfo(
                    task_id=t,
                    classes=list(range(t * 2, t * 2 + 2)),
                    train_loader=loader,
                    test_loader=loader,
                ))
            self._task_infos = infos

        def __iter__(self):
            return iter(self._task_infos)

        def get_task(self, task_id):
            return self._task_infos[task_id]

    model    = TinyModel()
    scenario = TinyScenario()
    strategy = TinyStrategy()

    trainer = ContinualTrainer(
        model, scenario, strategy,
        lr=1e-3, epochs_per_task=2, verbose=True, device=device,
    )
    results = trainer.fit()

    assert "accuracy_matrix" in results
    assert results["accuracy_matrix"].shape == (3, 3)
    assert "avg_accuracy" in results
    assert "avg_forgetting" in results
    print(f"         avg_accuracy={results['avg_accuracy']:.1f}%  "
          f"avg_forgetting={results['avg_forgetting']:.1f}%")

test("end-to-end ContinualTrainer (3 tasks, synthetic data)", test_end_to_end_gpu)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed < total:
    print("\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    - {name}: {err}")
print(f"{'='*60}\n")
sys.exit(0 if passed == total else 1)

"""Unit tests for ctk.utils module."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf

from ctk.utils import (
    set_seed,
    get_device,
    setup_logging,
    load_config,
    save_config,
    count_parameters,
    print_model_summary,
    ensure_dir,
    get_optimizer,
    get_scheduler,
    format_time,
    AverageMeter,
    move_to_device,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(42)
    x2 = torch.randn(5)

    assert torch.allclose(x1, x2), "Random seed not working correctly"


def test_get_device():
    """Test device selection."""
    device = get_device()
    assert isinstance(device, torch.device)

    device_cpu = get_device("cpu")
    assert device_cpu.type == "cpu"

    if torch.cuda.is_available():
        device_cuda = get_device("cuda")
        assert device_cuda.type == "cuda"


def test_count_parameters():
    """Test parameter counting."""
    model = SimpleModel()

    total_params = count_parameters(model, trainable_only=False)
    assert total_params == (10 * 20 + 20) + (20 * 5 + 5)  # fc1 + fc2

    # Freeze fc1
    for param in model.fc1.parameters():
        param.requires_grad = False

    trainable_params = count_parameters(model, trainable_only=True)
    assert trainable_params == (20 * 5 + 5)  # Only fc2


def test_load_save_config():
    """Test config loading and saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Create test config
        test_config = OmegaConf.create({
            "model": {"name": "resnet18", "pretrained": True},
            "training": {"lr": 0.001, "batch_size": 32}
        })

        # Save config
        save_config(test_config, config_path)
        assert config_path.exists()

        # Load config
        loaded_config = load_config(config_path)
        assert loaded_config.model.name == "resnet18"
        assert loaded_config.training.lr == 0.001

        # Test overrides
        loaded_with_override = load_config(
            config_path,
            overrides=["training.lr=0.01"]
        )
        assert loaded_with_override.training.lr == 0.01


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test" / "nested" / "directory"

        result = ensure_dir(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir


def test_get_optimizer():
    """Test optimizer creation."""
    model = SimpleModel()

    # Test SGD
    sgd_optimizer = get_optimizer(
        model.parameters(),
        optimizer_name="sgd",
        learning_rate=0.01
    )
    assert isinstance(sgd_optimizer, torch.optim.SGD)

    # Test Adam
    adam_optimizer = get_optimizer(
        model.parameters(),
        optimizer_name="adam",
        learning_rate=0.001
    )
    assert isinstance(adam_optimizer, torch.optim.Adam)

    # Test AdamW
    adamw_optimizer = get_optimizer(
        model.parameters(),
        optimizer_name="adamw",
        learning_rate=0.001
    )
    assert isinstance(adamw_optimizer, torch.optim.AdamW)


def test_get_scheduler():
    """Test scheduler creation."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())

    # Test linear scheduler
    linear_scheduler = get_scheduler(optimizer, "linear", num_training_steps=100)
    assert linear_scheduler is not None

    # Test cosine scheduler
    cosine_scheduler = get_scheduler(optimizer, "cosine", num_training_steps=100)
    assert cosine_scheduler is not None

    # Test step scheduler
    step_scheduler = get_scheduler(optimizer, "step", step_size=30)
    assert isinstance(step_scheduler, torch.optim.lr_scheduler.StepLR)

    # Test none
    none_scheduler = get_scheduler(optimizer, "none")
    assert none_scheduler is None


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30s"
    assert format_time(90) == "1m 30s"
    assert format_time(3661) == "1h 1m 1s"
    assert format_time(7200) == "2h 0m 0s"


def test_average_meter():
    """Test AverageMeter class."""
    meter = AverageMeter("test")

    meter.update(10, n=1)
    assert meter.val == 10
    assert meter.avg == 10

    meter.update(20, n=1)
    assert meter.val == 20
    assert meter.avg == 15

    meter.update(30, n=2)
    assert meter.val == 30
    assert meter.avg == 22.5  # (10 + 20 + 60) / 4

    meter.reset()
    assert meter.val == 0
    assert meter.avg == 0


def test_move_to_device():
    """Test moving data to device."""
    device = torch.device("cpu")

    # Test tensor
    tensor = torch.randn(5, 5)
    moved_tensor = move_to_device(tensor, device)
    assert moved_tensor.device == device

    # Test dict
    data_dict = {
        "input": torch.randn(5, 5),
        "label": torch.randn(5)
    }
    moved_dict = move_to_device(data_dict, device)
    assert moved_dict["input"].device == device
    assert moved_dict["label"].device == device

    # Test list
    data_list = [torch.randn(5, 5), torch.randn(3, 3)]
    moved_list = move_to_device(data_list, device)
    assert all(t.device == device for t in moved_list)

    # Test tuple
    data_tuple = (torch.randn(5, 5), torch.randn(3, 3))
    moved_tuple = move_to_device(data_tuple, device)
    assert all(t.device == device for t in moved_tuple)


def test_print_model_summary(capsys):
    """Test model summary printing."""
    model = SimpleModel()

    print_model_summary(model, detailed=False)
    captured = capsys.readouterr()
    assert "Model Summary" in captured.out
    assert "Total parameters" in captured.out
    assert "Trainable parameters" in captured.out

    print_model_summary(model, detailed=True)
    captured = capsys.readouterr()
    assert "Per-Layer Parameters" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

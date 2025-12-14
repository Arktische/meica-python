from typing import Any, Dict, List

import torch
import pytest

from trainer.instantiate import (
    _apply_post_configure,
    _PostConfigureContext,
    NodeRef,
    _instantiate,
)
from trainer import Trainer


def test_instantiate_linear_and_optimizer_and_scheduler_with_trainable_parameters() -> None:
    config: Dict[str, Any] = {
        "module": {
            "type": "torch.nn.Linear",
            "args": {"in_features": 2, "out_features": 1},
        },
        "tensor_x": {
            "type": "torch.ones",
            "args": {"size": [4, 2]},
        },
        "tensor_y": {
            "type": "torch.zeros",
            "args": {"size": [4, 1]},
        },
        "train_dataloader": {
            "type": "torch.utils.data.DataLoader",
            "args": {
                "dataset": {
                    "type": "torch.utils.data.TensorDataset",
                    "args": ["${tensor_x}", "${tensor_y}"],
                },
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
                "drop_last": False,
            },
            "purpose": "train"
        },
        "optimizer": {
            "type": "torch.optim.SGD",
            "args": {
                "lr": 0.1,
                "params": {
                    "object": "${module}",
                    "trainable_parameters": None,
                },
            },
        },
        "lr_scheduler": {
            "type": "torch.optim.lr_scheduler.StepLR",
            "args": {
                "optimizer": "${optimizer}",
                "step_size": 1,
            },
        },
    }
    trainer = Trainer()
    trainer.configure(config)
    assert isinstance(trainer.module, torch.nn.Module)
    optimizer = trainer.__optimizers__.get("optimizer")
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = trainer.__lr_schedulers__.get("lr_scheduler")
    assert scheduler is not None
    assert isinstance(trainer.__train_dataloader__, torch.utils.data.DataLoader)


def test_trainer_configure_merges_multiple_dict_configs() -> None:
    base_config: Dict[str, Any] = {
        "module": {
            "type": "torch.nn.Linear",
            "args": {"in_features": 2, "out_features": 1},
        },
        "tensor_x": {
            "type": "torch.ones",
            "args": {"size": [4, 2]},
        },
        "tensor_y": {
            "type": "torch.zeros",
            "args": {"size": [4, 1]},
        },
        "train_dataloader": {
            "type": "torch.utils.data.DataLoader",
            "args": {
                "dataset": {
                    "type": "torch.utils.data.TensorDataset",
                    "args": ["${tensor_x}", "${tensor_y}"],
                },
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
                "drop_last": False,
            },
            "purpose": "train",
        },
    }
    extra_config: Dict[str, Any] = {
        "optimizer": {
            "type": "torch.optim.SGD",
            "args": {
                "lr": 0.1,
                "params": {
                    "object": "${module}",
                    "trainable_parameters": None,
                },
            },
        },
        "lr_scheduler": {
            "type": "torch.optim.lr_scheduler.StepLR",
            "args": {
                "optimizer": "${optimizer}",
                "step_size": 1,
            },
        },
    }
    trainer = Trainer()
    trainer.configure(base_config, extra_config)
    assert isinstance(trainer.module, torch.nn.Module)
    optimizer = trainer.__optimizers__.get("optimizer")
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = trainer.__lr_schedulers__.get("lr_scheduler")
    assert scheduler is not None
    assert isinstance(trainer.__train_dataloader__, torch.utils.data.DataLoader)


def test_trainer_configure_accepts_config_path(tmp_path) -> None:
    config_text = """
module:
  type: torch.nn.Linear
  args:
    in_features: 2
    out_features: 1

tensor_x:
  type: torch.ones
  args:
    size: [4, 2]

tensor_y:
  type: torch.zeros
  args:
    size: [4, 1]

train_dataloader:
  type: torch.utils.data.DataLoader
  args:
    dataset:
      type: torch.utils.data.TensorDataset
      args:
        - ${tensor_x}
        - ${tensor_y}
    batch_size: 2
    shuffle: false
    num_workers: 0
    drop_last: false
  purpose: train

optimizer:
  type: torch.optim.SGD
  args:
    lr: 0.1
    params:
      object: ${module}
      trainable_parameters:

lr_scheduler:
  type: torch.optim.lr_scheduler.StepLR
  args:
    optimizer: ${optimizer}
    step_size: 1
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)
    trainer = Trainer()
    trainer.configure(str(config_path))
    assert isinstance(trainer.module, torch.nn.Module)
    optimizer = trainer.__optimizers__.get("optimizer")
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = trainer.__lr_schedulers__.get("lr_scheduler")
    assert scheduler is not None
    assert isinstance(trainer.__train_dataloader__, torch.utils.data.DataLoader)


def test_post_configure_context_collects_all_leaf_nodes() -> None:
    root: Dict[str, Any] = {
        "a": {"b": 1, "c": [2, 3]},
        "d": 4,
    }
    collected: List[NodeRef[Any]] = []

    def transform(xs: List[NodeRef[Any]]) -> None:
        collected.extend(xs)

    ctx = _PostConfigureContext(filters=[lambda _: True], transform=transform)
    _apply_post_configure(root=root, contexts=[ctx])
    keys_paths = sorted(x.keys_path for x in collected)
    assert keys_paths == ["a.b", "a.c[0]", "a.c[1]", "d"]


def test_post_configure_context_deduplicates_by_value_identity() -> None:
    shared = object()
    root: Dict[str, Any] = {
        "a": shared,
        "b": shared,
    }
    collected: List[NodeRef[Any]] = []

    def transform(xs: List[NodeRef[Any]]) -> None:
        collected.extend(xs)

    ctx = _PostConfigureContext(filters=[lambda _: True], transform=transform)
    _apply_post_configure(root=root, contexts=[ctx])
    assert len(collected) == 1


def test_string_full_reference_preserves_target_type() -> None:
    config: Dict[str, Any] = {
        "value": {
            "type": "builtins.int",
            "args": [1],
        },
        "ref": "${value}",
    }
    _instantiate(root=config, keys=[], node=config)
    assert isinstance(config["value"], int)
    assert config["ref"] == config["value"]


def test_string_interpolation_uses_reference_values_as_str() -> None:
    config: Dict[str, Any] = {
        "value": {
            "type": "builtins.int",
            "args": [2],
        },
        "text": "v=${value}",
    }
    _instantiate(root=config, keys=[], node=config)
    assert isinstance(config["value"], int)
    assert config["text"] == "v=2"


def test_string_literal_escape_uses_dollar_dollar_brace() -> None:
    config: Dict[str, Any] = {
        "value": 1,
        "text": "$${value}",
    }
    _instantiate(root=config, keys=[], node=config)
    assert config["text"] == "${value}"


def test_invalid_reference_syntax_raises_value_error() -> None:
    config: Dict[str, Any] = {
        "text": "x=${value",
    }
    with pytest.raises(ValueError):
        _instantiate(root=config, keys=[], node=config)


def test_missing_reference_path_raises_value_error() -> None:
    config: Dict[str, Any] = {
        "text": "${missing.path}",
    }
    with pytest.raises(ValueError) as exc:
        _instantiate(root=config, keys=[], node=config)
    assert "missing.path" in str(exc.value)

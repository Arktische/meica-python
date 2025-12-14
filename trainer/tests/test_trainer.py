from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset
import accelerate

from trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.loss_fn = torch.nn.MSELoss()

    def train_step(self, batch: Any, step: int) -> torch.Tensor:
        x, y = batch
        out = self.model(x)
        loss = self.loss_fn(out, y)
        return loss


class RecordingTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.loss_fn = torch.nn.MSELoss()
        self.seen: List[int] = []

    def train_step(self, batch: Any, step: int) -> torch.Tensor:
        x, y = batch
        self.seen.extend(int(v) for v in x.view(-1).tolist())
        out = self.model(x)
        loss = self.loss_fn(out, y)
        return loss


def make_dataset(n: int) -> DataLoader:
    x = torch.arange(float(n)).view(-1, 1)
    y = torch.zeros_like(x)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)


def configure_trainer_for_manual_use(trainer: Trainer, dataloader: DataLoader) -> None:
    trainer._accelerator = accelerate.Accelerator()
    trainer.__train_dataloader__ = dataloader
    trainer.__train_batch_size__ = dataloader.batch_size
    trainer.__trainable_modules__ = {}
    trainer.__optimizers__ = {}
    trainer.__lr_schedulers__ = {}
    if hasattr(trainer, "model"):
        model = getattr(trainer, "model")
        if isinstance(model, torch.nn.Module):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            trainer.__trainable_modules__["model"] = model
            trainer.__optimizers__["optimizer"] = optimizer
            trainer.__lr_schedulers__["scheduler"] = scheduler
    trainer.epoch = 1
    trainer.manual_backward = False
    trainer.__configure_done__ = True


def test_training_loop_runs_one_epoch_without_checkpoint(tmp_path) -> None:
    trainer = SimpleTrainer()
    dataloader = make_dataset(6)
    configure_trainer_for_manual_use(trainer, dataloader)
    trainer.checkpoint_dir = None
    trainer.checkpoint_every_n_steps = None
    trainer.checkpoint_every_n_epochs = None
    trainer._training_loop()
    for p in trainer.model.parameters():
        assert p.grad is None


def test_training_loop_raises_on_missing_components() -> None:
    trainer = SimpleTrainer()
    trainer._accelerator = accelerate.Accelerator()
    trainer.__train_dataloader__ = None
    trainer.epoch = 1
    trainer.manual_backward = False
    trainer.__configure_done__ = True
    try:
        trainer._training_loop()
        raise AssertionError("expected ValueError for missing dataloader")
    except ValueError as e:
        assert "train dataloader" in str(e)

    trainer.__train_dataloader__ = make_dataset(0)
    try:
        trainer._training_loop()
        raise AssertionError("expected ValueError for empty dataloader")
    except ValueError as e:
        assert "train dataloader is empty" in str(e)


def test_should_save_checkpoint_by_step_and_epoch() -> None:
    trainer = SimpleTrainer()
    dataloader = make_dataset(4)
    configure_trainer_for_manual_use(trainer, dataloader)
    trainer.checkpoint_dir = "unused"
    trainer.checkpoint_every_n_steps = 2
    trainer.checkpoint_every_n_epochs = 1
    assert trainer._should_save_checkpoint(epoch=0, step=1, global_step=2)
    last_step = len(dataloader) - 1
    assert trainer._should_save_checkpoint(epoch=0, step=last_step, global_step=4)


def test_checkpoint_saves_trainer_and_modules(tmp_path) -> None:
    trainer = SimpleTrainer()
    dataloader = make_dataset(4)
    configure_trainer_for_manual_use(trainer, dataloader)
    trainer.epoch = 1
    trainer.checkpoint_dir = str(tmp_path)
    trainer.checkpoint_every_n_steps = 2
    trainer.checkpoint_every_n_epochs = None
    trainer._training_loop()
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert dirs
    ckpt_dir = dirs[0]
    trainer_state_path = ckpt_dir / "trainer.pt"
    module_path = ckpt_dir / "model.pt"
    assert trainer_state_path.is_file()
    assert module_path.is_file()
    state: Dict[str, Any] = torch.load(trainer_state_path, weights_only=False)
    assert "rng_state" in state
    assert "optimizer_states" in state
    assert "lr_scheduler_states" in state


def test_resume_from_checkpoint_matches_continuous_training(tmp_path) -> None:
    dataloader1 = make_dataset(8)
    baseline = RecordingTrainer()
    configure_trainer_for_manual_use(baseline, dataloader1)
    baseline.epoch = 2
    baseline._training_loop()
    baseline_state = baseline.model.state_dict()

    dataloader2 = make_dataset(8)
    first = RecordingTrainer()
    configure_trainer_for_manual_use(first, dataloader2)
    first.epoch = 1
    first.checkpoint_dir = str(tmp_path)
    first.checkpoint_every_n_epochs = 1
    first.checkpoint_every_n_steps = None
    first._training_loop()
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert dirs
    ckpt_dir = dirs[-1]

    dataloader3 = make_dataset(8)
    resumed = RecordingTrainer()
    configure_trainer_for_manual_use(resumed, dataloader3)
    resumed.epoch = 2
    resumed.resume_from_checkpoint(str(ckpt_dir))
    resumed._training_loop()
    resumed_state = resumed.model.state_dict()
    assert set(baseline_state.keys()) == set(resumed_state.keys())
    for k in baseline_state.keys():
        assert torch.allclose(baseline_state[k], resumed_state[k])

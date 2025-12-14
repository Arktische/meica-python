import itertools
import os
import random
from omegaconf import OmegaConf
import torch
import accelerate
from types import SimpleNamespace
from torch.nn import Module
from tqdm.auto import tqdm
from .instantiate import (
    _apply_post_configure,
    _instantiate,
    NodeRef,
    _PostConfigureContext,
)
from typing import Callable, Any, Iterable, List, Dict, Union, TypeVar, Mapping

_T = TypeVar("_T")


def _foreach(
    items: Iterable[_T],
    func: Callable[[_T], None],
):
    for item in items:
        func(item)


def _trainable_parameters(self: Module):
    return filter(lambda p: p.requires_grad, self.parameters())


# Attach a convenience attribute to torch.nn.Module for filtering trainable params.
Module.trainable_parameters = _trainable_parameters  # type: ignore


class Trainer(accelerate.Accelerator):
    """High-level training loop powered by accelerate and config-driven wiring.

    Users subclass Trainer and implement train_step. All modules, optimizers,
    schedulers and dataloaders are constructed from a nested dict (typically
    loaded from an OmegaConf config) via ergosum.instantiate.
    """

    def __init__(
        self, resume_from_checkpoint: Union[str, None] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__post_configure_contexts__: List[_PostConfigureContext] = []
        self.__configure_done__ = False
        self.__trainable_modules__: Dict[str, torch.nn.Module] = {}
        self.__optimizers__: Dict[str, torch.optim.Optimizer] = {}
        self.__lr_schedulers__: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self.__train_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.__train_batch_size__: Union[int, None] = None
        self.__val_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.__val_batch_size__: Union[int, None] = None
        self.epoch: int = 0
        self.manual_backward: bool = False
        self.max_grad_norm: Union[float, None] = None
        self.checkpoint_dir: Union[str, None] = None
        self.checkpoint_every_n_epochs: Union[int, None] = None
        self.checkpoint_every_n_steps: Union[int, None] = None
        self.validate_every_n_steps: Union[int, None] = None
        self.validate_every_n_epochs: Union[int, None] = None
        self.validate_after_checkpoint: bool = False
        self.checkpoint_top_k_by_val_loss: Union[int, None] = None
        self.__resume_checkpoint_dir__: Union[str, None] = resume_from_checkpoint
        self.__start_epoch__: int = 0
        self.__start_step__: int = 0
        self.__resume_global_step__: int = 0

        self.__loss_sum__: Union[torch.Tensor, None] = None
        self.__sample_count__: Union[torch.Tensor, None] = None
        self.__val_loss_sum__: Union[torch.Tensor, None] = None
        self.__val_sample_count__: Union[torch.Tensor, None] = None
        self.__topk_checkpoints__: List[Dict[str, Any]] = []
        self.register_post_configure(
            filters=[
                lambda x: isinstance(x.value, torch.nn.Module),
                lambda x: isinstance(x.value, torch.optim.Optimizer),
                lambda x: isinstance(x.value, torch.optim.lr_scheduler.LRScheduler),
                lambda x: isinstance(x.value, torch.utils.data.DataLoader)
                and "train".casefold() in x.keys_path.casefold(),
                lambda x: isinstance(x.value, torch.utils.data.DataLoader)
                and "val".casefold() in x.keys_path.casefold(),
            ],
            transform=self._configure_training_components,  # type: ignore
        )

        # Attribute mirroring will be performed after post-configure transforms
        # to avoid interference with dedup strategies.

    def _configure_training_components(
        self,
        modules: List[NodeRef[torch.nn.Module]],
        optimizers: List[NodeRef[torch.optim.Optimizer]],
        lr_schedulers: List[NodeRef[torch.optim.lr_scheduler.LRScheduler]],
        train_dataloaders: List[NodeRef[torch.utils.data.DataLoader]],
        val_dataloaders: List[NodeRef[torch.utils.data.DataLoader]],
    ):
        if not len(train_dataloaders) == 1:
            raise ValueError(
                f"Only 1 dataloader for training is allowed but got {len(train_dataloaders)}."
            )
        if len(val_dataloaders) > 1:
            raise ValueError(
                "Either no dataloader for validation or only one dataloader for validation but got "
                f"{len(val_dataloaders)}."
            )
        if not len(optimizers) > 0:
            raise ValueError("No optimizer is configured.")

        if not len(modules) > 0:
            raise ValueError("No torch.nn.Module is configured.")

        raw_train_loader = train_dataloaders[0].value
        raw_val_loader = val_dataloaders[0].value if len(val_dataloaders) == 1 else None
        _foreach(
            func=lambda x: x.set(self.prepare(x.value)),
            items=itertools.chain(
                modules, optimizers, lr_schedulers, train_dataloaders, val_dataloaders
            ),
        )

        self.__train_dataloader__ = train_dataloaders[0].value
        self.__train_batch_size__ = getattr(raw_train_loader, "batch_size", None)

        if len(val_dataloaders) == 1:
            self.__val_dataloader__ = val_dataloaders[0].value
            self.__val_batch_size__ = getattr(raw_val_loader, "batch_size", None)

        _foreach(
            func=lambda x: (
                self.__trainable_modules__.update({x.keys_path: x.value})
                if any(p.requires_grad for p in x.value.parameters())
                else None
            ),
            items=modules,
        )
        _foreach(
            func=lambda x: self.__optimizers__.update({x.keys_path: x.value}),
            items=optimizers,
        )
        _foreach(
            func=lambda x: self.__lr_schedulers__.update({x.keys_path: x.value}),
            items=lr_schedulers,
        )

    def register_post_configure(
        self,
        filters: List[Callable[[NodeRef[Any]], bool]],
        transform: Callable[[List[NodeRef[Any]]], None],
    ):
        """Register a transformation to run on config values after instantiation."""
        self.__post_configure_contexts__.append(
            _PostConfigureContext(filters=filters, transform=transform)
        )

    def _set_attr_batch(self, x: List[NodeRef[Any]]) -> None:
        for node in x:
            self._set_attr_by_keys(node.keys, node.value)

    def _set_attr_by_keys(self, keys: List[Union[str, int]], value: Any):
        """Create attributes on the Trainer to mirror a nested key path."""
        current = self
        if not keys:
            return
        for i, key in enumerate(keys[:-1]):
            next_key = keys[i + 1] if i + 1 < len(keys) else None
            if isinstance(key, str):
                if not hasattr(current, key) or getattr(current, key) is None:
                    if isinstance(next_key, int):
                        setattr(current, key, [])
                    else:
                        setattr(current, key, SimpleNamespace())
                current = getattr(current, key)
            else:
                if not isinstance(current, list):
                    tmp_list = []
                    current = tmp_list
                while len(current) <= key:
                    current.append(SimpleNamespace())
                current = current[key]

        last = keys[-1]
        if isinstance(last, str):
            setattr(current, last, value)
        else:
            if not isinstance(current, list):
                lst = []
                while len(lst) <= last:
                    lst.append(None)
                lst[last] = value
            else:
                while len(current) <= last:
                    current.append(None)
                current[last] = value

    def _set_attrs_from_config(self, root: Dict[str, Any]) -> None:
        def walk(keys: List[Union[str, int]], node: Any):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk([*keys, k], v)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk([*keys, i], item)
            else:
                self._set_attr_by_keys(keys, node)
        walk([], root)

    def configure(
        self, *configs: Union[Dict[str, Any], Mapping[str, Any], str]
    ) -> None:
        """Instantiate one or more configs and wire results onto this Trainer.

        Each argument may be:
        - A path to a config file loadable by OmegaConf (YAML/JSON).
        - A mapping (e.g. dict) representing a config tree.

        All configs are merged in the given order (later overrides earlier),
        then instantiated as a single config tree.
        """
        if len(configs) == 0:
            raise ValueError("At least one config must be provided.")

        merged_conf = None
        for config in configs:
            if isinstance(config, str):
                piece = OmegaConf.load(config)
            elif isinstance(config, Mapping):
                piece = OmegaConf.create(config)
            else:
                raise TypeError("config must be a mapping or a path string")

            if merged_conf is None:
                merged_conf = piece
            else:
                merged_conf = OmegaConf.merge(merged_conf, piece)

        if merged_conf is None:
            raise ValueError("No valid config provided.")

        final_config = OmegaConf.to_container(merged_conf, resolve=False)
        if not isinstance(final_config, dict):
            raise TypeError("merged config must be a dictionary-like mapping")

        self.__configure_done__ = False
        _instantiate(root=final_config, node=final_config, keys=[])
        _apply_post_configure(
            root=final_config, contexts=self.__post_configure_contexts__
        )
        self._set_attrs_from_config(final_config)
        self.__configure_done__ = True

    def _update_loss_accumulator(self, loss: torch.Tensor) -> None:
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor")
        if self.__train_batch_size__ is None:
            return
        if self.__loss_sum__ is None or self.__sample_count__ is None:
            self.__loss_sum__ = torch.zeros((), device=loss.device, dtype=loss.dtype)
            self.__sample_count__ = torch.zeros(
                (), device=loss.device, dtype=loss.dtype
            )
        batch_size_tensor = torch.tensor(
            self.__train_batch_size__, device=loss.device, dtype=loss.dtype
        )
        self.__loss_sum__ = self.__loss_sum__ + loss.detach() * batch_size_tensor
        self.__sample_count__ = self.__sample_count__ + batch_size_tensor

    def _finalize_distributed_loss(self) -> Union[torch.Tensor, None]:
        if self.__loss_sum__ is None or self.__sample_count__ is None:
            return None
        loss_sum: torch.Tensor = self.gather(self.__loss_sum__)  # type: ignore
        count: torch.Tensor = self.gather(self.__sample_count__)  # type: ignore
        total_count = count.sum()
        if total_count.item() == 0:
            self.__loss_sum__ = None
            self.__sample_count__ = None
            return None
        avg_loss = loss_sum.sum() / total_count
        self.__loss_sum__ = None
        self.__sample_count__ = None
        return avg_loss.detach()

    def _update_val_loss_accumulator(self, loss: torch.Tensor) -> None:
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor")
        if self.__val_batch_size__ is None:
            return
        if self.__val_loss_sum__ is None or self.__val_sample_count__ is None:
            self.__val_loss_sum__ = torch.zeros((), device=loss.device, dtype=loss.dtype)
            self.__val_sample_count__ = torch.zeros(
                (), device=loss.device, dtype=loss.dtype
            )
        batch_size_tensor = torch.tensor(
            self.__val_batch_size__, device=loss.device, dtype=loss.dtype
        )
        self.__val_loss_sum__ = self.__val_loss_sum__ + loss.detach() * batch_size_tensor
        self.__val_sample_count__ = self.__val_sample_count__ + batch_size_tensor

    def _finalize_distributed_val_loss(self) -> Union[torch.Tensor, None]:
        if self.__val_loss_sum__ is None or self.__val_sample_count__ is None:
            return None
        loss_sum: torch.Tensor = self.gather(self.__val_loss_sum__)  # type: ignore
        count: torch.Tensor = self.gather(self.__val_sample_count__)  # type: ignore
        total_count = count.sum()
        if total_count.item() == 0:
            self.__val_loss_sum__ = None
            self.__val_sample_count__ = None
            return None
        avg_loss = loss_sum.sum() / total_count
        self.__val_loss_sum__ = None
        self.__val_sample_count__ = None
        return avg_loss.detach()

    def _get_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as np

            state["numpy"] = np.random.get_state()
        except Exception:
            pass
        state["python"] = random.getstate()
        return state

    def _set_rng_state(self, state: Dict[str, Any]) -> None:
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        numpy_state = state.get("numpy")
        if numpy_state is not None:
            try:
                import numpy as np

                np.random.set_state(numpy_state)
            except Exception:
                pass
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)

    def _should_save_checkpoint(self, epoch: int, step: int, global_step: int) -> bool:
        save_by_step = (
            self.checkpoint_every_n_steps is not None
            and self.checkpoint_every_n_steps > 0
            and global_step % self.checkpoint_every_n_steps == 0
        )
        save_by_epoch = False
        if (
            self.checkpoint_every_n_epochs is not None
            and self.checkpoint_every_n_epochs > 0
            and self.__train_dataloader__ is not None
        ):
            is_last_step = step + 1 == len(self.__train_dataloader__)
            if is_last_step and (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                save_by_epoch = True
        return save_by_step or save_by_epoch

    def _should_validate(self, epoch: int, step: int, global_step: int) -> bool:
        by_step = (
            self.validate_every_n_steps is not None
            and self.validate_every_n_steps > 0
            and global_step % self.validate_every_n_steps == 0
        )
        by_epoch = False
        if (
            self.validate_every_n_epochs is not None
            and self.validate_every_n_epochs > 0
            and self.__train_dataloader__ is not None
        ):
            is_last_step = step + 1 == len(self.__train_dataloader__)
            if is_last_step and (epoch + 1) % self.validate_every_n_epochs == 0:
                by_epoch = True
        return by_step or by_epoch

    def _run_validation(self, epoch: int, step: int, global_step: int) -> Union[torch.Tensor, None]:
        if self.__val_dataloader__ is None:
            return None
        modules = list(self.__trainable_modules__.values())
        prev_modes: List[bool] = [m.training for m in modules]
        for m in modules:
            m.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.__val_dataloader__):
                fn = getattr(self, "val_step", None)
                if callable(fn):
                    v = fn(batch, i)  # type: ignore
                else:
                    v = self.train_step(batch, i)
                if isinstance(v, torch.Tensor):
                    self._update_val_loss_accumulator(v)
        for m, mode in zip(modules, prev_modes):
            if mode:
                m.train()
            else:
                m.eval()
        val_loss = self._finalize_distributed_val_loss()
        if val_loss is not None and self.is_local_main_process:
            self.log({"val_loss": val_loss.item()}, step=global_step)
        return val_loss

    def _maybe_save_checkpoint_by_val_loss(
        self, val_loss: Union[torch.Tensor, None], epoch: int, step: int, global_step: int
    ) -> None:
        if self.checkpoint_top_k_by_val_loss is None or self.checkpoint_top_k_by_val_loss <= 0:
            return
        if val_loss is None:
            raise ValueError("validation loss required but missing")
        if self.checkpoint_dir is None:
            return
        if not self.is_local_main_process:
            return
        if not isinstance(val_loss, torch.Tensor):
            raise TypeError("validation loss must be a torch.Tensor")
        if not torch.isfinite(val_loss).all().item():
            raise ValueError("validation loss is not finite")
        k = int(self.checkpoint_top_k_by_val_loss)
        dirname = f"epoch-{epoch}-step-{global_step}"
        checkpoint_dir = os.path.join(self.checkpoint_dir, dirname)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = self._save_trainer_state(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )
        self.on_save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )
        record = {
            "dir": checkpoint_dir,
            "val_loss": float(val_loss.item()),
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
        }
        self.__topk_checkpoints__.append(record)
        self.__topk_checkpoints__.sort(key=lambda r: r["val_loss"])
        while len(self.__topk_checkpoints__) > k:
            worst = self.__topk_checkpoints__.pop(-1)
            try:
                if os.path.isdir(worst["dir"]):
                    for fn in os.listdir(worst["dir"]):
                        fp = os.path.join(worst["dir"], fn)
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
                    try:
                        os.rmdir(worst["dir"])
                    except Exception:
                        pass
            except Exception:
                pass

    def _save_trainer_state(
        self, checkpoint_dir: str, epoch: int, step: int, global_step: int
    ) -> Dict[str, Any]:
        rng_state = self._get_rng_state()
        optimizer_states: Dict[str, Any] = {
            name: optimizer.state_dict()
            for name, optimizer in self.__optimizers__.items()
        }
        lr_scheduler_states: Dict[str, Any] = {
            name: scheduler.state_dict()
            for name, scheduler in self.__lr_schedulers__.items()
        }
        trainer_state: Dict[str, Any] = {
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
            "max_epoch": int(self.epoch),
            "rng_state": rng_state,
            "optimizer_states": optimizer_states,
            "lr_scheduler_states": lr_scheduler_states,
            "num_trainable_modules": len(self.__trainable_modules__),
        }
        path = os.path.join(checkpoint_dir, "trainer.pt")
        torch.save(trainer_state, path)
        return trainer_state

    def _load_trainer_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        if not os.path.isdir(checkpoint_dir):
            raise ValueError("checkpoint_dir must be an existing directory")
        path = os.path.join(checkpoint_dir, "trainer.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"trainer state not found in {checkpoint_dir}")
        trainer_state = torch.load(path, map_location="cpu", weights_only=False)
        rng_state = trainer_state.get("rng_state")
        if isinstance(rng_state, dict):
            self._set_rng_state(rng_state)
        optimizer_states = trainer_state.get("optimizer_states", {})
        for name, state in optimizer_states.items():
            optimizer = self.__optimizers__.get(name)
            if optimizer is not None:
                optimizer.load_state_dict(state)
        lr_scheduler_states = trainer_state.get("lr_scheduler_states", {})
        for name, state in lr_scheduler_states.items():
            scheduler = self.__lr_schedulers__.get(name)
            if scheduler is not None:
                scheduler.load_state_dict(state)
        self.__start_epoch__ = int(trainer_state.get("epoch", 0))
        saved_step = int(trainer_state.get("step", -1))
        self.__start_step__ = max(saved_step + 1, 0)
        self.__resume_global_step__ = int(trainer_state.get("global_step", 0))
        return trainer_state

    def on_save_checkpoint(
        self, checkpoint_dir: str, epoch: int, step: int, global_step: int
    ) -> None:
        for key, module in self.__trainable_modules__.items():
            module_to_save = self.unwrap_model(module)
            state_dict = module_to_save.state_dict()
            path = os.path.join(checkpoint_dir, f"{key}.pt")
            torch.save(state_dict, path)

    def on_load_checkpoint(
        self, checkpoint_dir: str, trainer_state: Dict[str, Any]
    ) -> None:
        num_modules = trainer_state.get("num_trainable_modules")
        if num_modules is None:
            num_modules = len(self.__trainable_modules__)
        if num_modules != len(self.__trainable_modules__):
            raise ValueError("mismatched number of trainable modules in checkpoint")
        for key, module in self.__trainable_modules__.items():
            module_to_load: torch.nn.Module = self.unwrap_model(module)
            path = os.path.join(checkpoint_dir, f"{key}.pt")
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            module_to_load.load_state_dict(state_dict)

    def _save_checkpoint_internal(
        self, epoch: int, step: int, global_step: int
    ) -> None:
        if self.checkpoint_dir is None:
            return
        if not self.is_local_main_process:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        dirname = f"epoch-{epoch}-step-{global_step}"
        checkpoint_dir = os.path.join(self.checkpoint_dir, dirname)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._save_trainer_state(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )
        self.on_save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )

    def resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Restore trainer and module states from a checkpoint directory."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        trainer_state = self._load_trainer_state(checkpoint_dir)
        self.on_load_checkpoint(checkpoint_dir, trainer_state)

    def train_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        """Compute a loss tensor for a single optimization step.

        Subclasses must override this method. When manual_backward is False,
        the returned tensor is used to drive the optimizer step.
        """
        raise NotImplementedError("train_step must be implemented")

    def _training_loop(self) -> None:
        if self.__train_dataloader__ is None:
            raise ValueError("train dataloader not found which is required")
        if not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if len(self.__train_dataloader__) == 0:
            raise ValueError("train dataloader is empty")
        if not self.manual_backward and len(self.__optimizers__) == 0:
            raise ValueError("no optimizers configured while manual_backward is False")
        requires_val = (
            (self.validate_every_n_steps is not None and self.validate_every_n_steps > 0)
            or (self.validate_every_n_epochs is not None and self.validate_every_n_epochs > 0)
            or self.validate_after_checkpoint
            or (self.checkpoint_top_k_by_val_loss is not None and self.checkpoint_top_k_by_val_loss > 0)
        )
        if requires_val and self.__val_dataloader__ is None:
            raise ValueError("validation is required by config but no val_dataloader configured")
        start_epoch = self.__start_epoch__
        if start_epoch < 0:
            start_epoch = 0
        if start_epoch >= self.epoch:
            return
        global_step = self.__resume_global_step__
        trainable_modules = list(self.__trainable_modules__.values())
        for epoch in range(start_epoch, self.epoch):
            initial_step = self.__start_step__ if epoch == start_epoch else 0
            progress_bar = tqdm(
                range(0, len(self.__train_dataloader__)),
                initial=initial_step,
                desc=f"epoch={epoch} Steps",
                disable=not (self.is_local_main_process),
            )
            for step, batch in enumerate(self.__train_dataloader__):
                if epoch == start_epoch and step < self.__start_step__:
                    continue
                with self.accumulate(*trainable_modules):
                    loss = self.train_step(batch, step)
                    if not isinstance(loss, torch.Tensor) and not self.manual_backward:
                        raise TypeError(
                            f"train_step must return a {torch.Tensor} but got {type(loss)}"
                        )
                    if not self.manual_backward and isinstance(loss, torch.Tensor):
                        self._update_loss_accumulator(loss)
                        self.backward(loss)
                        if self.sync_gradients:
                            if self.max_grad_norm is not None:
                                self.clip_grad_norm_(
                                    itertools.chain(
                                        *[
                                            module.parameters()
                                            for module in trainable_modules
                                        ]
                                    ),
                                    self.max_grad_norm,
                                )
                            for optimizer in self.__optimizers__.values():
                                optimizer.step()
                                optimizer.zero_grad()
                            for lr_scheduler in self.__lr_schedulers__.values():
                                lr_scheduler.step()
                if self.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if not self.manual_backward and loss is not None:
                        avg_loss = self._finalize_distributed_loss()
                        if avg_loss is not None:
                            self.log({"loss": avg_loss.item()}, step=global_step)
                            progress_bar.set_postfix(loss=avg_loss.item())
                    ran_val = False
                    val_loss: Union[torch.Tensor, None] = None
                    if self._should_validate(epoch, step, global_step):
                        val_loss = self._run_validation(epoch, step, global_step)
                        ran_val = True
                        if requires_val:
                            if val_loss is None:
                                raise ValueError("validation loss required but missing")
                            if not isinstance(val_loss, torch.Tensor):
                                raise TypeError("validation loss must be a torch.Tensor")
                            if not torch.isfinite(val_loss).all().item():
                                raise ValueError("validation loss is not finite")
                    if self._should_save_checkpoint(epoch, step, global_step):
                        if self.checkpoint_top_k_by_val_loss is not None and self.checkpoint_top_k_by_val_loss > 0:
                            if not ran_val:
                                val_loss = self._run_validation(epoch, step, global_step)
                                ran_val = True
                            if val_loss is None:
                                raise ValueError("validation loss required but missing")
                            if not isinstance(val_loss, torch.Tensor):
                                raise TypeError("validation loss must be a torch.Tensor")
                            if not torch.isfinite(val_loss).all().item():
                                raise ValueError("validation loss is not finite")
                            self._maybe_save_checkpoint_by_val_loss(val_loss, epoch, step, global_step)
                        else:
                            self._save_checkpoint_internal(epoch, step, global_step)
                            if self.validate_after_checkpoint and not ran_val:
                                val_loss = self._run_validation(epoch, step, global_step)
                                if requires_val:
                                    if val_loss is None:
                                        raise ValueError("validation loss required but missing")
                                    if not isinstance(val_loss, torch.Tensor):
                                        raise TypeError("validation loss must be a torch.Tensor")
                                    if not torch.isfinite(val_loss).all().item():
                                        raise ValueError("validation loss is not finite")

    def fit(self):
        """Run the training loop for the configured number of epochs."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        if self.__resume_checkpoint_dir__ is not None:
            self.resume_from_checkpoint(self.__resume_checkpoint_dir__)
        self._training_loop()

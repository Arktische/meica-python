import itertools
import os
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
    PostConfigureContext,
)
from .progress import Progress
from .loss_accumulator import LossAccumulator
from typing import Callable, Any, Iterable, List, Dict, Union, TypeVar

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
    loaded from an OmegaConf config) via _instantiate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.project_configuration.project_dir is None:
            self.project_configuration.set_directories(os.getcwd())
        self.__post_configure_contexts__: List[PostConfigureContext] = []
        self.__configure_done__ = False
        self.__trainable_modules__: Dict[str, torch.nn.Module] = {}
        self.__optimizers__: Dict[str, torch.optim.Optimizer] = {}
        self.__lr_schedulers__: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self.__train_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.__train_batch_size__: int = 0
        self.__total_samples_count__: int = 0
        self.__val_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.epoch: int = 0
        self.manual_backward: bool = False
        self.max_grad_norm: Union[float, None] = None
        self.checkpoint_dir: Union[str, None] = None
        self.checkpoint_every_n_epochs: Union[int, None] = None
        self.checkpoint_every_n_steps: Union[int, None] = None
        self.validate_every_n_steps: Union[int, None] = None
        self.validate_every_n_epochs: Union[int, None] = None

        self.__progress__ = Progress(self)
        self.__train_loss_accumulator__ = LossAccumulator(self)
        self.__val_loss_accumulator__ = LossAccumulator(self)
        self.__topk_checkpoints__: List[Dict[str, Any]] = []

    @property
    def auto_backward(self) -> bool:
        return not self.manual_backward

    def __configure_training_components__(
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

        _foreach(
            func=lambda x: x.set(self.prepare(x.value)),
            items=itertools.chain(
                modules, optimizers, lr_schedulers, train_dataloaders, val_dataloaders
            ),
        )

        self.__train_dataloader__ = train_dataloaders[0].value

        if len(val_dataloaders) == 1:
            self.__val_dataloader__ = val_dataloaders[0].value

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
            PostConfigureContext(filters=filters, transform=transform)
        )

    def __set_attr_batch__(self, x: List[NodeRef[Any]]) -> None:
        for node in x:
            self.__set_attr_by_keys__(node.keys, node.value)

    def __set_attr_by_keys__(self, keys: List[Union[str, int]], value: Any):
        """Create attributes on the Trainer to mirror a nested key path."""
        current: Any = self
        parent: Any = None
        parent_key: Union[str, int, None] = None
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
                parent = current
                parent_key = key
                current = getattr(current, key)
            else:
                if not isinstance(current, list):
                    new_list: List[Any] = []
                    if parent is not None:
                        if isinstance(parent_key, str):
                            setattr(parent, parent_key, new_list)
                        elif isinstance(parent_key, int) and isinstance(parent, list):
                            while len(parent) <= parent_key:
                                parent.append(SimpleNamespace())
                            parent[parent_key] = new_list
                    current = new_list
                while len(current) <= key:
                    current.append(SimpleNamespace())
                parent = current
                parent_key = key
                current = current[key]

        last = keys[-1]
        if isinstance(last, str):
            setattr(current, last, value)
        else:
            if not isinstance(current, list):
                new_list = []
                if parent is not None:
                    if isinstance(parent_key, str):
                        setattr(parent, parent_key, new_list)
                    elif isinstance(parent_key, int) and isinstance(parent, list):
                        while len(parent) <= parent_key:
                            parent.append(SimpleNamespace())
                        parent[parent_key] = new_list
                current = new_list
            while len(current) <= last:
                current.append(None)
            current[last] = value

    def __set_attrs_from_config__(self, root: Dict) -> None:
        def walk(keys: List[Union[str, int]], node: Any):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk([*keys, k], v)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk([*keys, i], item)
            else:
                self.__set_attr_by_keys__(keys, node)

        walk([], root)

    def configure(self, *configs: Union[Dict, str]) -> None:
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
            elif isinstance(config, Dict):
                piece = OmegaConf.create(config)
            else:
                raise TypeError("config must be a dict or a path string")

            if merged_conf is None:
                merged_conf = piece
            else:
                merged_conf = OmegaConf.merge(merged_conf, piece)

        if merged_conf is None:
            raise ValueError("No valid config provided.")

        final_config = OmegaConf.to_container(merged_conf, resolve=False)
        if not isinstance(final_config, Dict):
            raise TypeError("merged config must be a dictionary-like mapping")

        self.__configure_done__ = False
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
            transform=self.__configure_training_components__,  # type: ignore
        )
        _instantiate(root=final_config, node=final_config, keys=[])

        _apply_post_configure(
            root=final_config, contexts=self.__post_configure_contexts__
        )

        self.__set_attrs_from_config__(final_config)
        self.register_for_checkpointing(self.__progress__)
        self.register_for_checkpointing(*list(self.__trainable_modules__.values()))
        self.register_for_checkpointing(*list(self.__optimizers__.values()))
        self.register_for_checkpointing(*list(self.__lr_schedulers__.values()))
        self.register_for_checkpointing(self.__train_loss_accumulator__)
        self.register_for_checkpointing(self.__val_loss_accumulator__)
        self.__configure_done__ = True

    def __should_save_checkpoint__(self) -> bool:
        save_by_step = (
            self.checkpoint_every_n_steps is not None
            and self.checkpoint_every_n_steps > 0
            and self.__progress__.global_step % self.checkpoint_every_n_steps == 0
            and self.__progress__.global_step > 0
        )
        save_by_epoch = False
        if (
            self.checkpoint_every_n_epochs is not None
            and self.checkpoint_every_n_epochs > 0
            and self.__train_dataloader__ is not None
        ):
            is_last_step = self.__progress__.step + 1 == len(self.__train_dataloader__)
            if (
                is_last_step
                and (self.__progress__.epoch + 1) % self.checkpoint_every_n_epochs == 0
            ):
                save_by_epoch = True
        return save_by_step or save_by_epoch

    def __save_checkpoint__(self) -> None:
        if self.project_configuration.automatic_checkpoint_naming:
            checkpoint_dir = self.project_dir
        else:
            checkpoint_dir = os.path.join(
                (
                    os.path.join(self.project_dir, "checkpoints")
                    if self.checkpoint_dir is None
                    else self.checkpoint_dir
                ),
                f"checkpoint_{self.__progress__.global_step}"
                f"_epoch_{self.__progress__.epoch}"
                f"_step_{self.__progress__.step}",
            )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_state(
            output_dir=checkpoint_dir,
            safe_serialization=True,
        )

    def __resume_from_checkpoint__(self, resume_dir: str) -> None:
        """Restore trainer and module states from a checkpoint directory."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        self.load_state(
            input_dir=resume_dir,
        )

    def train_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        """Compute a loss tensor for a single optimization step.

        Subclasses must override this method. When manual_backward is False,
        the returned tensor is used to drive the optimizer step.
        """
        raise NotImplementedError("train_step must be implemented")

    def val_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        """Compute a validation loss tensor for a single validation step.

        Subclasses must override this method to provide validation logic.
        """
        raise NotImplementedError("val_step must be implemented")

    def __should_validate__(self) -> bool:
        val_by_step = (
            self.validate_every_n_steps is not None
            and self.validate_every_n_steps > 0
            and self.__progress__.global_step % self.validate_every_n_steps == 0
        )
        val_by_epoch = False
        if (
            self.validate_every_n_epochs is not None
            and self.validate_every_n_epochs > 0
            and self.__train_dataloader__ is not None
        ):
            is_last_step = self.__progress__.step + 1 == len(self.__train_dataloader__)
            if (
                is_last_step
                and (self.__progress__.epoch + 1) % self.validate_every_n_epochs == 0
            ):
                val_by_epoch = True
        return val_by_step or val_by_epoch

    def __run_validation__(self) -> torch.Tensor:
        if self.__val_dataloader__ is None:
            raise ValueError("dataloader for validation not found which is required")
        # set all trainable modules to eval mode for validation temporarily
        for module in self.__trainable_modules__.values():
            module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.__val_dataloader__):
                v = self.val_step(batch, i)
                if isinstance(v, torch.Tensor):
                    self.__val_loss_accumulator__.update(v, batch)
        # set all trainable modules back to train mode
        for module in self.__trainable_modules__.values():
            module.train()
        val_loss = self.__val_loss_accumulator__.finalize()
        self.log({"val_loss": val_loss.item()}, step=self.__progress__.global_step)
        return val_loss

    def __training_loop__(self, resume_dir: Union[str, None] = None) -> None:
        if self.__train_dataloader__ is None:
            raise ValueError("train dataloader not found which is required")
        if not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if len(self.__train_dataloader__) == 0:
            raise ValueError("train dataloader is empty")
        if len(self.__optimizers__) == 0:
            raise ValueError("no optimizers configured, no way to train anything.")

        trainable_modules = list(self.__trainable_modules__.values())

        if resume_dir is not None:
            self.__resume_from_checkpoint__(resume_dir)

        skipped_dataloader = (
            self.skip_first_batches(self.__train_dataloader__, self.__progress__.step)
            if self.__progress__.step > 0
            else self.__train_dataloader__
        )

        for epoch in range(self.__progress__.epoch, self.epoch):
            progress_bar = tqdm(
                range(0, len(self.__train_dataloader__)),
                initial=(
                    self.__progress__.step if epoch == self.__progress__.epoch else 0
                ),
                desc=f"epoch={epoch}/{self.epoch-1}",
                disable=not (self.is_local_main_process),
            )
            for step, batch in enumerate(
                skipped_dataloader
                if epoch == self.__progress__.epoch
                else self.__train_dataloader__
            ):
                with self.accumulate(*trainable_modules):
                    loss = self.train_step(batch, step)
                    if not isinstance(loss, torch.Tensor) and self.auto_backward:
                        raise TypeError(
                            f"train_step must return a {torch.Tensor} in auto backward mode but got {type(loss)}"
                        )
                    if self.auto_backward and isinstance(loss, torch.Tensor):
                        self.__train_loss_accumulator__.update(loss, batch)
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
                    self.__progress__.update(epoch, step)
                    if self.auto_backward:
                        avg_loss = self.__train_loss_accumulator__.finalize()
                        self.log(
                            {"loss": avg_loss.item()},
                            step=self.__progress__.global_step,
                        )
                        progress_bar.set_postfix(loss=avg_loss.item())
                    if self.__should_validate__():
                        val_loss = self.__run_validation__()
                    if self.__should_save_checkpoint__():
                        self.__save_checkpoint__()
        self.wait_for_everyone()
        self.end_training()

    def fit(self, resume_dir: Union[str, None] = None):
        """Run the training loop for the configured number of epochs."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        self.__training_loop__(resume_dir=resume_dir)

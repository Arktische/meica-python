import itertools
import os
import logging
import shutil
import signal
from omegaconf import OmegaConf
import torch
import accelerate
from types import SimpleNamespace
from torch.nn import Module
from tqdm.auto import tqdm
from .instantiate import (
    instantiate,
    apply_post_configure_to_dict,
    apply_post_configure_to_class,
    NodeRef,
    PostConfigureContext,
)
from .progress_tracker import ProgressTracker
from .loss_accumulator import LossAccumulator
from typing import Callable, Any, Iterable, List, Dict, Union, TypeVar, Tuple

_T = TypeVar("_T")
_LOGGER = logging.getLogger("meica")
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    _handler.setFormatter(_formatter)
    _LOGGER.addHandler(_handler)
_LOGGER.setLevel(logging.INFO)


def _foreach(
    items: Iterable[_T],
    func: Callable[[_T], None],
):
    for item in items:
        func(item)


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

        self.__console_logger__: logging.Logger = _LOGGER
        self.epoch: int = kwargs.get("epoch", 0)
        self.manual_backward: bool = kwargs.get("manual_backward", False)
        self.manual_checkpointing: bool = kwargs.get("manual_checkpointing", False)
        self.max_grad_norm: Union[float, None] = kwargs.get("max_grad_norm", None)
        self.checkpoint_dir: Union[str, None] = kwargs.get("checkpoint_dir", None)
        self.checkpoint_every_n_epochs: Union[int, None] = kwargs.get(
            "checkpoint_every_n_epochs", None
        )
        self.checkpoint_every_n_steps: Union[int, None] = kwargs.get(
            "checkpoint_every_n_steps", None
        )
        self.validate_every_n_steps: Union[int, None] = kwargs.get(
            "validate_every_n_steps", None
        )
        self.validate_every_n_epochs: Union[int, None] = kwargs.get(
            "validate_every_n_epochs", None
        )
        self.early_stop_patience: Union[int, None] = kwargs.get(
            "early_stop_patience", None
        )
        self.early_stop_min_delta: float = kwargs.get("early_stop_min_delta", 1e-3)
        self.__early_stop_best_val_loss__: Union[float, None] = None
        self.__early_stop_bad_count__: int = 0
        self.__early_stop_last_val_loss__: Union[float, None] = None

        self.__post_configure_contexts__: List[PostConfigureContext] = []
        self.__configure_done__ = False
        self.__trainable_modules__: Dict[str, torch.nn.Module] = {}
        self.__optimizers__: Dict[str, torch.optim.Optimizer] = {}
        self.__lr_schedulers__: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self.__train_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.__val_dataloader__: Union[torch.utils.data.DataLoader, None] = None
        self.__progress__ = ProgressTracker(self)
        self.__train_loss_accumulator__ = LossAccumulator(self)
        self.__val_loss_accumulator__ = LossAccumulator(self)

    @property
    def auto_backward(self) -> bool:
        return not self.manual_backward

    def log_message(self, message: str, level: str = "info") -> None:
        if not self.is_local_main_process:
            return
        if level == "debug":
            self.__console_logger__.debug(message)
        elif level == "warning":
            self.__console_logger__.warning(message)
        elif level == "error":
            self.__console_logger__.error(message)
        else:
            self.__console_logger__.info(message)

    def __configure_tensors__(self, tensors: List[NodeRef[torch.Tensor]]):
        for tensor in tensors:
            tensor.set(tensor.value.to(self.device))

    def __configure_training_components__(
        self,
        modules: List[NodeRef[torch.nn.Module]],
        optimizers: List[NodeRef[torch.optim.Optimizer]],
        lr_schedulers: List[NodeRef[torch.optim.lr_scheduler.LRScheduler]],
    ):
        if not len(optimizers) > 0:
            raise ValueError("No optimizer is configured.")

        if not len(modules) > 0:
            raise ValueError("No torch.nn.Module is configured.")

        _foreach(
            func=lambda x: x.set(self.prepare(x.value)),
            items=itertools.chain(modules, optimizers, lr_schedulers),
        )

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

    def register_for_post_configure(
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

    def dry_configure_core(self, *configs: Union[Dict, str]) -> None:
        """Infer types from configs and wire them onto this Trainer without instantiation."""
        from .type_inference import dry_instantiate

        if self.__configure_done__:
            raise RuntimeError("Trainer is already configured.")

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

        # In dry mode, we skip post_configure and just infer types
        dry_instantiate(root=final_config, node=final_config, keys=[])

        # Wire inferred types onto the trainer
        self.__set_attrs_from_config__(final_config)

    def configure(self, dry_run: bool = False, *configs: Union[Dict, str]) -> None:
        if dry_run:
            self.dry_configure_core(*configs)
        else:
            self.configure_core(*configs)

    def configure_core(self, *configs: Union[Dict, str]) -> None:
        """Instantiate one or more configs and wire results onto this Trainer.

        Each argument may be:
        - A path to a config file loadable by OmegaConf (YAML/JSON).
        - A mapping (e.g. dict) representing a config tree.

        All configs are merged in the given order (later overrides earlier),
        then instantiated as a single config tree.
        """
        if self.__configure_done__:
            raise RuntimeError("Trainer is already configured.")

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
        self.register_for_post_configure(
            filters=[
                lambda x: isinstance(x.value, torch.nn.Module),
                lambda x: isinstance(x.value, torch.optim.Optimizer),
                lambda x: isinstance(x.value, torch.optim.lr_scheduler.LRScheduler),
            ],
            transform=self.__configure_training_components__,  # type: ignore
        )
        # move all tensors to device
        self.register_for_post_configure(
            filters=[lambda x: isinstance(x.value, torch.Tensor)],
            transform=self.__configure_tensors__,  # type: ignore
        )
        # instantiate all nodes in the config tree
        instantiate(root=final_config, node=final_config, keys=[])

        # 1. Collect nodes from instantiated config tree
        apply_post_configure_to_dict(
            root=final_config,
            keys=[],
            node=final_config,
            contexts=self.__post_configure_contexts__,
        )

        # 2. Collect nodes from self attributes (subclass components)
        apply_post_configure_to_class(
            obj=self, contexts=self.__post_configure_contexts__
        )

        # 3. Apply all collected transformations
        for ctx in self.__post_configure_contexts__:
            ctx.apply()

        self.__set_attrs_from_config__(final_config)

        # Configure dataloaders
        train_dl_candidate = getattr(self, "train_dataloader", None)
        if train_dl_candidate is not None:
            if callable(train_dl_candidate) and not isinstance(
                train_dl_candidate, torch.utils.data.DataLoader
            ):
                train_dl = train_dl_candidate()
            else:
                train_dl = train_dl_candidate
            if train_dl is not None:
                self.__train_dataloader__ = self.prepare(train_dl)

        val_dl_candidate = getattr(self, "val_dataloader", None)
        if val_dl_candidate is not None:
            if callable(val_dl_candidate) and not isinstance(
                val_dl_candidate, torch.utils.data.DataLoader
            ):
                val_dl = val_dl_candidate()
            else:
                val_dl = val_dl_candidate
            if val_dl is not None:
                self.__val_dataloader__ = self.prepare(val_dl)

        self.register_for_checkpointing(self.__progress__)
        if not self.manual_checkpointing:
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

    def __save_checkpoint__(
        self, prefix: str = "checkpoint", apply_total_limit: bool = True
    ) -> None:
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = self.project_dir
        else:
            base_dir = (
                os.path.join(self.project_dir, "checkpoints")
                if self.checkpoint_dir is None
                else self.checkpoint_dir
            )
            output_dir = os.path.join(
                base_dir,
                f"{prefix}_{self.__progress__.global_step}"
                f"_epoch_{self.__progress__.epoch}"
                f"_step_{self.__progress__.step}",
            )
            if apply_total_limit and self.project_configuration.total_limit is not None:
                if os.path.isdir(base_dir):
                    entries = [
                        x
                        for x in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, x))
                    ]

                    def _extract_sort_key(name: str) -> Tuple[int, float]:
                        full_path = os.path.join(base_dir, name)
                        ctime = os.path.getctime(full_path)
                        try:
                            if "_epoch_" not in name:
                                return (-1, ctime)
                            left_part = name.split("_epoch_")[0]
                            step = int(left_part.split("_")[-1])
                            return (step, ctime)
                        except ValueError:
                            return (-1, ctime)

                    entries.sort(key=_extract_sort_key)
                    while len(entries) >= int(self.project_configuration.total_limit):
                        oldest = entries.pop(0)
                        shutil.rmtree(
                            os.path.join(base_dir, oldest), ignore_errors=True
                        )
        os.makedirs(output_dir, exist_ok=True)
        self.save_state(
            output_dir=output_dir,
            safe_serialization=True,
        )

    def __resume_from_checkpoint__(self, resume_dir: str) -> None:
        """Restore trainer and module states from a checkpoint directory."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        self.load_state(
            input_dir=resume_dir,
        )

    def train_dataloader(self) -> Union[torch.utils.data.DataLoader, None]:
        """Return the training dataloader.

        Subclasses can override this method to provide the training dataloader.
        """
        return getattr(self, "train_dataloader", None)

    def val_dataloader(self) -> Union[torch.utils.data.DataLoader, None]:
        """Return the validation dataloader.

        Subclasses can override this method to provide the validation dataloader.
        """
        return getattr(self, "val_dataloader", None)

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

    def __should_early_stop__(self) -> bool:
        if self.early_stop_patience is None or self.early_stop_patience <= 0:
            return False
        if self.__early_stop_last_val_loss__ is None:
            return False
        if self.__early_stop_best_val_loss__ is None:
            self.__early_stop_best_val_loss__ = self.__early_stop_last_val_loss__
            self.__early_stop_bad_count__ = 0
            return False
        if (
            self.__early_stop_best_val_loss__ - self.__early_stop_last_val_loss__
            > self.early_stop_min_delta
        ):
            self.__early_stop_best_val_loss__ = self.__early_stop_last_val_loss__
            self.__early_stop_bad_count__ = 0
            return False
        self.__early_stop_bad_count__ += 1
        if self.__early_stop_bad_count__ >= self.early_stop_patience:
            self.log_message(
                "Early stopping triggered at global_step="
                + str(self.__progress__.global_step)
                + ", epoch="
                + str(self.__progress__.epoch)
                + ", step="
                + str(self.__progress__.step)
                + ", best_val_loss="
                + f"{self.__early_stop_best_val_loss__:.6f}"
                + ", last_val_loss="
                + f"{self.__early_stop_last_val_loss__:.6f}"
            )
            return True
        return False

    def __run_validation__(self):
        if self.__val_dataloader__ is None:
            raise ValueError("dataloader for validation not found which is required")

        # Store original training states
        original_states = {
            name: module.training for name, module in self.__trainable_modules__.items()
        }

        # set all trainable modules to eval mode for validation temporarily
        for module in self.__trainable_modules__.values():
            module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.__val_dataloader__):
                v = self.val_step(batch, i)
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"val_step must return a torch.Tensor, but got {type(v)}"
                    )
                self.__val_loss_accumulator__.update(v, batch)
        # set all trainable modules back to original mode
        for name, module in self.__trainable_modules__.items():
            module.train(original_states[name])

        val_loss = self.__val_loss_accumulator__.finalize()
        val_loss_value = float(val_loss.item())
        self.__early_stop_last_val_loss__ = val_loss_value
        self.log({"val_loss": val_loss_value}, step=self.__progress__.global_step)
        # return val_loss

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

        def loop_internal():
            if self.__train_dataloader__ is None:
                raise ValueError("train dataloader not found which is required")
            for epoch in range(self.__progress__.epoch, self.epoch):
                progress_bar = tqdm(
                    range(0, len(self.__train_dataloader__)),
                    initial=(
                        self.__progress__.step
                        if epoch == self.__progress__.epoch
                        else 0
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
                            self.__run_validation__()
                            if self.__should_early_stop__():
                                return
                        if self.__should_save_checkpoint__():
                            self.__save_checkpoint__()

        try:
            loop_internal()
        except KeyboardInterrupt:
            self.log_message(
                "KeyboardInterrupt received, synchronizing and saving checkpoint before exit."
            )
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.wait_for_everyone()
            self.__save_checkpoint__(prefix="interrupt", apply_total_limit=False)
        except Exception as e:
            self.log_message("Exception during training: " + repr(e), level="error")
            old_sigint = signal.getsignal(signal.SIGINT)
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                self.wait_for_everyone()
                self.__save_checkpoint__(prefix="error", apply_total_limit=False)
            finally:
                signal.signal(signal.SIGINT, old_sigint)
            raise
        finally:
            self.wait_for_everyone()
            self.end_training()

    def fit(self, resume_dir: Union[str, None] = None):
        """Run the training loop for the configured number of epochs."""
        if not self.__configure_done__:
            raise ValueError("trainer is not configured. call configure() first")
        self.__training_loop__(resume_dir=resume_dir)

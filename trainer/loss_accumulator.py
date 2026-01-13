import torch
import accelerate
import logging
from typing import Dict, Any
from .batch_util import get_batch_size


class LossAccumulator:
    def __init__(self, accelerator: accelerate.Accelerator):
        self.accelerator = accelerator
        self.loss_sum: torch.Tensor = torch.zeros(
            (),
            device=self.accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.sample_count: torch.Tensor = torch.zeros(
            (),
            device=self.accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self.last_batch_size: int = 0
        self.last_batch_size_tensor: torch.Tensor = torch.tensor(
            self.last_batch_size,
            device=self.accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def update(self, loss: torch.Tensor, batch: Any):
        batch_size = get_batch_size(batch)
        if self.last_batch_size_tensor != batch_size:
            self.last_batch_size_tensor.fill_(batch_size)
        self.last_batch_size = batch_size

        self.loss_sum.add_(loss.detach() * self.last_batch_size_tensor)
        self.sample_count.add_(self.last_batch_size_tensor)

    def reset(self):
        self.loss_sum.fill_(0.0)
        self.sample_count.fill_(0.0)
        self.last_batch_size = 0
        self.last_batch_size_tensor.fill_(self.last_batch_size)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "loss_sum": self.loss_sum.detach().clone(),
            "sample_count": self.sample_count.detach().clone(),
            "last_batch_size": self.last_batch_size,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        loss_sum = state.get("loss_sum", None)
        sample_count = state.get("sample_count", None)
        last_batch_size = state.get("last_batch_size", None)
        if not isinstance(loss_sum, torch.Tensor):
            raise TypeError(f"loss_sum must be a torch.Tensor but got {type(loss_sum)}")
        if not isinstance(sample_count, torch.Tensor):
            raise TypeError(
                f"sample_count must be a torch.Tensor but got {type(sample_count)}"
            )
        if not isinstance(last_batch_size, int):
            raise TypeError(
                f"last_batch_size must be an int but got {type(last_batch_size)}"
            )
        self.loss_sum.fill_(float(loss_sum.detach().float().item()))
        self.sample_count.fill_(float(sample_count.detach().float().item()))
        self.last_batch_size = int(last_batch_size)
        self.last_batch_size_tensor.fill_(int(self.last_batch_size))

    def finalize(self) -> torch.Tensor:
        loss_sum = self.accelerator.gather(self.loss_sum)
        sample_count = self.accelerator.gather(self.sample_count)
        if isinstance(loss_sum, torch.Tensor) and isinstance(
            sample_count, torch.Tensor
        ):
            total_count = sample_count.sum().item()
            if total_count == 0:
                logging.getLogger("meica").warning(
                    "LossAccumulator: total_count is 0, returning 0.0"
                )
                return torch.tensor(0.0, device=self.accelerator.device)
            res = (loss_sum.sum() / sample_count.sum()).detach()
            self.reset()
            return res
        else:
            raise ValueError(
                f"expect loss_sum and sample_count to be torch.Tensor but got {type(loss_sum)} and {type(sample_count)}"
            )

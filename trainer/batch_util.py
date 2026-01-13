import torch
from typing import Dict, List, Tuple, Any


def get_batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        if batch.ndim >= 1:
            return batch.shape[0]
        raise ValueError(
            f"batch tensor must have at least 1 dimension, but got {batch.ndim}"
        )
    elif isinstance(batch, Dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        raise ValueError(
            f"batch dict must contain at least one tensor with at least 1 dimension"
        )
    elif isinstance(batch, (List, Tuple)):
        for v in batch:
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        raise ValueError(
            f"batch list/tuple must contain at least one tensor with at least 1 dimension"
        )
    else:
        raise ValueError(f"unrecognized batch type: {type(batch)}")

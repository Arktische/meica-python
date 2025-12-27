from trainer import Trainer
# from diffusers import QwenImageEditPipeline
import torch
from typing import Union

class QwenImageEditFinetuner(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        
        pass
        # return super().train_step(batch, step)
import torch
from diffusers import QwenImageEditPipeline
from typing import Tuple, Optional, Any


def encode_prompt(
    pipeline: Any, prompt: str, prompt_image: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a prompt into a tensor of token ids.

    Args:
        text_encoder (Any): The text encoder to use.
        prompt (str): The prompt to encode.
        prompt_image (Optional[torch.Tensor]): The image to use for prompt encoding. Defaults to None.

    Returns:
        torch.Tensor: The encoded prompt.
        torch.Tensor: The attention mask.
    """
    # if isinstance(text_encoder, QwenImageEditPipeline):
    if type(pipeline) == QwenImageEditPipeline:
        return pipeline.encode_prompt(
            image=prompt_image,
            prompt=[prompt],
            device=pipeline.device,
            num_images_per_prompt=1,
            max_sequence_length=1024,
        )
    else:
        raise NotImplementedError(
            f"Prompt encoding not implemented for {type(pipeline)}"
        )

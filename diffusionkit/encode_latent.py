from typing import Any, Union
from diffusers import AutoencoderKLQwenImage, AutoencoderKL
import torch


def encode_latent(
    vae: Union[AutoencoderKLQwenImage, AutoencoderKL],
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Encode an image into a latent tensor.

    Args:
        vae (Union[AutoencoderKLQwenImage, AutoencoderKL]): The VAE to use.
        image (torch.Tensor): The image to encode.

    Returns:
        torch.Tensor: The encoded latent tensor.
    """
    return vae.encode(image).latent_dist.sample()[0]

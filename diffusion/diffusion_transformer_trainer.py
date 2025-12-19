from trainer import Trainer
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class DiffusionTransformerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_noise_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        raise NotImplementedError("get_noise_scheduler is not implemented")

    def get_vae(self) -> diffusers.AutoencoderKL:
        raise NotImplementedError("get_vae is not implemented")

    def get_pipeline(self) -> diffusers.DiffusionPipeline:
        raise NotImplementedError("get_pipeline is not implemented")

    def get_latents_std(self) -> torch.Tensor:
        raise NotImplementedError("get_latents_std is not implemented")

    def get_latents_mean(self) -> torch.Tensor:
        raise NotImplementedError("get_latents_mean is not implemented")

    def normalize_latents(self, pixel_latents: torch.Tensor) -> torch.Tensor:
        pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
        pixel_latents = (
            pixel_latents - self.get_latents_mean()
        ) * self.get_latents_std()
        return pixel_latents

    def get_noise(self, pixel_latents: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(pixel_latents)

    def get_sigmas(self, pixel_latents: torch.Tensor) -> torch.Tensor:
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=pixel_latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.get_noise_scheduler().config.num_train_timesteps).long()
        timesteps = self.get_noise_scheduler().timesteps[indices].to(device=self.device)
        sigmas = self.get_noise_scheduler().sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.get_noise_scheduler().timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < pixel_latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_noise_scheduler_timesteps(self) -> torch.Tensor:
        return self.get_noise_scheduler().timesteps.to(self.device)

    def get_noise_scheduler_sigmas(self) -> torch.Tensor:
        return self.get_noise_scheduler().sigmas.to(self.device)

    def add_noise_to_latents(self, pixel_latents: torch.Tensor) -> torch.Tensor:
        noise = self.get_noise(pixel_latents)
        sigmas = self.get_sigmas(pixel_latents)
        return (1 - sigmas) * pixel_latents + noise * sigmas

    def pack_noisy_latents_for_input(self, pixel_latents: torch.Tensor) -> torch.Tensor:
        return self.add_noise_to_latents(pixel_latents)
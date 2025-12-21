from trainer import Trainer
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)
from typing import List


class DiffusionTransformerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__noise__: torch.Tensor
        self.__sigmas__: torch.Tensor
        self.__latents_std__: torch.Tensor
        self.__latents_mean__: torch.Tensor
        self.__pixel_latents__: torch.Tensor
        self.__packed_noisy_latents__: torch.Tensor
        self.__pred_latents__: torch.Tensor
        self.__unpacked_pred_latents__: torch.Tensor
        self.__loss_weighting__: torch.Tensor

    def get_noise_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        raise NotImplementedError("get_noise_scheduler is not implemented")

    def get_vae(self):
        raise NotImplementedError("get_vae is not implemented")

    def get_pipeline(self):
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
        timesteps = self.get_noise_scheduler_timesteps()[indices]
        sigmas = self.get_noise_scheduler_sigmas()
        schedule_timesteps = self.get_noise_scheduler_timesteps()
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < pixel_latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_noise_scheduler_timesteps(self) -> torch.Tensor:
        return self.get_noise_scheduler().timesteps.to(self.device)

    def get_noise_scheduler_sigmas(self) -> torch.Tensor:
        return self.get_noise_scheduler().sigmas.to(self.device)

    def add_noise_to_latents(self, *pixel_latents: torch.Tensor) -> List[torch.Tensor]:
        noise = self.get_noise(pixel_latents)
        sigmas = self.get_sigmas(pixel_latents)
        return [(1 - sigmas) * latents + noise * sigmas for latents in pixel_latents]

    def pack_noisy_latents(self, *pixel_latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("pack_noisy_latents_for_input is not implemented")

    def unpack_pred_latents(self, pred_latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("unpack_noisy_latents_from_output is not implemented")

    def predict(
        self,
        pixel_latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        prompt_embeddings_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("predict is not implemented")

    def loss(self) -> torch.Tensor:
        target = self.__noise__ - self.__pixel_latents__
        target = target.permute(0, 2, 1, 3, 4)
        loss = torch.mean(
            (
                self.__loss_weighting__.float()
                * (self.__unpacked_pred_latents__.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss

    def train_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        return super().train_step(batch, step)

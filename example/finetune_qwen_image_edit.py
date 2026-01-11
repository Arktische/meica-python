from trainer import Trainer
import torch
from typing import Union
import diffusers
import torch.nn.functional as F


class QwenImageEditFinetuner(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, batch, step: int) -> Union[torch.Tensor, None]:
        latent: torch.Tensor = batch["latent"]
        control_latent: torch.Tensor = batch["control_latent"]
        embedding: torch.Tensor = batch["embedding"]
        mask: torch.Tensor = batch["mask"]

        b, _, c, h, w = latent.shape
        u: torch.Tensor = (
            diffusers.training_utils.compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=b,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
        )
        indices: torch.Tensor = (
            u * self.noise_scheduler.config.num_train_timesteps
        ).long()
        timesteps: torch.Tensor = self.noise_scheduler.timesteps[indices].to(
            self.device
        )
        sigmas: torch.Tensor = self.noise_scheduler.sigmas[indices].to(self.device)

        noise: torch.Tensor = torch.randn_like(latent)
        noisy_latent: torch.Tensor = torch.lerp(latent, noise, sigmas)

        packed_noisy_latent = diffusers.QwenImageEditPipeline._pack_latents(
            noisy_latent, b, c, h, w
        )
        packed_control_latent = diffusers.QwenImageEditPipeline._pack_latents(
            control_latent, b, c, h, w
        )

        concated_latent = torch.cat([packed_noisy_latent, packed_control_latent], dim=1)
        model_pred = self.transformer(
            hidden_states=concated_latent,
            encoder_hidden_states=embedding,
            encoder_hidden_states_mask=mask,
            guidance=None,
            return_dict=False,
            text_seq_lens=mask.sum(dim=1).tolist(),
            timestep=timesteps / 1000,
            img_shapes=[
                [
                    (1, h // 2, w // 2),
                    (1, h // 2, w // 2),
                ]
            ]
            * b,
        )[0]
        unpacked_pred = diffusers.QwenImageEditPipeline._unpack_latents(
            model_pred[:, : control_latent.shape[1]],
            h * self.vae_scale_factor,
            w * self.vae_scale_factor,
            b,
            self.vae_scale_factor,
        )
        weighting = diffusers.training_utils.compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = (noise - latent).permute(0, 2, 1, 3, 4)
        loss = F.mse_loss(unpacked_pred, target, reduction="none") * weighting
        return loss.mean()

"""Learned inversion baseline: gradient encoder + image decoder, no temporal model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder


class SingleFrameInversion(nn.Module):
    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        encoder_type: str = "residual",
        decoder_type: str = "residual",
        dropout: float = 0.1,
        image_size: int = 84,
        encoder_hidden_dim: Optional[int] = None,
        **decoder_kwargs,
    ):
        super().__init__()

        encoder_kwargs = {}
        if encoder_type == "residual" and encoder_hidden_dim is not None:
            encoder_kwargs["hidden_dim"] = encoder_hidden_dim

        self.gradient_encoder = get_encoder(
            encoder_type=encoder_type,
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            **encoder_kwargs,
        )

        self.image_decoder = get_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
            **decoder_kwargs,
        )

    def forward(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        latents = self.gradient_encoder(gradients)
        images, actions = self.image_decoder(latents)
        return images, actions, latents, None

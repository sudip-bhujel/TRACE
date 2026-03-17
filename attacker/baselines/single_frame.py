"""
Learned inversion model without the temporal transformer.
"""

from typing import Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder


class SingleFrameInversion(nn.Module):
    """
    Learned inversion model without the temporal transformer.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        encoder_type: str = "residual",
        decoder_type: str = "residual",
        dropout: float = 0.1,
        image_size: int = 84,
        **decoder_kwargs,
    ):
        super().__init__()

        self.gradient_encoder = get_encoder(
            encoder_type=encoder_type,
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            dropout=dropout,
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
        """
        Args:
            gradients: (B, T, gradient_dim) or (B, gradient_dim)

        Returns:
            images:       (B, T, 3, H, W)
            actions:      (B, T, num_actions)
            latents:      (B, T, latent_dim)
            token_logits: None  (for API compatibility)
        """
        latents = self.gradient_encoder(gradients)
        images, actions = self.image_decoder(latents)
        return images, actions, latents, None

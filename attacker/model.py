from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from attacker.decoder import get_decoder
from attacker.encoder import get_encoder
from attacker.transformer import CausalTemporalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalGradientInversion(nn.Module):
    """
    Full temporal gradient inversion model.

    Takes a sequence of gradients and produces a sequence of images,
    using causal attention for temporal modeling.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        encoder_hidden_dims: Optional[List[int]] = None,
        encoder_type: str = "basic",
        decoder_type: str = "basic",
        dropout: float = 0.1,
        image_size: int = 84,
    ):
        super().__init__()

        # Use factory function to get encoder
        encoder_kwargs = {}
        if encoder_type == "basic" and encoder_hidden_dims is not None:
            encoder_kwargs["hidden_dims"] = encoder_hidden_dims

        self.gradient_encoder = get_encoder(
            encoder_type=encoder_type,
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            **encoder_kwargs,
        )

        self.temporal_transformer = CausalTemporalTransformer(
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Use factory function to get decoder
        self.image_decoder = get_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
        )

    def forward(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            gradients: (B, T, gradient_dim)
        Returns:
            images: (B, T, 3, H, W)
            actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
        """
        # Encode each gradient
        latents = self.gradient_encoder(gradients)  # (B, T, latent_dim)

        # Apply temporal transformer (causal attention)
        latents = self.temporal_transformer(latents)  # (B, T, latent_dim)

        # Decode to images and actions
        images, actions = self.image_decoder(
            latents
        )  # (B, T, 3, H, W), (B, T, num_actions)

        return images, actions, latents

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder
from attacker.models.transformer import TemporalTransformer

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
        skip_transformer: bool = False,
        is_causal: bool = True,
        **decoder_kwargs,
    ):
        super().__init__()

        self.skip_transformer = skip_transformer

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

        if not skip_transformer:
            self.temporal_transformer = TemporalTransformer(
                latent_dim=latent_dim,
                num_layers=num_transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                is_causal=is_causal,
            )

        # Use factory function to get decoder
        self.image_decoder = get_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
            **decoder_kwargs,
        )

    def forward(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            gradients: (B, T, gradient_dim)
        Returns:
            images: (B, T, 3, H, W)
            actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
            token_logits: (B*T, num_tokens, codebook_size) or None
        """
        # Encode each gradient
        latents = self.gradient_encoder(gradients)  # (B, T, latent_dim)

        # Apply temporal transformer (causal attention) unless skipped
        if not self.skip_transformer:
            latents = self.temporal_transformer(latents)  # (B, T, latent_dim)

        # Decode to images and actions
        # VQ-GAN decoder returns (images, actions, token_logits)
        # Standard decoders return (images, actions)
        output = self.image_decoder(latents)

        if len(output) == 3:
            images, actions, token_logits = output
        else:
            images, actions = output
            token_logits = None

        return images, actions, latents, token_logits

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder
from attacker.models.temporal_baselines import get_temporal_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalGradientInversion(nn.Module):
    """Encodes a gradient sequence and decodes it to a sequence of images and actions."""

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        encoder_hidden_dims: Optional[List[int]] = None,
        encoder_hidden_dim: Optional[int] = None,
        encoder_num_blocks: Optional[int] = None,
        encoder_expansion: Optional[int] = None,
        encoder_projection_rank: Optional[int] = None,
        encoder_type: str = "basic",
        decoder_type: str = "basic",
        dropout: float = 0.1,
        image_size: int = 84,
        skip_transformer: bool = False,
        is_causal: bool = True,
        temporal_model_type: str = "transformer",
        ff_multiplier: int = 4,
        use_rope: bool = False,
        **decoder_kwargs,
    ):
        super().__init__()

        self.skip_transformer = skip_transformer

        encoder_kwargs = {}
        if encoder_type == "basic" and encoder_hidden_dims is not None:
            encoder_kwargs["hidden_dims"] = encoder_hidden_dims
        if encoder_type == "residual":
            if encoder_hidden_dim is not None:
                encoder_kwargs["hidden_dim"] = encoder_hidden_dim
            if encoder_num_blocks is not None:
                encoder_kwargs["num_blocks"] = encoder_num_blocks
            if encoder_expansion is not None:
                encoder_kwargs["expansion"] = encoder_expansion
            encoder_kwargs["projection_rank"] = encoder_projection_rank

        self.gradient_encoder = get_encoder(
            encoder_type=encoder_type,
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            **encoder_kwargs,
        )

        if not skip_transformer:
            self.temporal_model = get_temporal_model(
                temporal_model_type=temporal_model_type,
                latent_dim=latent_dim,
                num_layers=num_transformer_layers,
                num_heads=num_heads,
                ff_multiplier=ff_multiplier,
                dropout=dropout,
                is_causal=is_causal,
                use_rope=use_rope,
            )

        self.image_decoder = get_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
            **decoder_kwargs,
        )

    def forward(
        self, gradients: torch.Tensor, use_flash_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        latents = self.gradient_encoder(gradients)

        if not self.skip_transformer:
            latents = self.temporal_model(
                latents, use_flash_attention=use_flash_attention
            )

        output = self.image_decoder(latents)

        if len(output) == 3:
            images, actions, token_logits = output
        else:
            images, actions = output
            token_logits = None

        return images, actions, latents, token_logits

"""
Gradient Encoder Architectures for Gradient Inversion

This module provides multiple encoder architectures for encoding gradient vectors
into latent representations for image reconstruction.

Available encoders:
- GradientEncoder: Basic MLP encoder with LayerNorm and GELU
- ResidualGradientEncoder: Encoder with residual connections for better gradient flow
"""

from typing import List

import torch
import torch.nn as nn


class GradientEncoder(nn.Module):
    """Basic MLP encoder with LayerNorm and GELU activation."""

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        hidden_dims: List[int] = [4096, 2048, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = gradient_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, gradient_dim) or (B, gradient_dim)
        Returns:
            (B, T, latent_dim) or (B, latent_dim)
        """
        return self.encoder(x)


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization."""

    def __init__(self, dim: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        hidden_dim = dim * expansion
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualGradientEncoder(nn.Module):
    """
    Gradient encoder with residual connections for better gradient flow.

    Uses a projection layer followed by stacked residual blocks.
    This helps with training deeper networks and preserves information flow.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        hidden_dim: int = 2048,
        num_blocks: int = 4,
        dropout: float = 0.1,
        expansion: int = 4,
    ):
        super().__init__()

        # Project to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(gradient_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stacked residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout, expansion) for _ in range(num_blocks)]
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def get_encoder(
    encoder_type: str,
    gradient_dim: int,
    latent_dim: int = 512,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get encoder by type.

    Args:
        encoder_type: One of 'basic', 'residual'
        gradient_dim: Input gradient dimension
        latent_dim: Output latent dimension
        dropout: Dropout rate
        **kwargs: Additional encoder-specific arguments

    Returns:
        Encoder module
    """
    encoders = {
        "basic": GradientEncoder,
        "residual": ResidualGradientEncoder,
    }

    if encoder_type not in encoders:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Available: {list(encoders.keys())}"
        )

    return encoders[encoder_type](
        gradient_dim=gradient_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        **kwargs,
    )

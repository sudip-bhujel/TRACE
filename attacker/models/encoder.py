"""Gradient encoders that map flat gradient vectors into latent representations."""

from typing import List, Optional

import torch
import torch.nn as nn


class GradientEncoder(nn.Module):
    """Basic MLP encoder with LayerNorm and GELU."""

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
        return self.encoder(x)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

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
    Gradient encoder with stacked residual blocks. The input projection may be
    factorised as ``gradient_dim -> projection_rank -> hidden_dim`` to reduce
    parameters versus a single dense layer.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        dropout: float = 0.1,
        expansion: int = 4,
        projection_rank: Optional[int] = None,
    ):
        super().__init__()

        if projection_rank is None:
            self.input_proj = nn.Sequential(
                nn.Linear(gradient_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            if projection_rank <= 0:
                raise ValueError(
                    f"projection_rank must be > 0 or None, got {projection_rank}"
                )
            self.input_proj = nn.Sequential(
                nn.Linear(gradient_dim, projection_rank),
                nn.LayerNorm(projection_rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_rank, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout, expansion) for _ in range(num_blocks)]
        )

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
    """Construct an encoder by name. Supported: ``basic``, ``residual``."""
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

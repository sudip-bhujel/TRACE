"""
Image Decoder Architectures for Gradient Inversion

This module provides multiple decoder architectures for decoding latent
representations into images and action predictions.

Available decoders:
- ImageDecoder: Basic transposed convolution decoder
- ResidualImageDecoder: Decoder with residual blocks for better detail
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """Basic transposed convolution decoder."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4  # 21

        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # Transposed convolutions for image generation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 21 -> 42
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 42 -> 84
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),  # 84 -> 84
            nn.Sigmoid(),
        )

        # Action prediction head
        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3
        time_shape = None

        if has_time_dim:
            B, T, D = x.shape
            time_shape = (B, T)
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)
        images = self.decoder(h)

        actions = self.action_head(x_flat)

        if has_time_dim and time_shape is not None:
            batch_size, sequence_length = time_shape
            images = images.view(
                batch_size, sequence_length, 3, self.image_size, self.image_size
            )
            actions = actions.view(batch_size, sequence_length, -1)

        return images, actions


class ResidualBlock2d(nn.Module):
    """2D Residual block for image decoder."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = min(32, channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class ResidualImageDecoder(nn.Module):
    """
    Decoder with residual blocks for better reconstruction quality.

    Adds residual blocks after each upsampling layer to preserve details
    and improve gradient flow during training.

    Uses GroupNorm instead of BatchNorm to avoid inplace running-stat
    updates that conflict with autoregressive rollout training.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4

        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # First upsample block with residuals
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(32, 128)
        self.res1 = nn.Sequential(
            *[ResidualBlock2d(128) for _ in range(num_res_blocks)]
        )

        # Second upsample block with residuals
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(32, 64)
        self.res2 = nn.Sequential(*[ResidualBlock2d(64) for _ in range(num_res_blocks)])

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3
        time_shape = None

        if has_time_dim:
            B, T, D = x.shape
            time_shape = (B, T)
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        h = F.relu(self.gn1(self.up1(h)))
        h = self.res1(h)

        h = F.relu(self.gn2(self.up2(h)))
        h = self.res2(h)

        images = self.final(h)
        actions = self.action_head(x_flat)

        if has_time_dim and time_shape is not None:
            batch_size, sequence_length = time_shape
            images = images.view(
                batch_size, sequence_length, 3, self.image_size, self.image_size
            )
            actions = actions.view(batch_size, sequence_length, -1)

        return images, actions


def get_decoder(
    decoder_type: str,
    latent_dim: int = 512,
    image_size: int = 84,
    num_actions: int = 5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get decoder by type.

    Args:
        decoder_type: One of 'basic', 'residual'
        latent_dim: Input latent dimension
        image_size: Output image size
        num_actions: Number of actions for action head
        **kwargs: Additional decoder-specific arguments

    Returns:
        Decoder module
    """
    decoders = {
        "basic": ImageDecoder,
        "residual": ResidualImageDecoder,
    }

    if decoder_type not in decoders:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. Available: {list(decoders.keys())}"
        )

    return decoders[decoder_type](
        latent_dim=latent_dim,
        image_size=image_size,
        num_actions=num_actions,
        **kwargs,
    )

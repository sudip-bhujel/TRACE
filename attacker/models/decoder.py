"""Image decoders that map latent representations to images and action logits."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """Basic transposed-convolution image decoder with an action head."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4

        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
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
    Image decoder with residual blocks after each upsample. Uses GroupNorm so
    running statistics do not conflict with autoregressive rollout training.
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

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(32, 128)
        self.res1 = nn.Sequential(
            *[ResidualBlock2d(128) for _ in range(num_res_blocks)]
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(32, 64)
        self.res2 = nn.Sequential(*[ResidualBlock2d(64) for _ in range(num_res_blocks)])

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
    """Construct a decoder by name. Supported: ``basic``, ``residual``."""
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

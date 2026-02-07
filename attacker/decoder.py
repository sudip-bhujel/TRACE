"""
Image Decoder Architectures for Gradient Inversion

This module provides multiple decoder architectures for decoding latent
representations into images and action predictions.

Available decoders:
- ImageDecoder: Basic transposed convolution decoder
- ResidualImageDecoder: Decoder with residual blocks for better detail
- UNetImageDecoder: U-Net style decoder with skip connections
- StyleImageDecoder: StyleGAN-inspired decoder with adaptive instance norm
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
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 42 -> 84
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),  # 84 -> 84
            nn.Sigmoid(),
        )

        # Action prediction head
        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)
        images = self.decoder(h)

        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class ResidualBlock2d(nn.Module):
    """2D Residual block for image decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class ResidualImageDecoder(nn.Module):
    """
    Decoder with residual blocks for better reconstruction quality.

    Adds residual blocks after each upsampling layer to preserve details
    and improve gradient flow during training.
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
        self.bn1 = nn.BatchNorm2d(128)
        self.res1 = nn.Sequential(
            *[ResidualBlock2d(128) for _ in range(num_res_blocks)]
        )

        # Second upsample block with residuals
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = nn.Sequential(*[ResidualBlock2d(64) for _ in range(num_res_blocks)])

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        h = F.relu(self.bn1(self.up1(h)))
        h = self.res1(h)

        h = F.relu(self.bn2(self.up2(h)))
        h = self.res2(h)

        images = self.final(h)
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class UNetImageDecoder(nn.Module):
    """
    U-Net style decoder with internal skip connections.

    Creates an internal encoder-decoder structure with skip connections
    that help preserve spatial information and fine details.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 8  # Start smaller for U-Net

        # Project to initial feature map
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        # Encoder (contracting path within decoder)
        self.enc1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder (expanding path)
        self.up1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),  # 256 + 256 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Initial projection
        h = self.fc(x_flat)
        h = h.view(-1, 512, self.init_size, self.init_size)

        # Encoder
        e1 = self.enc1(h)  # Save for skip

        # Decoder with skip connections
        d1 = self.up1(e1)
        d1 = torch.cat([d1, F.interpolate(e1, size=d1.shape[2:])], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        images = self.dec3(d3)

        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style-based generation."""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style = self.style(style)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


class StyleBlock(nn.Module):
    """Style-modulated convolution block."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adain(x, style)
        return self.activation(x)


class StyleImageDecoder(nn.Module):
    """
    StyleGAN-inspired decoder with Adaptive Instance Normalization.

    Uses style vectors to modulate the feature maps, allowing for
    better control over generated image characteristics.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        style_dim: int = 256,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4

        # Mapping network (latent -> style)
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initial constant input
        self.const = nn.Parameter(torch.randn(1, 256, 1, 1))

        # Style blocks
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        self.style1 = StyleBlock(256, 128, style_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.style2 = StyleBlock(128, 64, style_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.style3 = StyleBlock(64, 32, style_dim)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Generate style vector
        style = self.mapping(x_flat)

        # Project to initial feature map
        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        # Style-modulated upsampling
        h = self.style1(h, style)
        h = self.up1(h)

        h = self.style2(h, style)
        h = self.up2(h)

        h = self.style3(h, style)

        images = self.to_rgb(h)
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

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
        decoder_type: One of 'basic', 'residual', 'unet', 'style'
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
        "unet": UNetImageDecoder,
        "style": StyleImageDecoder,
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

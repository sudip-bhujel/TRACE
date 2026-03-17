"""
Super-Resolution Enhancer for Gradient Inversion

A lightweight residual U-Net that sharpens blurry reconstructions from the
baseline model. Learns the residual: output = input + correction.

Architecture: 3-level U-Net (~1.5M params)
    (3, 84, 84) -> encoder -> bottleneck -> decoder + skip connections -> (3, 84, 84)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EnhancerUNet(nn.Module):
    """
    Lightweight residual U-Net for image enhancement.

    Uses residual learning: output = input + network(input)
    so the network only needs to learn the correction/sharpening.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            ResBlock(c),
        )
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)  # 84 -> 42

        self.enc2 = nn.Sequential(
            ResBlock(c * 2),
        )
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)  # 42 -> 21

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(c * 4),
            ResBlock(c * 4),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)  # 21 -> 42
        self.dec2 = nn.Sequential(
            ResBlock(c * 2),  # after skip concat and 1x1 reduce
        )
        self.skip_reduce2 = nn.Conv2d(c * 4, c * 2, 1)  # reduce concatenated skip

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)  # 42 -> 84
        self.dec1 = nn.Sequential(
            ResBlock(c),
        )
        self.skip_reduce1 = nn.Conv2d(c * 2, c, 1)  # reduce concatenated skip

        # Output projection (residual correction)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 84, 84) blurry input image in [0, 1]
        Returns:
            (B, 3, 84, 84) enhanced image in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)  # (B, c, 84, 84)
        e2 = self.enc2(self.down1(e1))  # (B, 2c, 42, 42)

        # Bottleneck
        b = self.bottleneck(self.down2(e2))  # (B, 4c, 21, 21)

        # Decoder with skip connections
        d2 = self.up2(b)  # (B, 2c, 42, 42)
        d2 = self.skip_reduce2(torch.cat([d2, e2], 1))  # concat + reduce
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # (B, c, 84, 84)
        d1 = self.skip_reduce1(torch.cat([d1, e1], 1))  # concat + reduce
        d1 = self.dec1(d1)

        # Residual correction
        correction = self.out_conv(d1)
        return (x + correction).clamp(0, 1)

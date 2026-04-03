"""
Image Encoder for Autoregressive Gradient Inversion.

Maps 84x84x3 images to latent vectors, mirroring the decoder's inverse
structure. Used to encode previously reconstructed (or ground-truth) images
as context for the autoregressive transformer.
"""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """CNN encoder that maps images to latent vectors.

    Architecture mirrors the ImageDecoder in reverse:
        (3, 84, 84) → Conv layers with stride-2 → flatten → Linear → latent_dim
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 84):
        super().__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            # 84 → 42
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            # 42 → 21
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # 21 → 10
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
        )

        # Compute flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            conv_out = self.encoder(dummy)
            self._flat_size = conv_out.numel()

        self.projection = nn.Sequential(
            nn.Linear(self._flat_size, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent vectors.

        Args:
            x: (B, 3, H, W) or (B, T, 3, H, W)

        Returns:
            (B, latent_dim) or (B, T, latent_dim)
        """
        has_time = x.dim() == 5
        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        h = self.encoder(x)
        h = h.flatten(1)
        out = self.projection(h)

        if has_time:
            out = out.reshape(B, T, -1)

        return out

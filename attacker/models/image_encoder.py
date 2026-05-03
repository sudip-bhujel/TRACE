"""Image encoder used for autoregressive context conditioning."""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """CNN encoder mapping (3, H, W) images to a latent vector."""

    def __init__(self, latent_dim: int = 512, image_size: int = 84):
        super().__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            conv_out = self.encoder(dummy)
            self._flat_size = conv_out.numel()

        self.projection = nn.Sequential(
            nn.Linear(self._flat_size, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

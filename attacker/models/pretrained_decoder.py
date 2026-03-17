"""
Pre-trained VAE Decoder for Gradient Inversion

Uses a frozen Stable Diffusion VAE decoder that has world knowledge from
pre-training on billions of images. A small LatentMapper learns to project
gradient latents into the VAE's latent space.

This solves the unseen object problem: the decoder already knows how to
render indoor objects, furniture, etc. — we only need to learn the mapping.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMapper(nn.Module):
    """
    Maps gradient latent (1024-dim) to VAE latent space (4 x H x W).

    Uses an MLP with residual connections to learn the mapping between
    the gradient information space and the pre-trained VAE's latent space.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        latent_channels: int = 4,
        latent_h: int = 11,
        latent_w: int = 11,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        output_dim = latent_channels * latent_h * latent_w  # 4 * 11 * 11 = 484

        # MLP with residual connections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) gradient latent
        Returns:
            (B, latent_channels, latent_h, latent_w) VAE-compatible latent
        """
        h = self.input_proj(x)
        h = h + self.res_block1(h)
        h = h + self.res_block2(h)
        h = self.output_proj(h)
        return h.view(-1, self.latent_channels, self.latent_h, self.latent_w)


class PretrainedVAEDecoder(nn.Module):
    """
    Decoder using a frozen pre-trained Stable Diffusion VAE.

    Architecture:
        gradient_latent (1024) → LatentMapper → VAE latent (4x11x11)
                                               → Frozen VAE Decoder → image (3x88x88)
                                                                     → center crop → (3x84x84)

    The VAE decoder is frozen (no gradient updates) — only the LatentMapper
    and action_head are trained.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        image_size: int = 84,
        num_actions: int = 5,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        mapper_hidden_dim: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.vae_model_name = vae_model
        self.is_pretrained_decoder = True  # Flag for model/train dispatch

        # Compute padded size (must be divisible by 8 for SD VAE)
        self.padded_size = ((image_size + 7) // 8) * 8  # 84 → 88
        self.latent_h = self.padded_size // 8  # 11
        self.latent_w = self.padded_size // 8  # 11

        # Latent mapper: gradient_latent → VAE latent
        self.mapper = LatentMapper(
            input_dim=latent_dim,
            latent_channels=4,
            latent_h=self.latent_h,
            latent_w=self.latent_w,
            hidden_dim=mapper_hidden_dim,
        )

        # Action head (from gradient latent, not VAE latent)
        self.action_head = nn.Linear(latent_dim, num_actions)

        # Load and freeze VAE decoder
        self._load_vae(vae_model)

    def _load_vae(self, model_name: str):
        """Load pre-trained VAE and freeze decoder weights."""
        try:
            from diffusers import AutoencoderKL
        except ImportError:
            raise ImportError(
                "diffusers is required for PretrainedVAEDecoder. "
                "Install with: pip install diffusers transformers accelerate"
            )

        print(f"Loading pre-trained VAE: {model_name}")
        vae = AutoencoderKL.from_pretrained(model_name)

        # Store decoder and post_quant_conv (needed for decoding)
        self.vae_decoder = vae.decoder
        self.vae_post_quant_conv = vae.post_quant_conv

        # Store encoder for latent space supervision during training
        self.vae_encoder = vae.encoder
        self.vae_quant_conv = vae.quant_conv

        # Store scaling factor
        self.vae_scaling_factor = vae.config.scaling_factor  # 0.18215

        # Freeze all VAE components
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        for param in self.vae_post_quant_conv.parameters():
            param.requires_grad = False
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_quant_conv.parameters():
            param.requires_grad = False

        print(
            f"  VAE loaded and frozen. Padded size: {self.padded_size}x{self.padded_size}, "
            f"latent: 4x{self.latent_h}x{self.latent_w}"
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VAE latent space (for latent supervision).

        Args:
            images: (B, 3, 84, 84) in [0, 1]
        Returns:
            latents: (B, 4, latent_h, latent_w) scaled latents
        """
        # Pad to divisible-by-8 size
        padded = self._pad_images(images)

        # Convert to [-1, 1] (SD VAE convention)
        padded = padded * 2.0 - 1.0

        # Encode
        h = self.vae_encoder(padded)
        moments = self.vae_quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)

        # Use mean (no sampling for supervision targets)
        latents = mean * self.vae_scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode VAE latents to images.

        Args:
            latents: (B, 4, latent_h, latent_w) scaled latents
        Returns:
            images: (B, 3, image_size, image_size) in [0, 1]
        """
        # Unscale
        z = latents / self.vae_scaling_factor

        # Post-quant conv + decode
        z = self.vae_post_quant_conv(z)
        decoded = self.vae_decoder(z)

        # Convert from [-1, 1] to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)

        # Center crop to original size
        decoded = self._crop_images(decoded)
        return decoded

    def _pad_images(self, images: torch.Tensor) -> torch.Tensor:
        """Pad images to padded_size (84 → 88) using reflection padding."""
        if self.image_size == self.padded_size:
            return images
        pad = self.padded_size - self.image_size  # 4
        pad_left = pad // 2  # 2
        pad_right = pad - pad_left  # 2
        return F.pad(images, (pad_left, pad_right, pad_left, pad_right), mode="reflect")

    def _crop_images(self, images: torch.Tensor) -> torch.Tensor:
        """Center crop from padded_size to image_size (88 → 84)."""
        if self.image_size == self.padded_size:
            return images
        pad = self.padded_size - self.image_size
        start = pad // 2
        return images[
            :, :, start : start + self.image_size, start : start + self.image_size
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard decoder interface.

        Args:
            x: (B, T, latent_dim) or (B, latent_dim) gradient latent
        Returns:
            images: (B, T, 3, H, W) or (B, 3, H, W)
            actions: (B, T, num_actions) or (B, num_actions)
        """
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Map to VAE latent space
        vae_latents = self.mapper(x_flat)  # (B*T, 4, 11, 11)

        # Decode with frozen VAE
        with torch.no_grad():
            images = self.decode_latents(vae_latents)  # (B*T, 3, 84, 84)

        # Action prediction
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions

    def forward_with_latents(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns predicted VAE latents (for training).

        Args:
            x: (B*T, latent_dim) gradient latent (flattened)
        Returns:
            images: (B*T, 3, H, W)
            actions: (B*T, num_actions)
            vae_latents: (B*T, 4, latent_h, latent_w)
        """
        vae_latents = self.mapper(x)  # (B*T, 4, 11, 11)
        images = self.decode_latents(vae_latents)
        actions = self.action_head(x)
        return images, actions, vae_latents

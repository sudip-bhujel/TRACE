"""
VQ-VAE (Vector Quantized Variational AutoEncoder) for indoor scene images.

Learns a discrete visual codebook from AI2-THOR environment images.
The codebook serves as a visual vocabulary for the gradient inversion
token prediction decoder.

Architecture:
    Image (3, 84, 84) -> Encoder -> (embed_dim, 6, 6) -> Quantize -> Decoder -> Image

Usage:
    # Training
    model = VQVAE(codebook_size=512, embedding_dim=256)
    x_recon, indices, vq_loss = model(images)

    # Encoding images to tokens
    indices = model.encode_to_tokens(images)  # (B, 36)

    # Decoding tokens to images
    images = model.decode_from_tokens(indices)  # (B, 3, 84, 84)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block for VQ-VAE encoder/decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VQEncoder(nn.Module):
    """
    Downsampling encoder: (3, 84, 84) -> (embed_dim, 6, 6).

    Uses strided convolutions for downsampling with residual blocks
    for feature refinement at each scale.

    Downsampling path:
      84 -> 42 (stride 2) -> 21 (stride 2) -> 7 (stride 3) -> 6 (adjust)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        # 84 -> 42
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )

        # 42 -> 21
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )

        # 21 -> 7
        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=3, padding=0),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
        )

        # Residual blocks at bottleneck
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_channels * 2) for _ in range(num_res_blocks)]
        )

        # Project to embedding dim: 7 -> 7 (keep spatial, change channels)
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, embedding_dim, 1),
            nn.GroupNorm(8, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 84, 84)
        Returns:
            z: (B, embed_dim, 7, 7)
        """
        x = self.down1(x)  # (B, 128, 42, 42)
        x = self.down2(x)  # (B, 128, 21, 21)
        x = self.down3(x)  # (B, 256, 7, 7)
        x = self.res_blocks(x)
        x = self.proj(x)  # (B, embed_dim, 7, 7)
        return x


class VQDecoder(nn.Module):
    """
    Upsampling decoder: (embed_dim, 7, 7) -> (3, 84, 84).

    Symmetric to encoder with transposed convolutions.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        # Unproject from embedding dim
        self.unproj = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels * 2, 1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_channels * 2) for _ in range(num_res_blocks)]
        )

        # 7 -> 21
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, 3, stride=3, padding=0
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )

        # 21 -> 42
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )

        # 42 -> 84
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim, 7, 7)
        Returns:
            image: (B, 3, 84, 84)
        """
        x = self.unproj(x)  # (B, 256, 7, 7)
        x = self.res_blocks(x)
        x = self.up1(x)  # (B, 128, 21, 21)
        x = self.up2(x)  # (B, 128, 42, 42)
        x = self.up3(x)  # (B, 3, 84, 84)
        return x


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA (Exponential Moving Average) updates.

    Maps continuous encoder output to nearest codebook entries.
    Uses EMA for stable codebook learning (no gradient through codebook).

    Args:
        num_embeddings: Codebook size (number of discrete tokens)
        embedding_dim: Dimension of each codebook entry
        commitment_cost: Weight for commitment loss (encoder -> codebook)
        decay: EMA decay rate for codebook updates
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_dw", torch.zeros(num_embeddings, embedding_dim))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, embed_dim, H, W) continuous encoder output

        Returns:
            z_q: (B, embed_dim, H, W) quantized output
            indices: (B, H*W) codebook indices
            vq_loss: scalar commitment + codebook loss
        """
        B, C, H, W = z.shape

        # Reshape: (B, C, H, W) -> (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        # Find nearest codebook entry for each spatial position
        # distances: (B*H*W, num_embeddings)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        # Get nearest indices
        indices = distances.argmin(dim=1)  # (B*H*W,)

        # Quantize
        z_q = self.embedding(indices)  # (B*H*W, C)

        # EMA codebook update (during training only)
        if self.training:
            # One-hot encode assignments
            encodings = F.one_hot(indices, self.num_embeddings).float()

            # Update cluster sizes
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # Update embedding sums
            dw = encodings.t() @ z_flat
            self.ema_dw.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing for cluster sizes
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            )

            # Update codebook
            self.embedding.weight.data.copy_(self.ema_dw / cluster_size.unsqueeze(1))

        # Compute losses
        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = F.mse_loss(z_flat, z_q.detach())

        # Codebook loss: move codebook entries toward encoder outputs
        codebook_loss = F.mse_loss(z_q, z_flat.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape back to spatial
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        indices = indices.reshape(B, H * W)  # (B, H*W)

        return z_q, indices, vq_loss

    def get_codebook_usage(self) -> float:
        """Return fraction of codebook entries being used."""
        return (self.ema_cluster_size > 1.0).float().mean().item()


class VQVAE(nn.Module):
    """
    Full VQ-VAE model for indoor scene images.

    Encodes images into a grid of discrete tokens from a learned codebook,
    then decodes back to images.

    Args:
        in_channels: Input image channels (3 for RGB)
        hidden_channels: Hidden channel dimension in encoder/decoder
        embedding_dim: Codebook entry dimension
        codebook_size: Number of codebook entries
        commitment_cost: VQ commitment loss weight
        num_res_blocks: Residual blocks per encoder/decoder stage
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.encoder = VQEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = VQDecoder(
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
        )

        # Store config for serialization
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "embedding_dim": embedding_dim,
            "codebook_size": codebook_size,
            "commitment_cost": commitment_cost,
            "num_res_blocks": num_res_blocks,
        }

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: (B, 3, 84, 84) input images

        Returns:
            x_recon: (B, 3, 84, 84) reconstructed images
            indices: (B, H*W) codebook indices (token grid)
            vq_loss: scalar VQ loss
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete token indices.

        Args:
            x: (B, 3, 84, 84) images

        Returns:
            indices: (B, num_tokens) token indices
        """
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode images from token indices.

        Args:
            indices: (B, num_tokens) token indices

        Returns:
            images: (B, 3, 84, 84) decoded images
        """
        B = indices.shape[0]
        # Compute spatial size from total tokens
        num_tokens = indices.shape[1]
        spatial = int(num_tokens**0.5)

        embeddings = self.quantizer.embedding(indices)  # (B, num_tokens, embed_dim)
        z_q = embeddings.reshape(B, spatial, spatial, -1).permute(
            0, 3, 1, 2
        )  # (B, embed_dim, H, W)
        images = self.decoder(z_q)
        return images

    @property
    def num_tokens_per_image(self) -> int:
        """Number of tokens produced per image (spatial_h * spatial_w)."""
        # 84 -> 42 -> 21 -> 7: spatial size is 7x7 = 49 tokens
        return 49

    @property
    def spatial_size(self) -> int:
        """Spatial dimension of the token grid."""
        return 7


def load_vqvae(checkpoint_path: str, device: str = "cpu") -> VQVAE:
    """Load a trained VQ-VAE from checkpoint.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint
        device: Device to load model on

    Returns:
        Loaded VQVAE model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    model = VQVAE(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model

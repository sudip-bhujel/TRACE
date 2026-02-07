"""
Gradient Encoder Architectures for Gradient Inversion

This module provides multiple encoder architectures for encoding gradient vectors
into latent representations for image reconstruction.

Available encoders:
- GradientEncoder: Basic MLP encoder with LayerNorm and GELU
- ResidualGradientEncoder: Encoder with residual connections for better gradient flow
- BottleneckGradientEncoder: Encoder with bottleneck layers (compress-expand pattern)
- GatedGradientEncoder: Encoder with gating mechanism for selective feature learning
- HierarchicalGradientEncoder: Multi-scale encoder that processes gradients hierarchically
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BottleneckGradientEncoder(nn.Module):
    """
    Encoder with bottleneck layers (compress-expand-compress pattern).

    This architecture first heavily compresses the gradient, then expands to
    learn rich representations, and finally compresses to the latent space.
    Inspired by U-Net and autoencoder designs.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        bottleneck_dim: int = 256,
        hidden_dims: List[int] = [4096, 1024, 2048, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encoder path (compress to bottleneck)
        encoder_layers = []
        in_dim = gradient_dim
        for hidden_dim in hidden_dims[:2]:
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        # Bottleneck
        encoder_layers.extend(
            [
                nn.Linear(in_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder path (expand from bottleneck)
        decoder_layers = []
        in_dim = bottleneck_dim
        for hidden_dim in hidden_dims[2:]:
            decoder_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        decoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x)
        return self.decoder(bottleneck)


class GatedLinear(nn.Module):
    """Linear layer with gating mechanism (GLU-style)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedGradientEncoder(nn.Module):
    """
    Encoder with gating mechanism for selective feature learning.

    Uses Gated Linear Units (GLU) which allow the network to selectively
    pass or block information, improving representation learning.
    """

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
                    GatedLinear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HierarchicalGradientEncoder(nn.Module):
    """
    Multi-scale hierarchical encoder that processes gradients at different scales.

    Splits the gradient into chunks and processes them separately, then combines
    the representations. This captures both local and global gradient patterns.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_chunks: int = 8,
        chunk_hidden_dim: int = 512,
        global_hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_size = gradient_dim // num_chunks

        # Make sure gradient_dim is divisible by num_chunks
        assert gradient_dim % num_chunks == 0, (
            f"gradient_dim ({gradient_dim}) must be divisible by num_chunks ({num_chunks})"
        )

        # Per-chunk encoder (shared weights)
        self.chunk_encoder = nn.Sequential(
            nn.Linear(self.chunk_size, chunk_hidden_dim),
            nn.LayerNorm(chunk_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chunk_hidden_dim, chunk_hidden_dim),
            nn.LayerNorm(chunk_hidden_dim),
            nn.GELU(),
        )

        # Global encoder (combines all chunk representations)
        self.global_encoder = nn.Sequential(
            nn.Linear(chunk_hidden_dim * num_chunks, global_hidden_dim),
            nn.LayerNorm(global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, latent_dim),
        )

        # Optional: attention over chunks
        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=chunk_hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(chunk_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, gradient_dim) or (B, gradient_dim)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
        else:
            B = x.shape[0]
            T = None

        # Split into chunks: (B, num_chunks, chunk_size)
        chunks = x.view(x.shape[0], self.num_chunks, self.chunk_size)

        # Encode each chunk: (B, num_chunks, chunk_hidden_dim)
        chunk_encodings = self.chunk_encoder(chunks)

        # Self-attention over chunks for inter-chunk relationships
        attn_out, _ = self.chunk_attention(
            chunk_encodings, chunk_encodings, chunk_encodings
        )
        chunk_encodings = self.attn_norm(chunk_encodings + attn_out)

        # Flatten and combine: (B, num_chunks * chunk_hidden_dim)
        combined = chunk_encodings.view(chunk_encodings.shape[0], -1)

        # Global encoding
        output = self.global_encoder(combined)

        # Reshape back if needed
        if T is not None:
            output = output.view(B, T, -1)

        return output


class MoEGradientEncoder(nn.Module):
    """
    Mixture of Experts encoder with learnable routing.

    Uses multiple expert encoders and a gating network to combine their outputs.
    Different experts can specialize in different types of gradient patterns.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_experts: int = 4,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        top_k: int = 2,  # Number of experts to use per input
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gradient_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, latent_dim),
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(gradient_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, gradient_dim) or (B, gradient_dim)
        """
        # Get gating weights
        gate_logits = self.gate(x)  # (..., num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute expert outputs and combine
        output = torch.zeros(
            *x.shape[:-1], self.experts[0][-1].out_features, device=x.device
        )

        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # (...,)
            weight = top_k_weights[..., k : k + 1]  # (..., 1)

            # Gather expert outputs (simplified - could be optimized)
            for i in range(self.num_experts):
                mask = expert_idx == i
                if mask.any():
                    expert_out = self.experts[i](x)
                    output = output + weight * expert_out * mask.unsqueeze(-1).float()

        return output


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
        encoder_type: One of 'basic', 'residual', 'bottleneck', 'gated',
                     'hierarchical', 'moe'
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
        "bottleneck": BottleneckGradientEncoder,
        "gated": GatedGradientEncoder,
        "hierarchical": HierarchicalGradientEncoder,
        "moe": MoEGradientEncoder,
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

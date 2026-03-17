from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _math_only_sdpa_context():
    """
    Return an SDPA context manager that forces the math backend.

    Uses the new torch.nn.attention API when available, with a fallback to the
    legacy torch.backends.cuda API for older PyTorch versions.
    """
    if (
        hasattr(torch.nn, "attention")
        and hasattr(torch.nn.attention, "sdpa_kernel")
        and hasattr(torch.nn.attention, "SDPBackend")
    ):
        return torch.nn.attention.sdpa_kernel(
            backends=[torch.nn.attention.SDPBackend.MATH]
        )

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )

    return nullcontext()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using F.scaled_dot_product_attention for Flash Attention.

    This uses PyTorch's SDPA which automatically selects the best backend:
    - Flash Attention (fastest, CUDA + fp16/bf16)
    - Memory-efficient attention (CUDA)
    - Math attention (fallback)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        self.is_causal = is_causal

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # Combined QKV projection (more efficient - single matmul)
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, use_flash_attention: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, latent_dim)
            use_flash_attention: If False, disable Flash Attention kernels on CUDA.
        Returns:
            (B, T, latent_dim)
        """
        B, T, _ = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention toggle via SDPA backend selection.
        # If disabled, force non-Flash math backend on CUDA.
        sdpa_ctx = nullcontext()
        if (
            q.is_cuda
            and not use_flash_attention
        ):
            sdpa_ctx = _math_only_sdpa_context()

        with sdpa_ctx:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,  # Enables efficient causal masking
            )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, self.latent_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""

    def __init__(
        self,
        latent_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(latent_dim, num_heads, dropout, is_causal)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, use_flash_attention: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, latent_dim)
            use_flash_attention: If False, disables Flash kernels in attention.
        Returns:
            (B, T, latent_dim)
        """
        # Pre-norm architecture (like norm_first=True)
        x = x + self.dropout(
            self.attention(self.norm1(x), use_flash_attention=use_flash_attention)
        )
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer with causal (autoregressive) attention using custom blocks.

    Each position can only attend to previous positions.
    Uses F.scaled_dot_product_attention which automatically enables Flash Attention
    when conditions are met (CUDA, fp16/bf16, PyTorch 2.0+).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        is_causal: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, latent_dim) * 0.02
        )

        # Stack of custom transformer blocks (uses SDPA for Flash Attention)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    is_causal=is_causal,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, x: torch.Tensor, use_flash_attention: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, latent_dim)
            use_flash_attention: If False, disables Flash kernels in all blocks.
        Returns:
            (B, T, latent_dim)
        """
        B, T, D = x.shape

        # Add positional embeddings
        x = x + self.pos_embedding[:, :T, :]

        # Apply transformer layers (Flash Attention auto-enabled via SDPA)
        for layer in self.layers:
            x = layer(x, use_flash_attention=use_flash_attention)

        x = self.norm(x)

        return x

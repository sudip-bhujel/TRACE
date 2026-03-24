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


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Encodes *relative* temporal distance directly in the attention scores,
    so adjacent frames naturally attend more to each other.
    """

    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        # cos and sin caches: (1, 1, seq_len, dim//2)
        self.register_buffer(
            "cos_cache", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "sin_cache", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False
        )

    def forward(self, seq_len: int):
        """Return (cos, sin) each of shape (1, 1, seq_len, dim//2)."""
        if seq_len > self.cos_cache.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cache[:, :, :seq_len], self.sin_cache[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (B, heads, T, head_dim)
        cos, sin: (1, 1, T, head_dim//2) — will be broadcast.

    Returns:
        (q_rotated, k_rotated) with same shape as input.
    """
    # Expand cos/sin to full head_dim by repeating
    cos = cos.repeat(1, 1, 1, 2)  # (1, 1, T, head_dim)
    sin = sin.repeat(1, 1, 1, 2)  # (1, 1, T, head_dim)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using F.scaled_dot_product_attention.
    Supports optional Rotary Position Embeddings (RoPE).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        is_causal: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 256,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        self.is_causal = is_causal
        self.use_rope = use_rope

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # Combined QKV projection
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = dropout

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

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

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(T)
            cos = cos.to(q.dtype).to(q.device)
            sin = sin.to(q.dtype).to(q.device)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Flash Attention toggle
        sdpa_ctx = nullcontext()
        if q.is_cuda and not use_flash_attention:
            sdpa_ctx = _math_only_sdpa_context()

        with sdpa_ctx:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, self.latent_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with gated residual connections.

    Instead of the standard ``x = x + attn(x)``, uses a learnable gate
    initialized near zero so that the block starts as a near-identity
    mapping.  The model gradually discovers when temporal context helps,
    preventing the "shortcut" problem observed in per-frame-dominant
    architectures.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        is_causal: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 256,
    ):
        super().__init__()
        ff_dim = latent_dim * ff_multiplier

        self.attention = MultiHeadAttention(
            latent_dim, num_heads, dropout, is_causal, use_rope, max_seq_len
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.gate_attn = nn.Parameter(torch.tensor(-1.0))
        self.gate_ffn = nn.Parameter(torch.tensor(-1.0))

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
        # Gated pre-norm residual
        attn_out = self.dropout_layer(
            self.attention(self.norm1(x), use_flash_attention=use_flash_attention)
        )
        x = x + torch.sigmoid(self.gate_attn) * attn_out

        ffn_out = self.dropout_layer(self.ffn(self.norm2(x)))
        x = x + torch.sigmoid(self.gate_ffn) * ffn_out

        return x


class TemporalTransformer(nn.Module):
    """
    Transformer with causal (autoregressive) attention using custom blocks.

    Supports gated residual connections, configurable FFN width, and
    optional Rotary Position Embeddings.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        is_causal: bool = True,
        use_rope: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.use_rope = use_rope

        if not use_rope:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, latent_dim) * 0.02
            )
        else:
            self.pos_embedding = None

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                    is_causal=is_causal,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
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

        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :T, :]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, use_flash_attention=use_flash_attention)

        x = self.norm(x)

        return x

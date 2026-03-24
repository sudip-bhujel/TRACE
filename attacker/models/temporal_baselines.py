"""
Temporal Baseline Models for Gradient Inversion.

Drop-in alternatives to ``TemporalTransformer`` for ablation studies.
All models share the same interface:

    forward(x: Tensor, **kw) -> Tensor

where ``x`` is ``(B, T, latent_dim)`` and the output has the same shape.

Available baselines:
    - TemporalGRU: Bidirectional GRU
    - TemporalConv1D: Causal 1-D dilated convolution stack
    - TemporalMLPMixer: Token-mixing MLP (no attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGRU(nn.Module):
    """Bidirectional GRU over the latent time-series.

    Projects the concatenated forward/backward hidden states back to
    ``latent_dim`` so the decoder interface stays unchanged.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)."""
        residual = x
        h, _ = self.gru(x)  # (B, T, 2*D)
        out = self.out_proj(h)  # (B, T, D)
        return self.norm(residual + out)


class CausalConv1dBlock(nn.Module):
    """Single causal conv1d block with pre-norm and gated residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)."""
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, C, T)
        h = F.pad(h, (self.padding, 0))  # causal pad
        h = self.conv(h)  # (B, 2*C, T)
        gate, val = h.chunk(2, dim=1)
        h = val * torch.sigmoid(gate)
        h = h.transpose(1, 2)  # (B, T, C)
        return residual + self.dropout(h)


class TemporalConv1D(nn.Module):
    """Causal 1-D dilated convolution stack.

    Uses exponentially increasing dilation ``(1, 2, 4, ...)`` so the
    receptive field grows efficiently.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalConv1dBlock(
                    channels=latent_dim,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MixerBlock(nn.Module):
    """Single MLP-Mixer block with token-mixing and channel-mixing."""

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        token_expansion: int = 2,
        channel_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Token-mixing MLP (operates across the time axis)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * token_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * token_expansion, seq_len),
            nn.Dropout(dropout),
        )

        # Channel-mixing MLP (operates across the feature axis)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * channel_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * channel_expansion, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)."""
        h = self.norm1(x).transpose(1, 2)  # (B, D, T)
        x = x + self.token_mix(h).transpose(1, 2)  # (B, T, D)

        x = x + self.channel_mix(self.norm2(x))
        return x


class TemporalMLPMixer(nn.Module):
    """MLP-Mixer for temporal modelling (no attention).

    Note: ``seq_len`` must match the actual sequence length at runtime
    because the token-mixing MLP has a fixed time-axis width.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [
                MixerBlock(
                    latent_dim=latent_dim,
                    seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """(B, T, D) -> (B, T, D).  T must equal self.seq_len."""
        B, T, D = x.shape
        if T != self.seq_len:
            if T < self.seq_len:
                x = F.pad(x, (0, 0, 0, self.seq_len - T))
            else:
                x = x[:, : self.seq_len]

        for layer in self.layers:
            x = layer(x)
        out = self.norm(x)

        return out[:, :T]


def get_temporal_model(
    temporal_model_type: str,
    latent_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    ff_multiplier: int = 4,
    dropout: float = 0.1,
    max_seq_len: int = 32,
    is_causal: bool = True,
    use_rope: bool = False,
) -> nn.Module:
    """Factory function to create a temporal model by type.

    Args:
        temporal_model_type: One of ``"transformer"``, ``"gru"``,
            ``"conv1d"``, ``"mlp_mixer"``.

    Returns:
        Temporal model module.
    """
    if temporal_model_type == "transformer":
        from attacker.models.transformer import TemporalTransformer

        return TemporalTransformer(
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            max_seq_len=max_seq_len,
            is_causal=is_causal,
            use_rope=use_rope,
        )
    elif temporal_model_type == "gru":
        return TemporalGRU(
            latent_dim=latent_dim,
            num_layers=min(num_layers, 4),
            dropout=dropout,
        )
    elif temporal_model_type == "conv1d":
        return TemporalConv1D(
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif temporal_model_type == "mlp_mixer":
        return TemporalMLPMixer(
            latent_dim=latent_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown temporal_model_type: {temporal_model_type!r}. "
            f"Available: transformer, gru, conv1d, mlp_mixer"
        )

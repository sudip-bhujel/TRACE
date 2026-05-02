"""Uniform per-vector gradient quantization defense (QSGD-style)."""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class GradientQuantization(GradientDefense):
    """
    Quantize each gradient vector to ``2**n_bits`` levels over its own
    ``[min, max]`` range, then dequantize. ``stochastic=True`` selects
    unbiased stochastic rounding instead of nearest-level rounding.
    """

    def __init__(self, n_bits: int = 8, stochastic: bool = False):
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        mode = "_stoch" if stochastic else ""
        super().__init__(name=f"quantization_{n_bits}bit{mode}")
        self.n_bits = n_bits
        self.stochastic = stochastic
        self.levels = 2**n_bits

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        if self.n_bits == 1:
            return gradients.sign()

        original_shape = gradients.shape
        flat = gradients.reshape(-1, original_shape[-1])

        g_min = flat.min(dim=-1, keepdim=True).values
        g_max = flat.max(dim=-1, keepdim=True).values
        scale = (g_max - g_min).clamp(min=1e-8)

        normalized = (flat - g_min) / scale
        scaled = normalized * (self.levels - 1)

        if self.stochastic:
            floored = scaled.floor()
            frac = scaled - floored
            quantized = floored + (torch.rand_like(frac) < frac).float()
        else:
            quantized = scaled.round()

        quantized = quantized.clamp(0, self.levels - 1)

        dequantized = (quantized / (self.levels - 1)) * scale + g_min

        return dequantized.reshape(original_shape)

    def summary(self) -> Dict[str, Any]:
        return {
            **super().summary(),
            "n_bits": self.n_bits,
            "levels": self.levels,
            "stochastic": self.stochastic,
        }

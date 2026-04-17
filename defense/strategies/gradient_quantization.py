"""
Gradient Quantization / Compression Defense.

Simulates the lossy compression applied in distributed RL before gradient
sharing: each gradient vector is uniformly quantized to ``n_bits`` levels
using per-sample dynamic range scaling.

References:
    - Alistarh et al., "QSGD: Communication-Efficient SGD via Gradient
      Quantization and Encoding," NeurIPS 2017.
    - Seide et al., "1-Bit Stochastic Gradient Descent," Interspeech 2014.
"""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class GradientQuantization(GradientDefense):
    """
    Uniform gradient quantization defense.

    Each gradient vector is independently quantized to ``2**n_bits``
    evenly-spaced levels spanning its own observed dynamic range
    ``[min, max]``, then dequantized back to float.  This simulates the
    information loss incurred by low-bit gradient compression schemes used
    in communication-efficient distributed RL (e.g. QSGD, 1-bit SGD).

    Args:
        n_bits: Quantization bit-width.  ``1`` applies sign compression
            (``±1``). Typical values: 1, 2, 4, 8.
        stochastic: If ``True``, uses stochastic rounding (unbiased
            estimator): each value is rounded up with probability equal
            to its fractional part and down otherwise.  If ``False``
            (default), nearest-level rounding is used.
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
        """
        Quantize and dequantize the gradient tensor.

        Args:
            gradients: (..., D) gradient tensor of any batch shape.

        Returns:
            Quantized-then-dequantized gradient tensor of the same shape.
            Values that map to the same quantization level become identical,
            introducing reconstruction error that grows as ``n_bits``
            decreases.
        """
        if self.n_bits == 1:
            # Sign compression: {-1, 0, +1}
            return gradients.sign()

        original_shape = gradients.shape
        flat = gradients.reshape(-1, original_shape[-1])  # (N, D)

        # Per-sample dynamic range
        g_min = flat.min(dim=-1, keepdim=True).values   # (N, 1)
        g_max = flat.max(dim=-1, keepdim=True).values   # (N, 1)
        scale = (g_max - g_min).clamp(min=1e-8)

        # Normalize to [0, levels-1]
        normalized = (flat - g_min) / scale              # [0, 1]
        scaled = normalized * (self.levels - 1)          # [0, levels-1]

        if self.stochastic:
            floored = scaled.floor()
            frac = scaled - floored
            quantized = floored + (torch.rand_like(frac) < frac).float()
        else:
            quantized = scaled.round()

        quantized = quantized.clamp(0, self.levels - 1)

        # Dequantize back to original range
        dequantized = (quantized / (self.levels - 1)) * scale + g_min

        return dequantized.reshape(original_shape)

    def summary(self) -> Dict[str, Any]:
        return {
            **super().summary(),
            "n_bits": self.n_bits,
            "levels": self.levels,
            "stochastic": self.stochastic,
        }

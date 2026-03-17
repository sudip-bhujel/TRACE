"""
Differential Privacy (DP-SGD) Defense.

Implements per-sample gradient clipping followed by calibrated Gaussian
noise addition, following the DP-SGD mechanism (Abadi et al. 2016).

The defense guarantees (epsilon, delta)-differential privacy when applied
to individual gradient updates.  The privacy budget can be estimated
with :meth:`DPSGDDefense.compute_epsilon` using the Rényi DP accountant
(if ``dp-accounting`` or a compatible library is available).

Usage::

    from defense.strategies.dp_sgd import DPSGDDefense

    defense = DPSGDDefense(max_grad_norm=1.0, noise_multiplier=0.1)
    defended = defense.apply(gradients)

    # Estimate privacy cost
    eps = defense.compute_epsilon(
        num_steps=1000, sample_rate=0.01, delta=1e-5
    )
"""

import math
from typing import Any, Dict, Optional

import torch

from defense.base import GradientDefense


class DPSGDDefense(GradientDefense):
    """
    DP-SGD defense: per-sample gradient clipping + calibrated noise.

    For each gradient vector g the mechanism produces:

        g' = clip(g, C) + N(0, (sigma * C)^2 I)

    where clip(g, C) = g * min(1, C / ||g||_2) and sigma is the
    *noise_multiplier*.

    Args:
        max_grad_norm: Clipping bound *C* for the L2 norm.
        noise_multiplier: Ratio of noise standard deviation to clipping
            bound.  The actual noise std is ``noise_multiplier * max_grad_norm``.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.1,
    ):
        if max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
        if noise_multiplier < 0.0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
        super().__init__(name=f"dpsgd_C{max_grad_norm}_sigma{noise_multiplier}")
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def _clip(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Per-vector L2 norm clipping.

        Args:
            gradients: (..., D) gradient tensor.

        Returns:
            Clipped gradient tensor of the same shape.
        """
        original_shape = gradients.shape
        flat = gradients.reshape(-1, original_shape[-1])

        norms = flat.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_grad_norm / norms.clamp(min=1e-8), max=1.0)
        clipped = flat * scale

        return clipped.reshape(original_shape)

    def _add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Add calibrated Gaussian noise.

        Args:
            gradients: (..., D) gradient tensor (already clipped).

        Returns:
            Noisy gradient tensor.
        """
        noise_std = self.noise_multiplier * self.max_grad_norm
        if noise_std == 0.0:
            return gradients
        noise = torch.randn_like(gradients) * noise_std
        return gradients + noise

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Apply DP-SGD defense: clip then add noise.

        Args:
            gradients: (..., D) gradient tensor.

        Returns:
            Defended gradient tensor of the same shape.
        """
        clipped = self._clip(gradients)
        return self._add_noise(clipped)

    def compute_epsilon(
        self,
        num_steps: int,
        sample_rate: float,
        delta: float = 1e-5,
        alphas: Optional[list] = None,
    ) -> Optional[float]:
        """
        Estimate (epsilon, delta)-DP using the Rényi DP accountant.

        Uses the analytical Gaussian mechanism RDP bound.

        Args:
            num_steps: Total number of gradient update steps.
            sample_rate: Probability that each sample is included in a
                mini-batch (= batch_size / dataset_size).
            delta: Target delta for (epsilon, delta)-DP.
            alphas: RDP orders to evaluate.  Defaults to a standard range.

        Returns:
            Estimated epsilon, or ``None`` if noise_multiplier is zero.
        """
        if self.noise_multiplier == 0:
            return None

        if alphas is None:
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        def _rdp_gaussian(alpha: float) -> float:
            return alpha / (2 * self.noise_multiplier**2)

        def _rdp_subsample(alpha: float, rdp: float, q: float) -> float:
            if q == 0:
                return 0.0
            if q == 1.0:
                return rdp
            # Tight bound for subsampled mechanisms (Mironov et al. 2019 approx)
            log_term = (alpha - 1) * rdp
            if log_term > 500:
                return q**2 * log_term / (alpha - 1)
            return math.log(1 + q**2 * (math.exp(log_term) - 1)) / (alpha - 1)

        best_eps = float("inf")
        for alpha in alphas:
            rdp = _rdp_gaussian(alpha)
            rdp_sub = _rdp_subsample(alpha, rdp, sample_rate) * num_steps
            eps = rdp_sub - math.log(delta) / (alpha - 1)
            best_eps = min(best_eps, eps)

        return best_eps

    def summary(self) -> Dict[str, Any]:
        return {
            **super().summary(),
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "noise_std": self.noise_multiplier * self.max_grad_norm,
        }

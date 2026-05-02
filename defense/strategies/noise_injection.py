"""Additive Gaussian noise defense."""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class NoiseInjection(GradientDefense):
    """
    Add Gaussian noise of standard deviation ``sigma`` to each gradient.
    When ``relative=True``, noise is scaled by the L2 norm of each vector.
    """

    def __init__(self, sigma: float = 0.01, relative: bool = False):
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        mode = "rel" if relative else "abs"
        super().__init__(name=f"noise_{mode}_sigma{sigma}")
        self.sigma = sigma
        self.relative = relative

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0.0:
            return gradients

        noise = torch.randn_like(gradients)

        if self.relative:
            original_shape = gradients.shape
            flat = gradients.reshape(-1, original_shape[-1])
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            noise_flat = noise.reshape(-1, original_shape[-1])
            noise_flat = noise_flat * norms * self.sigma
            return gradients + noise_flat.reshape(original_shape)

        return gradients + noise * self.sigma

    def summary(self) -> Dict[str, Any]:
        return {
            **super().summary(),
            "sigma": self.sigma,
            "relative": self.relative,
        }

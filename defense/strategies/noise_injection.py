"""
Gaussian Noise Injection Defense.

Adds isotropic Gaussian noise to gradients before sharing.
Supports both absolute and relative noise.
"""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class NoiseInjection(GradientDefense):
    """
    Additive Gaussian noise defense.

    Args:
        sigma: Standard deviation of the Gaussian noise.
        relative: If ``True``, *sigma* is multiplied by the L2 norm of
            each gradient vector so that the noise magnitude scales with
            the gradient.  If ``False`` (default), *sigma* is used as an
            absolute standard deviation.
    """

    def __init__(self, sigma: float = 0.01, relative: bool = False):
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        mode = "rel" if relative else "abs"
        super().__init__(name=f"noise_{mode}_sigma{sigma}")
        self.sigma = sigma
        self.relative = relative

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the gradient tensor.

        Args:
            gradients: (..., D) gradient tensor of any batch shape.

        Returns:
            Noisy gradient tensor of the same shape.
        """
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

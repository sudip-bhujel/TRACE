"""
Differential Privacy (DP-SGD) Defense.

Implements per-sample gradient clipping followed by calibrated Gaussian
noise addition.
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

    There are two ways to configure DP-SGD:

    1.  **Specify noise_multiplier directly** — the resulting epsilon
        is computed via the RDP accountant (if accounting params are
        given).
    2.  **Specify target_epsilon and target_delta** — the required
        noise_multiplier is *calibrated* automatically via binary
        search so that the mechanism achieves the desired budget.

    Args:
        max_grad_norm: Clipping bound *C* for the L2 norm.
        noise_multiplier: Ratio of noise standard deviation to clipping
            bound.  Ignored when ``target_epsilon`` is set.
        num_steps: Number of gradient update steps (for privacy accounting).
        sample_rate: Subsampling probability per step (batch_size / dataset_size).
        delta: Target delta for (epsilon, delta)-DP.
        target_epsilon: If provided, calibrate ``noise_multiplier`` so
            that the mechanism achieves this epsilon (requires
            ``num_steps`` and ``sample_rate``).
        target_delta: Delta to pair with ``target_epsilon``.  Defaults
            to ``delta`` if not given.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.1,
        num_steps: Optional[int] = None,
        sample_rate: Optional[float] = None,
        delta: float = 1e-5,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
    ):
        if max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")

        self.delta = target_delta if target_delta is not None else delta
        self._num_steps = num_steps
        self._sample_rate = sample_rate

        if target_epsilon is not None:
            if num_steps is None or sample_rate is None:
                raise ValueError(
                    "num_steps and sample_rate are required when "
                    "specifying target_epsilon"
                )
            noise_multiplier = self.calibrate_noise_multiplier(
                target_epsilon=target_epsilon,
                num_steps=num_steps,
                sample_rate=sample_rate,
                delta=self.delta,
            )
            self.epsilon = target_epsilon

        elif num_steps is not None and sample_rate is not None:
            self.epsilon = self._compute_epsilon_static(
                noise_multiplier=noise_multiplier,
                num_steps=num_steps,
                sample_rate=sample_rate,
                delta=self.delta,
            )
        else:
            self.epsilon = None

        if noise_multiplier < 0.0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")

        if self.epsilon is not None:
            name_str = f"dpsgd_eps{self.epsilon}_sigma{noise_multiplier:.6g}"
        else:
            name_str = f"dpsgd_C{max_grad_norm}_sigma{noise_multiplier:.6g}"

        super().__init__(name=name_str)
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

    @staticmethod
    def _compute_epsilon_static(
        noise_multiplier: float,
        num_steps: int,
        sample_rate: float,
        delta: float = 1e-5,
        alphas: Optional[list] = None,
    ) -> Optional[float]:
        """
        Estimate (epsilon, delta)-DP using the Rényi DP accountant.

        Uses the analytical Gaussian mechanism RDP bound.

        Args:
            noise_multiplier: Noise-to-clipping ratio sigma.
            num_steps: Total number of gradient update steps.
            sample_rate: Probability that each sample is included in a
                mini-batch (= batch_size / dataset_size).
            delta: Target delta for (epsilon, delta)-DP.
            alphas: RDP orders to evaluate.  Defaults to a standard range.

        Returns:
            Estimated epsilon, or ``None`` if noise_multiplier is zero.
        """
        if noise_multiplier == 0:
            return None

        if alphas is None:
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        def _rdp_gaussian(alpha: float) -> float:
            return alpha / (2 * noise_multiplier**2)

        def _rdp_subsample(alpha: float, rdp: float, q: float) -> float:
            if q == 0:
                return 0.0
            if q == 1.0:
                return rdp
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

    def compute_epsilon(
        self,
        num_steps: int,
        sample_rate: float,
        delta: float = 1e-5,
        alphas: Optional[list] = None,
    ) -> Optional[float]:
        """Instance convenience wrapper around :meth:`_compute_epsilon_static`."""
        return self._compute_epsilon_static(
            noise_multiplier=self.noise_multiplier,
            num_steps=num_steps,
            sample_rate=sample_rate,
            delta=delta,
            alphas=alphas,
        )

    @classmethod
    def calibrate_noise_multiplier(
        cls,
        target_epsilon: float,
        num_steps: int,
        sample_rate: float,
        delta: float = 1e-5,
        sigma_lo: float = 1e-4,
        sigma_hi: float = 1000.0,
        tol: float = 1e-3,
        max_iter: int = 100,
    ) -> float:
        """
        Binary-search for the noise_multiplier that achieves *target_epsilon*.

        Higher sigma → lower epsilon (more privacy). The search finds the
        smallest sigma whose epsilon is ≤ target_epsilon.

        Args:
            target_epsilon: Desired privacy budget epsilon.
            num_steps: Total gradient update steps.
            sample_rate: Subsampling probability (batch / dataset).
            delta: Target delta.
            sigma_lo: Lower bound for search.
            sigma_hi: Upper bound for search.
            tol: Convergence tolerance on sigma.
            max_iter: Maximum binary-search iterations.

        Returns:
            Calibrated noise_multiplier.

        Raises:
            ValueError: If the target epsilon cannot be achieved within
                the search bounds.
        """
        # Verify bounds are feasible
        eps_hi = cls._compute_epsilon_static(
            sigma_lo, num_steps, sample_rate, delta
        )
        eps_lo = cls._compute_epsilon_static(
            sigma_hi, num_steps, sample_rate, delta
        )

        if eps_lo is None or eps_hi is None:
            raise ValueError("Cannot calibrate with zero noise bounds")

        if target_epsilon > eps_hi:
            return sigma_lo
        if target_epsilon < eps_lo:
            raise ValueError(
                f"target_epsilon={target_epsilon} is too small; even "
                f"sigma={sigma_hi} only achieves eps={eps_lo:.4f}. "
                f"Increase sigma_hi or relax epsilon."
            )

        for _ in range(max_iter):
            sigma_mid = (sigma_lo + sigma_hi) / 2
            eps_mid = cls._compute_epsilon_static(
                sigma_mid, num_steps, sample_rate, delta
            )
            if eps_mid is None:
                sigma_hi = sigma_mid
                continue
            if eps_mid > target_epsilon:
                sigma_lo = sigma_mid
            else:
                sigma_hi = sigma_mid
            if sigma_hi - sigma_lo < tol:
                break

        return sigma_hi

    @property
    def privacy_budget(self) -> Optional[Dict[str, float]]:
        """Return (epsilon, delta) dict if privacy accounting is available."""
        if self.epsilon is not None:
            return {"epsilon": self.epsilon, "delta": self.delta}
        return None

    def summary(self) -> Dict[str, Any]:
        info = {
            **super().summary(),
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "noise_std": self.noise_multiplier * self.max_grad_norm,
        }
        if self.epsilon is not None:
            info["epsilon"] = self.epsilon
            info["delta"] = self.delta
        return info

"""
Gradient Pruning Defense.

Applies top-k magnitude sparsification to gradients: only the largest
``keep_ratio`` fraction of gradient entries (by absolute value) are
retained; the rest are zeroed out.

This is a common communication-efficient aggregation technique in
federated learning that also provides a privacy benefit by discarding
low-magnitude gradient components.

Usage::

    from defense.strategies.gradient_pruning import GradientPruning

    defense = GradientPruning(keep_ratio=0.1)
    defended = defense.apply(gradients)  # 90% of entries zeroed
"""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class GradientPruning(GradientDefense):
    """
    Top-k sparsification defense.

    Retains only the top ``keep_ratio`` fraction of gradient entries by
    absolute magnitude and zeros out the rest.

    Args:
        keep_ratio: Fraction of gradient entries to keep (0, 1].
            For example, 0.1 means keep the top 10% and zero out 90%.
    """

    def __init__(self, keep_ratio: float = 0.1):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        super().__init__(name=f"pruning_keep{keep_ratio}")
        self.keep_ratio = keep_ratio

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k pruning to gradient tensor.

        Args:
            gradients: (..., D) gradient tensor of any batch shape.

        Returns:
            Pruned gradient tensor with the same shape. Entries outside
            the top-k by magnitude are set to zero.
        """
        if self.keep_ratio >= 1.0:
            return gradients

        original_shape = gradients.shape
        flat = gradients.reshape(-1, original_shape[-1])

        k = max(1, int(flat.shape[-1] * self.keep_ratio))

        abs_vals = flat.abs()
        # Per-vector threshold: k-th largest value
        threshold = abs_vals.topk(k, dim=-1).values[:, -1:]

        mask = abs_vals >= threshold
        pruned = flat * mask.float()

        return pruned.reshape(original_shape)

    def summary(self) -> Dict[str, Any]:
        return {
            **super().summary(),
            "keep_ratio": self.keep_ratio,
            "zero_ratio": 1.0 - self.keep_ratio,
        }

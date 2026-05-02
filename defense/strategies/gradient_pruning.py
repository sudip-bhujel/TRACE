"""Top-k magnitude sparsification defense."""

from typing import Any, Dict

import torch

from defense.base import GradientDefense


class GradientPruning(GradientDefense):
    """Retain the top ``keep_ratio`` fraction of gradient entries by magnitude."""

    def __init__(self, keep_ratio: float = 0.1):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        super().__init__(name=f"pruning_keep{keep_ratio}")
        self.keep_ratio = keep_ratio

    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0:
            return gradients

        original_shape = gradients.shape
        flat = gradients.reshape(-1, original_shape[-1])

        k = max(1, int(flat.shape[-1] * self.keep_ratio))

        abs_vals = flat.abs()
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

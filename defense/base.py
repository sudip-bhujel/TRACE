"""
Base class for gradient defense mechanisms.

All defenses accept a gradient tensor and return a defended (transformed)
gradient tensor of the same shape.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class GradientDefense(ABC):
    """
    Abstract base class for gradient defense transformations.

    Subclasses must implement :meth:`apply` which takes a gradient tensor
    and returns the defended version.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Apply the defense to a batch of gradient sequences.

        Args:
            gradients: (B, T, D) or (T, D) or (D,) gradient tensor.

        Returns:
            Defended gradient tensor of the same shape.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def summary(self) -> Dict[str, Any]:
        """Return a dict describing this defense for logging."""
        return {"name": self.name, "type": self.__class__.__name__}


def get_defense(defense_type: str, **kwargs) -> GradientDefense:
    """
    Factory function for gradient defenses.

    Args:
        defense_type: One of ``"pruning"``, ``"noise"``, ``"dpsgd"``.
        **kwargs: Forwarded to the defense constructor.

    Returns:
        Configured :class:`GradientDefense` instance.

    Raises:
        ValueError: If *defense_type* is unknown.
    """
    from defense.strategies.dp_sgd import DPSGDDefense
    from defense.strategies.gradient_pruning import GradientPruning
    from defense.strategies.noise_injection import NoiseInjection

    registry = {
        "pruning": GradientPruning,
        "noise": NoiseInjection,
        "dpsgd": DPSGDDefense,
    }

    if defense_type not in registry:
        raise ValueError(
            f"Unknown defense type '{defense_type}'. Available: {list(registry.keys())}"
        )

    return registry[defense_type](**kwargs)

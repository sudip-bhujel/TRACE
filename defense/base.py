"""Base class and factory for gradient defense mechanisms."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class GradientDefense(ABC):
    """Abstract base class for gradient defense transformations."""

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def apply(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply the defense and return a tensor of the same shape."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__}


def get_defense(defense_type: str, **kwargs) -> GradientDefense:
    """Construct a defense by name. Supported: pruning, quantization, noise, dpsgd."""
    from defense.strategies.dp_sgd import DPSGDDefense
    from defense.strategies.gradient_pruning import GradientPruning
    from defense.strategies.gradient_quantization import GradientQuantization
    from defense.strategies.noise_injection import NoiseInjection

    registry = {
        "pruning": GradientPruning,
        "quantization": GradientQuantization,
        "noise": NoiseInjection,
        "dpsgd": DPSGDDefense,
    }

    if defense_type not in registry:
        raise ValueError(
            f"Unknown defense type '{defense_type}'. Available: {list(registry.keys())}"
        )

    return registry[defense_type](**kwargs)

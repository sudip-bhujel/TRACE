"""Gradient defense mechanisms applied before sharing gradients."""

from defense.base import GradientDefense, get_defense
from defense.strategies.dp_sgd import DPSGDDefense
from defense.strategies.gradient_pruning import GradientPruning
from defense.strategies.gradient_quantization import GradientQuantization
from defense.strategies.noise_injection import NoiseInjection

__all__ = [
    "GradientDefense",
    "GradientPruning",
    "GradientQuantization",
    "NoiseInjection",
    "DPSGDDefense",
    "get_defense",
]

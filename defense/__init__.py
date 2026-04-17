"""
Gradient Defense Mechanisms for Federated RL Privacy.

This package implements defenses that a server or client can apply to
shared gradients before transmission, reducing the information available
to a gradient inversion attacker.

Available defenses:
    - GradientPruning: Top-k sparsification by magnitude.
    - GradientQuantization: Uniform quantization to n-bit levels (QSGD-style).
    - NoiseInjection: Additive Gaussian noise.
    - DPSGDDefense: Differential-privacy calibrated clipping + noise.
"""

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

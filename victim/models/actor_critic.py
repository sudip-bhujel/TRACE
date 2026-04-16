from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn

Transition = namedtuple(
    "Transition", ["obs", "action", "logp", "reward", "done", "value"]
)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: List of rewards
        values: List of values (including bootstrap value at end)
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        advantages: List of advantages
        returns: List of returns
    """
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def conv_block(in_c: int, out_c: int, k: int = 3, s: int = 2, p: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.BatchNorm2d(out_c),
    )


class _ReshapeFlatten(nn.Module):
    """Flatten using reshape (not view) for MPS backward compatibility.

    ``nn.Flatten`` calls ``Tensor.view`` which requires contiguous memory.
    On the MPS backend, BatchNorm2d can produce non-contiguous gradient
    tensors for large batches, causing ``view`` to fail during backward.
    ``reshape`` handles this transparently by copying when needed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network with shared CNN encoder."""

    def __init__(
        self, in_channels: int = 3, num_actions: int = 5, hidden_size: int = 512
    ):
        super().__init__()

        # CNN encoder
        self.encoder = nn.Sequential(
            conv_block(in_channels, 32, k=8, s=4, p=2),
            conv_block(32, 64, k=4, s=2, p=1),
            conv_block(64, 64, k=3, s=1, p=1),
            conv_block(64, 64, k=3, s=2, p=1),  # ! For smaller model
            _ReshapeFlatten(),
        )

        # Compute conv output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.encoder(dummy)
            conv_dim = conv_out.shape[1]

        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.ReLU(),
        )

        # Policy and value heads
        self.policy = nn.Linear(hidden_size, num_actions)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) as uint8 or float

        Returns:
            logits: Action logits (B, num_actions)
            value: State value (B,)
        """
        x = x.float() / 255.0
        z = self.encoder(x)
        h = self.fc(z)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value


# A2C uses the same shared CNN encoder + policy head + value head architecture
# as ActorCritic.  The only difference is the training objective: policy
# gradient without importance-sampling ratio or clipping.
A2C = ActorCritic

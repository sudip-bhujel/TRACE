from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from victim.models.actor_critic import _ReshapeFlatten, conv_block


class ReplayBuffer:
    """Fixed-size experience replay buffer for off-policy RL algorithms.

    Stores observations as uint8 to minimise memory; converts to float32 on
    sample.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of a single observation, e.g. ``(3, 84, 84)``.
        device: Device used for sampled tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idxs], device=self.device),
            torch.tensor(self.rewards[idxs], device=self.device),
            torch.tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idxs], device=self.device),
        )

    def __len__(self) -> int:
        return self.size


class SAC(nn.Module):
    """Discrete Soft Actor-Critic.

    Shares a CNN backbone between the stochastic actor and twin Q-critics.
    Designed for environments with discrete action spaces.

    Architecture
    ------------
    encoder (CNN, shared) → fc (shared) → actor_head  (logits over actions)
                                        → q1_head     (Q-values per action)
                                        → q2_head     (Q-values per action)

    Training uses *separate* optimisers so that the actor optimiser only
    updates ``actor_head`` while the critic optimiser updates the full
    shared encoder plus both Q-heads.

    Reference
    ---------
    Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings".

    Args:
        in_channels: Input image channels.
        num_actions: Number of discrete actions.
        hidden_size: Width of the shared FC layer.
    """

    def __init__(
        self, in_channels: int = 3, num_actions: int = 5, hidden_size: int = 512
    ):
        super().__init__()
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            conv_block(in_channels, 32, k=8, s=4, p=2),
            conv_block(32, 64, k=4, s=2, p=1),
            conv_block(64, 64, k=3, s=1, p=1),
            conv_block(64, 64, k=3, s=2, p=1),
            _ReshapeFlatten(),
        )
        with torch.no_grad():
            conv_dim = self.encoder(torch.zeros(1, in_channels, 84, 84)).shape[1]

        self.fc = nn.Sequential(nn.Linear(conv_dim, hidden_size), nn.ReLU())

        # Actor head — outputs action logits
        self.actor_head = nn.Linear(hidden_size, num_actions)

        # Twin Q-heads — output Q(s, a) for all a simultaneously
        self.q1_head = nn.Linear(hidden_size, num_actions)
        self.q2_head = nn.Linear(hidden_size, num_actions)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x.float() / 255.0))

    def actor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, probs)`` each of shape ``(B, A)``."""
        logits = self.actor_head(self._encode(x))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, log_probs.exp()

    def critics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q1, Q2)`` each of shape ``(B, A)``."""
        h = self._encode(x)
        return self.q1_head(h), self.q2_head(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, Q1, Q2)``."""
        h = self._encode(x)
        log_probs = F.log_softmax(self.actor_head(h), dim=-1)
        return log_probs, self.q1_head(h), self.q2_head(h)

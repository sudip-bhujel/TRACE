from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from victim.models.actor_critic import _ReshapeFlatten, conv_block


class ReplayBuffer:
    """Fixed-size replay buffer storing observations as uint8."""

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
    """
    Discrete Soft Actor-Critic (Christodoulou, 2019).

    Shared CNN encoder feeds an actor head and twin Q-heads. The critic
    optimiser updates the encoder and Q-heads; the actor optimiser updates
    only the actor head.
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

        self.actor_head = nn.Linear(hidden_size, num_actions)
        self.q1_head = nn.Linear(hidden_size, num_actions)
        self.q2_head = nn.Linear(hidden_size, num_actions)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x.float() / 255.0))

    def actor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor_head(self._encode(x))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, log_probs.exp()

    def critics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        return self.q1_head(h), self.q2_head(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        log_probs = F.log_softmax(self.actor_head(h), dim=-1)
        return log_probs, self.q1_head(h), self.q2_head(h)

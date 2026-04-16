from typing import List, Tuple

import numpy as np

from victim.models.actor_critic import compute_gae


class PPOGradientBuffer:
    """Buffer episode data needed to compute PPO-style per-step gradients."""

    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clear()

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae_and_returns(
        self, next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns with explicit bootstrap value."""
        if len(self.dones) == 0:
            return [], []
        bootstrap = 0.0 if self.dones[-1] else next_value
        values_for_gae = self.values + [bootstrap]
        advantages, returns = compute_gae(
            self.rewards, values_for_gae, self.dones, gamma=self.gamma, lam=self.lam
        )
        return advantages, returns

    def clear(self) -> None:
        self.obs_list: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

    def __len__(self) -> int:
        return len(self.obs_list)

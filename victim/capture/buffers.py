from typing import List, Optional, Tuple

import numpy as np

from victim.models.actor_critic import compute_gae


class PPOGradientBuffer:
    """Stores per-step rollout data needed to compute GAE-based per-step gradients."""

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
        recurrent_state: Optional[np.ndarray] = None,
    ) -> None:
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.recurrent_states.append(recurrent_state)

    def compute_gae_and_returns(
        self, next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        if len(self.dones) == 0:
            return [], []
        bootstrap = 0.0 if self.dones[-1] else next_value
        values_for_gae = self.values + [bootstrap]
        return compute_gae(
            self.rewards, values_for_gae, self.dones, gamma=self.gamma, lam=self.lam
        )

    def clear(self) -> None:
        self.obs_list: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.recurrent_states: List[Optional[np.ndarray]] = []

    def __len__(self) -> int:
        return len(self.obs_list)

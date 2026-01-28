"""
PPO Agent for AI2-THOR Navigation Task

Based on standard PPO implementation with batched rollouts and minibatch updates.
"""

import os
import random
from collections import namedtuple
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ai2thor.controller import Controller

# ------------------------------
# Hyperparameters
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# AI2-THOR Navigation Environment
# ------------------------------
class AI2THORNavEnv:
    """
    Gym-like wrapper around AI2-THOR for point navigation tasks.

    Observations: RGB image (C, H, W) resized to (3, 84, 84)
    Actions: 0=MoveAhead, 1=RotateLeft(90), 2=RotateRight(90), 3=LookDown(30), 4=LookUp(30)
    Reward: -0.01 per step, +1.0 if agent reaches target position
    """

    ACTIONS = [
        {"action": "MoveAhead"},
        {"action": "RotateLeft", "degrees": 90},
        {"action": "RotateRight", "degrees": 90},
        {"action": "LookDown", "degrees": 30},
        {"action": "LookUp", "degrees": 30},
    ]

    def __init__(
        self,
        scene: str = "FloorPlan1",
        image_size: Tuple[int, int] = (84, 84),
        max_steps: int = 200,
        headless: bool = True,
        grid_size: float = 0.25,
    ):
        self.scene = scene
        self.image_size = image_size
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.step_count = 0

        self.controller = Controller(
            scene=scene,
            gridSize=grid_size,
            headless=headless,
        )

        # Try to initialize
        try:
            self.controller.step({"action": "Initialize", "gridSize": grid_size})
        except Exception:
            pass

        # Action and observation spaces
        self.action_space_n = len(self.ACTIONS)
        self.observation_shape = (3, image_size[0], image_size[1])

        self.last_event = None
        self.target_position = None

    def _get_observation(self) -> np.ndarray:
        """Get RGB observation as (C, H, W) uint8 array."""
        frame = self.last_event.frame  # (H, W, 3)
        img = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        img = np.transpose(img, (2, 0, 1)).astype(np.uint8)  # (C, H, W)
        return img

    def _get_agent_position(self) -> Dict[str, float]:
        """Get agent position."""
        return self.last_event.metadata["agent"]["position"]

    def _get_reachable_positions(self) -> List[Dict[str, float]]:
        """Get all reachable positions."""
        event = self.controller.step({"action": "GetReachablePositions"})
        return event.metadata.get("actionReturn", [])

    def _distance(self, p1: Dict, p2: Dict) -> float:
        """Euclidean distance between positions."""
        return np.sqrt(
            (p1["x"] - p2["x"]) ** 2
            + (p1["y"] - p2["y"]) ** 2
            + (p1["z"] - p2["z"]) ** 2
        )

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.step_count = 0
        self.controller.reset(self.scene)

        # Try to initialize
        try:
            self.controller.step({"action": "Initialize", "gridSize": self.grid_size})
        except Exception:
            pass

        # Get reachable positions
        reachable = self._get_reachable_positions()

        # Teleport to random position with random rotation
        if reachable:
            pos = random.choice(reachable)
            rot = random.choice([0, 90, 180, 270])
            try:
                self.controller.step(
                    {
                        "action": "TeleportFull",
                        "x": pos["x"],
                        "y": pos["y"],
                        "z": pos["z"],
                        "rotation": {"x": 0, "y": rot, "z": 0},
                        "horizon": 0.0,
                    }
                )
            except Exception:
                pass

        # Get first frame
        self.last_event = self.controller.step({"action": "Pass"})

        # Set random target position (different from agent)
        if reachable:
            agent_pos = self._get_agent_position()
            valid_targets = [p for p in reachable if self._distance(p, agent_pos) > 1.0]
            if valid_targets:
                self.target_position = random.choice(valid_targets)
            else:
                self.target_position = random.choice(reachable)
        else:
            self.target_position = self._get_agent_position()

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return (obs, reward, done, info)."""
        self.step_count += 1

        # Execute action
        try:
            self.last_event = self.controller.step(self.ACTIONS[action])
        except Exception as e:
            print(f"Controller error: {e}")
            self.controller.reset(self.scene)
            self.last_event = self.controller.step({"action": "Pass"})

        obs = self._get_observation()

        # Compute reward
        agent_pos = self._get_agent_position()
        dist = self._distance(agent_pos, self.target_position)

        reward = -0.01  # Step penalty
        done = False

        # Success if close to target
        if dist < 1.0:
            reward += 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        info = {"distance": dist}
        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        try:
            self.controller.stop()
        except Exception:
            pass


# ------------------------------
# Actor-Critic Network
# ------------------------------
def conv_block(in_c: int, out_c: int, k: int = 3, s: int = 2, p: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.BatchNorm2d(out_c),
    )


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
            nn.Flatten(),
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


# ------------------------------
# PPO Utilities
# ------------------------------
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


# ------------------------------
# Training Function
# ------------------------------
def train(
    env: AI2THORNavEnv,
    total_updates: int = 500,
    steps_per_update: int = 1024,
    mini_batch_size: int = 64,
    ppo_epochs: int = 4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 2.5e-4,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    save_dir: str = "checkpoints",
    resume_from: str = None,  # Path to checkpoint to resume from
) -> List[float]:
    """
    Train PPO agent.

    Args:
        env: AI2THOR environment
        total_updates: Number of policy updates
        steps_per_update: Steps to collect before each update
        mini_batch_size: Minibatch size for PPO updates
        ppo_epochs: Number of epochs per update
        gamma: Discount factor
        lam: GAE lambda
        clip_eps: PPO clip epsilon
        lr: Learning rate
        ent_coef: Entropy coefficient
        vf_coef: Value loss coefficient
        max_grad_norm: Max gradient norm for clipping
        save_dir: Directory to save checkpoints
        resume_from: Path to checkpoint file to resume training from

    Returns:
        List of episode rewards
    """
    num_actions = env.action_space_n

    model = ActorCritic(in_channels=3, num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    start_update = 1
    episode_rewards = []
    current_episode_reward = 0.0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_update = checkpoint.get("update", 0) + 1
        episode_rewards = checkpoint.get("episode_rewards", [])
        print(
            f"Resumed at update {start_update}, global_step {global_step}, {len(episode_rewards)} episodes"
        )

    for update in range(start_update, total_updates + 1):
        # Collect rollout
        obs_list, actions_list, logp_list = [], [], []
        rewards_list, dones_list, values_list = [], [], []

        obs = env.reset()

        for _ in range(steps_per_update):
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits, value = model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=device)).item()

            next_obs, reward, done, info = env.step(action)

            obs_list.append(obs)
            actions_list.append(action)
            logp_list.append(logp)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value.item())

            current_episode_reward += reward
            global_step += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = env.reset()
            else:
                obs = next_obs

        # Get last value for bootstrapping
        last_obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        _, last_val = model(last_obs_tensor)
        values_for_gae = values_list + [last_val.item()]

        # Compute GAE and returns
        advantages, returns = compute_gae(
            rewards_list, values_for_gae, dones_list, gamma=gamma, lam=lam
        )

        # Convert to tensors
        obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
        actions_batch = torch.tensor(actions_list, dtype=torch.long, device=device)
        old_logp_batch = torch.tensor(logp_list, dtype=torch.float32, device=device)
        returns_batch = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Normalize advantages
        adv_batch = (adv_batch - adv_batch.mean()) / (
            adv_batch.std(unbiased=False) + 1e-8
        )

        # PPO update with minibatches
        n_samples = obs_batch.size(0)
        inds = np.arange(n_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_samples, mini_batch_size):
                mb_inds = inds[start : start + mini_batch_size]
                mb_obs = obs_batch[mb_inds]
                mb_actions = actions_batch[mb_inds]
                mb_old_logp = old_logp_batch[mb_inds]
                mb_returns = returns_batch[mb_inds]
                mb_adv = adv_batch[mb_inds]

                logits, values = model(mb_obs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                mb_logp = dist.log_prob(mb_actions)

                # PPO clipped objective
                ratio = torch.exp(mb_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        if update % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            print(
                f"[Update {update:4d}] "
                f"Steps: {global_step:7d} | "
                f"Episodes: {len(episode_rewards):4d} | "
                f"Avg Reward: {avg_reward:7.3f} | "
                f"Entropy: {entropy.item():.4f}"
            )

        # Save checkpoint with full metadata
        if update % 100 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "episode_rewards": episode_rewards,
                "hyperparameters": {
                    "total_updates": total_updates,
                    "steps_per_update": steps_per_update,
                    "mini_batch_size": mini_batch_size,
                    "ppo_epochs": ppo_epochs,
                    "gamma": gamma,
                    "lam": lam,
                    "clip_eps": clip_eps,
                    "lr": lr,
                    "ent_coef": ent_coef,
                    "vf_coef": vf_coef,
                    "max_grad_norm": max_grad_norm,
                },
            }
            checkpoint_path = os.path.join(save_dir, f"ppo_ai2thor_{update}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final save with full metadata
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "global_step": global_step,
        "episode_rewards": episode_rewards,
        "hyperparameters": {
            "total_updates": total_updates,
            "steps_per_update": steps_per_update,
            "mini_batch_size": mini_batch_size,
            "ppo_epochs": ppo_epochs,
            "gamma": gamma,
            "lam": lam,
            "clip_eps": clip_eps,
            "lr": lr,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        },
    }
    torch.save(final_checkpoint, os.path.join(save_dir, "ppo_ai2thor_final.pt"))
    print("Training finished, model saved.")

    return episode_rewards


def plot_results(returns: List[float], save_path: str = "output/ppo_thor_results.png"):
    """Plot training results."""
    if not returns:
        print("No episodes completed, skipping plot.")
        return

    # Calculate running average
    running_avg = []
    for i in range(len(returns)):
        start_idx = max(0, i - 99)
        running_avg.append(np.mean(returns[start_idx : i + 1]))

    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Episode Reward", alpha=0.3, color="blue", lw=1)
    plt.plot(running_avg, label="Average Reward (100 episodes)", color="red", lw=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training on AI2-THOR Navigation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved as '{save_path}'")


# ------------------------------
# Main Entry
# ------------------------------
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("Initializing AI2-THOR environment...")
    env = AI2THORNavEnv(
        scene="FloorPlan1",
        image_size=(84, 84),
        max_steps=200,
        headless=False,
    )

    try:
        print("Training PPO agent...")
        episode_rewards = train(
            env,
            total_updates=500,
            steps_per_update=256,  # Reduced for faster iteration
            mini_batch_size=64,
            ppo_epochs=4,
            lr=2.5e-4,
            save_dir="checkpoints",
        )
        plot_results(episode_rewards)
    finally:
        env.close()
        print("\nTraining complete!")

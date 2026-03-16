import os
import random
import sys
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf

from victim.environment import AI2THORNavEnv
from victim.model import ActorCritic, compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    save_dir: str = "ckpts",
    resume_from: str = None,
    scenes: List[str] = None,  # List of scenes for interleaved training
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
        scenes: List of scenes for interleaved training (shuffles per episode)

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

    # Per-scene tracking

    scene_rewards = defaultdict(list)  # scene -> list of episode rewards
    current_scene = scenes[0] if scenes else env.scene

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

        # Reset with random scene if multi-scene training
        if scenes and len(scenes) > 1:
            current_scene = random.choice(scenes)
            obs = env.reset(scene=current_scene)
        else:
            current_scene = env.scene
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
                scene_rewards[current_scene].append(current_episode_reward)
                current_episode_reward = 0.0
                # Shuffle scene for next episode
                if scenes and len(scenes) > 1:
                    current_scene = random.choice(scenes)
                    obs = env.reset(scene=current_scene)
                else:
                    current_scene = env.scene
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

        # Per-scene logging every 50 updates
        if update % 50 == 0 and scenes and len(scenes) > 1:
            print("  Per-scene stats (last 20 episodes):")
            for scene in sorted(scene_rewards.keys()):
                recent = scene_rewards[scene][-20:]
                if recent:
                    scene_avg = np.mean(recent)
                    scene_count = len(scene_rewards[scene])
                    print(f"    {scene}: avg={scene_avg:6.3f}, total_eps={scene_count}")

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
    # plt.title("PPO Training on AI2-THOR Navigation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved as '{save_path}'")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m victim.train <config_path>"

    cfg = OmegaConf.load(sys.argv[1])

    # Extract settings from config
    env_cfg = cfg.get("environment", {})
    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    if "scenes" in env_cfg:
        scenes = list(env_cfg.get("scenes"))
    elif "scene" in env_cfg:
        scenes = [env_cfg.get("scene")]
    else:
        scenes = ["FloorPlan1"]

    # Set seed
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    save_dir = output_cfg.get("save_dir", "ckpts/victim")
    plot_path = output_cfg.get("plot_path", "output/victim_results.png")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    # Print config summary
    print("=" * 60)
    print("Victim PPO Training Configuration")
    print("=" * 60)
    print(f"Scenes: {scenes}")
    print(f"Total updates: {ppo_cfg.get('total_updates', 500)}")
    print(f"Steps per update: {ppo_cfg.get('steps_per_update', 1024)}")
    print(f"Headless: {env_cfg.get('headless', False)}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    # Create environment with first scene
    print(f"\nInitializing AI2-THOR environment (starting scene={scenes[0]})...")
    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=tuple(env_cfg.get("image_size", [84, 84])),
        max_steps=env_cfg.get("max_steps", 200),
        headless=env_cfg.get("headless", False),
    )

    try:
        episode_rewards = train(
            env,
            total_updates=ppo_cfg.get("total_updates", 500),
            steps_per_update=ppo_cfg.get("steps_per_update", 1024),
            mini_batch_size=ppo_cfg.get("mini_batch_size", 64),
            ppo_epochs=ppo_cfg.get("ppo_epochs", 4),
            gamma=ppo_cfg.get("gamma", 0.99),
            lam=ppo_cfg.get("gae_lambda", 0.95),
            clip_eps=ppo_cfg.get("clip_eps", 0.2),
            lr=ppo_cfg.get("learning_rate", 2.5e-4),
            ent_coef=ppo_cfg.get("entropy_coef", 0.01),
            vf_coef=ppo_cfg.get("value_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            save_dir=save_dir,
            resume_from=train_cfg.get("resume_from"),
            scenes=scenes,
        )
    finally:
        env.close()

    # Plot results
    if episode_rewards:
        plot_results(episode_rewards, plot_path)

    print("\nTraining complete!")

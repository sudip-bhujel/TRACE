import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import ActorCritic, compute_gae

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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

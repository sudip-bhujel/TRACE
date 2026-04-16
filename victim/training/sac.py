import copy
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from victim.environment import AI2THORNavEnv
from victim.models.sac import ReplayBuffer, SAC

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_sac(
    env: AI2THORNavEnv,
    total_steps: int = 500_000,
    batch_size: int = 256,
    buffer_size: int = 100_000,
    learning_starts: int = 5_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    target_entropy_ratio: float = 0.98,
    train_freq: int = 1,
    save_dir: str = "ckpts",
    resume_from: str = None,
    scenes: List[str] = None,
) -> List[float]:
    """
    Train Discrete SAC agent.

    Implements Christodoulou (2019): actor and twin critics share a CNN
    backbone.  The critic optimiser updates the shared encoder; the actor
    optimiser only updates the actor head.  Temperature α is tuned
    automatically to match ``target_entropy_ratio * log(A)``.

    Args:
        env: AI2THOR environment.
        total_steps: Total environment steps to collect.
        batch_size: Minibatch size for each gradient update.
        buffer_size: Replay buffer capacity.
        learning_starts: Steps of random exploration before training begins.
        gamma: Discount factor.
        tau: Soft target-network update coefficient.
        lr: Learning rate for all optimisers.
        target_entropy_ratio: Target entropy as a fraction of max entropy
            ``log(A)``.  0.98 ≈ near-uniform policy target.
        train_freq: Number of env steps between gradient updates.
        save_dir: Directory for checkpoints.
        resume_from: Optional checkpoint path to resume from.
        scenes: List of scenes for multi-scene training.

    Returns:
        List of episode rewards.
    """
    num_actions = env.action_space_n
    model = SAC(in_channels=3, num_actions=num_actions).to(device)
    target_model = copy.deepcopy(model)
    target_model.requires_grad_(False)

    # Critic optimiser: shared encoder + fc + both Q-heads
    critic_params = (
        list(model.encoder.parameters())
        + list(model.fc.parameters())
        + list(model.q1_head.parameters())
        + list(model.q2_head.parameters())
    )
    # Actor optimiser: only the actor head (encoder not updated by actor loss)
    actor_params = list(model.actor_head.parameters())

    critic_optimizer = optim.Adam(critic_params, lr=lr)
    actor_optimizer = optim.Adam(actor_params, lr=lr)

    # Automatic entropy tuning
    target_entropy = np.log(num_actions) * target_entropy_ratio
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=lr)

    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        obs_shape=env.observation_shape,
        device=device,
    )

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        target_model = copy.deepcopy(model)
        target_model.requires_grad_(False)
        critic_optimizer.load_state_dict(ckpt["optimizers"]["critic"])
        actor_optimizer.load_state_dict(ckpt["optimizers"]["actor"])
        alpha_optimizer.load_state_dict(ckpt["optimizers"]["alpha"])
        log_alpha.data.fill_(ckpt["log_alpha"])

    os.makedirs(save_dir, exist_ok=True)
    episode_rewards = []
    current_episode_reward = 0.0

    obs = env.reset(scene=scenes[0]) if (scenes and len(scenes) > 1) else env.reset()

    for step in range(1, total_steps + 1):
        alpha = log_alpha.exp().item()

        # --- collect transition ---
        if step < learning_starts:
            action = np.random.randint(num_actions)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                log_probs, probs = model.actor(obs_t)
            action = torch.distributions.Categorical(probs=probs).sample().item()

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        current_episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs = (
                env.reset(scene=random.choice(scenes))
                if (scenes and len(scenes) > 1)
                else env.reset()
            )

        # --- gradient updates ---
        if step >= learning_starts and len(replay_buffer) >= batch_size and step % train_freq == 0:
            obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(batch_size)

            with torch.no_grad():
                next_log_probs, next_probs = target_model.actor(next_obs_b)
                next_q1, next_q2 = target_model.critics(next_obs_b)
                # Discrete SAC soft Bellman backup
                next_v = (next_probs * (torch.min(next_q1, next_q2) - alpha * next_log_probs)).sum(dim=-1)
                target_q = rew_b + gamma * (1.0 - done_b) * next_v

            # Critic update
            q1, q2 = model.critics(obs_b)
            q1_a = q1.gather(1, act_b.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, act_b.unsqueeze(1)).squeeze(1)
            critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

            model.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update (encoder gradients flow but are not applied)
            log_probs, probs = model.actor(obs_b)
            with torch.no_grad():
                min_q = torch.min(*model.critics(obs_b))
            actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

            model.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Temperature update
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=-1).mean()
            alpha_loss = -log_alpha * (entropy - target_entropy)

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # Clamp log_alpha to prevent collapse to -∞
            with torch.no_grad():
                log_alpha.clamp_(min=-10.0)

            # Soft update target networks
            with torch.no_grad():
                for p, p_tgt in zip(model.parameters(), target_model.parameters()):
                    p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

        if step % 10_000 == 0:
            avg = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            print(
                f"[Step {step:8d}] Episodes: {len(episode_rewards):4d} | "
                f"Avg Reward: {avg:7.3f} | Alpha: {log_alpha.exp().item():.4f}"
            )

        if step % 100_000 == 0:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizers": {
                    "critic": critic_optimizer.state_dict(),
                    "actor": actor_optimizer.state_dict(),
                    "alpha": alpha_optimizer.state_dict(),
                },
                "log_alpha": log_alpha.item(),
                "step": step,
                "episode_rewards": episode_rewards,
            }
            path = os.path.join(save_dir, f"sac_ai2thor_{step}.pt")
            torch.save(ckpt, path)
            print(f"Checkpoint saved: {path}")

    torch.save(
        {"model_state_dict": model.state_dict(), "step": total_steps, "episode_rewards": episode_rewards},
        os.path.join(save_dir, "sac_ai2thor_final.pt"),
    )
    print("SAC training finished, model saved.")
    return episode_rewards

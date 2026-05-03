import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import A2C, compute_gae

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_a2c(
    env: AI2THORNavEnv,
    total_updates: int = 500,
    steps_per_update: int = 512,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 7e-4,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    save_dir: str = "ckpts",
    resume_from: str = None,
    scenes: List[str] = None,
) -> List[float]:
    """Train an A2C agent (REINFORCE + GAE, no clipping) and return per-episode rewards."""
    num_actions = env.action_space_n
    model = A2C(in_channels=3, num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    start_update = 1
    episode_rewards = []
    current_episode_reward = 0.0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("update", 0) + 1
        episode_rewards = ckpt.get("episode_rewards", [])

    if scenes and len(scenes) > 1:
        obs = env.reset(scene=scenes[0])
    else:
        obs = env.reset()

    for update in range(start_update, total_updates + 1):
        obs_list, act_list, logp_list = [], [], []
        rew_list, done_list, val_list = [], [], []

        for _ in range(steps_per_update):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=device)).item()

            next_obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            rew_list.append(reward)
            done_list.append(done)
            val_list.append(value.item())
            current_episode_reward += reward
            global_step += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = (
                    env.reset(scene=random.choice(scenes))
                    if (scenes and len(scenes) > 1)
                    else env.reset()
                )
            else:
                obs = next_obs

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_val = model(obs_t)

        advantages, returns = compute_gae(
            rew_list, val_list + [last_val.item()], done_list, gamma=gamma, lam=lam
        )

        obs_b = torch.tensor(
            np.ascontiguousarray(np.stack(obs_list)), dtype=torch.float32, device=device
        )
        act_b = torch.tensor(act_list, dtype=torch.long, device=device)
        adv_b = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_b = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

        logits, values = model(obs_b)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        log_probs = dist.log_prob(act_b)

        policy_loss = -(log_probs * adv_b).mean()
        value_loss = F.mse_loss(values, ret_b)
        entropy = dist.entropy().mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if update % 10 == 0:
            avg = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            print(
                f"[Update {update:4d}] Steps: {global_step:7d} | "
                f"Episodes: {len(episode_rewards):4d} | "
                f"Avg Reward: {avg:7.3f} | Entropy: {entropy.item():.4f}"
            )

        if update % 100 == 0:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "episode_rewards": episode_rewards,
            }
            path = os.path.join(save_dir, f"a2c_ai2thor_{update}.pt")
            torch.save(ckpt, path)
            print(f"Checkpoint saved: {path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "update": total_updates,
            "global_step": global_step,
            "episode_rewards": episode_rewards,
        },
        os.path.join(save_dir, "a2c_ai2thor_final.pt"),
    )
    print("A2C training finished, model saved.")
    return episode_rewards

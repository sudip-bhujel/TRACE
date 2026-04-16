from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.buffers import PPOGradientBuffer
from victim.capture.utils import _extract_flat_gradient, create_hdf5_dataset, flatten_gradients
from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import ActorCritic

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def compute_gradients(
    model: ActorCritic,
    observation: torch.Tensor,
    action: int,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute gradients for the simple probe loss."""
    model.zero_grad()

    logits, value = model(observation)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    log_prob = dist.log_prob(torch.tensor([action], device=device))
    policy_loss = -log_prob.mean()
    value_loss = value.mean()
    loss = policy_loss + 0.5 * value_loss

    loss.backward()

    gradients = {}
    dtype = np.float16 if use_float16 else np.float32
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if gradient_layers is not None and not any(
            layer in name for layer in gradient_layers
        ):
            continue
        gradients[name] = param.grad.detach().cpu().numpy().astype(dtype)
    return gradients


def compute_ppo_gradients(
    model: ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute gradients using the PPO-style loss used in capture mode."""
    model.zero_grad()

    logits, value = model(observation)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    new_logp = dist.log_prob(action)
    ratio = torch.exp(new_logp - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, returns)
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    loss.backward()

    gradients = {}
    dtype = np.float16 if use_float16 else np.float32
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if gradient_layers is not None and not any(
            layer in name for layer in gradient_layers
        ):
            continue
        gradients[name] = param.grad.detach().cpu().numpy().astype(dtype)
    return gradients


def _client_gradient_ppo(
    model: ActorCritic,
    obs: torch.Tensor,            # (T, C, H, W)
    actions: torch.Tensor,        # (T,)  long
    old_log_probs: torch.Tensor,  # (T,)
    advantages: torch.Tensor,     # (T,)
    returns: torch.Tensor,        # (T,)
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Exact PPO loss gradient over one local rollout — identical to train.py."""
    model.zero_grad()
    logits, values = model(obs)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()
    return _extract_flat_gradient(model, gradient_layers)


def _client_gradient_a2c(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Exact A2C loss gradient over one local rollout — identical to train_a2c().

    Same rollout data as PPO but uses the plain REINFORCE gradient weighted
    by GAE advantage, with no importance-sampling ratio or clipping.
    """
    model.zero_grad()
    logits, values = model(obs)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()
    return _extract_flat_gradient(model, gradient_layers)


def capture_ppo_gradients(
    model: ActorCritic,
    env: AI2THORNavEnv,
    save_path: str,
    num_trajectories: int = 100,
    max_steps: int = 200,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    scenes: Optional[List[str]] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> int:
    """
    Capture per-step gradients using a PPO-style objective.

    This is PPO-style per-step capture from a frozen checkpoint, not a client-level
    federated update. The ratio is near 1.0 on the original observations because the
    same checkpoint provides both old_log_prob and new_logp.
    """
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(0, dtype=torch.long, device=device)
    test_grads = compute_ppo_gradients(
        model=model,
        observation=obs_tensor,
        action=action_tensor,
        old_log_prob=torch.tensor(0.0, device=device),
        advantage=torch.tensor(1.0, device=device),
        returns=torch.tensor(1.0, device=device),
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gradient_layers=gradient_layers,
    )
    gradient_size = len(flatten_gradients(test_grads))
    estimated_steps = num_trajectories * (max_steps // 2)

    print("PPO Gradient Capture (per-step, frozen checkpoint)")
    print(f"  PPO params: clip_eps={clip_eps}, vf_coef={vf_coef}, ent_coef={ent_coef}")
    print(f"  GAE params: gamma={gamma}, lambda={lam}")
    print(
        f"Gradient size: {gradient_size:,} values ({gradient_size * 2 / 1024:.1f} KB per step)"
    )
    print(f"Estimated total steps: ~{estimated_steps:,}")

    create_hdf5_dataset(
        save_path,
        num_steps=estimated_steps,
        gradient_size=gradient_size,
        image_shape=obs.shape,
        compression=compression,
    )

    ppo_buffer = PPOGradientBuffer(gamma=gamma, lam=lam)
    step_idx = 0
    episode_rewards = []

    with h5py.File(save_path, "a") as f:
        for traj_idx in range(num_trajectories):
            obs = (
                env.reset(scene=scenes[traj_idx % len(scenes)])
                if scenes and len(scenes) > 1
                else env.reset()
            )
            done = False
            ep_steps = 0
            ep_reward = 0.0

            while not done and ep_steps < max_steps:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    logits, value = model(obs_tensor)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action, device=device)).item()

                next_obs, reward, done, info = env.step(action)
                ep_reward += reward
                ppo_buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value.item(),
                    log_prob=log_prob,
                )
                obs = next_obs
                ep_steps += 1

            if done:
                bootstrap_value = 0.0
            else:
                with torch.no_grad():
                    next_obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    _, next_val = model(next_obs_tensor)
                    bootstrap_value = next_val.item()

            advantages, returns = ppo_buffer.compute_gae_and_returns(
                next_value=bootstrap_value
            )
            if len(advantages) > 1:
                adv_array = np.array(advantages)
                adv_mean = adv_array.mean()
                adv_std = adv_array.std() + 1e-8
                advantages = [(a - adv_mean) / adv_std for a in advantages]

            for step_in_episode in range(len(ppo_buffer)):
                step_obs_tensor = torch.tensor(
                    ppo_buffer.obs_list[step_in_episode],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                step_action_tensor = torch.tensor(
                    ppo_buffer.actions[step_in_episode], dtype=torch.long, device=device
                )
                step_old_logp_tensor = torch.tensor(
                    ppo_buffer.log_probs[step_in_episode],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                step_advantage_tensor = torch.tensor(
                    advantages[step_in_episode], dtype=torch.float32, device=device
                ).unsqueeze(0)
                step_return_tensor = torch.tensor(
                    returns[step_in_episode], dtype=torch.float32, device=device
                ).unsqueeze(0)

                grads = compute_ppo_gradients(
                    model=model,
                    observation=step_obs_tensor,
                    action=step_action_tensor,
                    old_log_prob=step_old_logp_tensor,
                    advantage=step_advantage_tensor,
                    returns=step_return_tensor,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    gradient_layers=gradient_layers,
                )
                step_flat_grads = flatten_gradients(grads)

                if step_idx >= f["images"].shape[0]:
                    new_size = step_idx + estimated_steps
                    for key in [
                        "images",
                        "gradients",
                        "actions",
                        "rewards",
                        "episode_ids",
                        "done",
                    ]:
                        f[key].resize(new_size, axis=0)

                f["images"][step_idx] = ppo_buffer.obs_list[step_in_episode]
                f["gradients"][step_idx] = step_flat_grads
                f["actions"][step_idx] = ppo_buffer.actions[step_in_episode]
                f["rewards"][step_idx] = ppo_buffer.rewards[step_in_episode]
                f["episode_ids"][step_idx] = traj_idx
                f["done"][step_idx] = ppo_buffer.dones[step_in_episode]
                step_idx += 1

            ppo_buffer.clear()
            success = "✓" if info.get("distance", float("inf")) < 1.0 else "✗"
            episode_rewards.append(ep_reward)
            print(
                f"Trajectory {traj_idx + 1:4d}/{num_trajectories} | Steps: {ep_steps:3d} | Reward: {ep_reward:7.2f} | Success: {success} | Total: {step_idx:,} steps"
            )

        for key in ["images", "gradients", "actions", "rewards", "episode_ids", "done"]:
            f[key].resize(step_idx, axis=0)

        f.attrs["num_trajectories"] = num_trajectories
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["avg_reward"] = np.mean(episode_rewards)
        f.attrs["success_rate"] = sum(1 for r in episode_rewards if r > 0) / len(
            episode_rewards
        )
        f.attrs["loss_type"] = "ppo"
        f.attrs["ppo_clip_eps"] = clip_eps
        f.attrs["ppo_vf_coef"] = vf_coef
        f.attrs["ppo_ent_coef"] = ent_coef
        f.attrs["gae_gamma"] = gamma
        f.attrs["gae_lambda"] = lam
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    return step_idx

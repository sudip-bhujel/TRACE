from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.utils import (
    _extract_flat_gradient,
    create_hdf5_dataset,
    flatten_gradients,
)
from victim.environment import AI2THORNavEnv
from victim.models.sac import SAC

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def compute_sac_gradients(
    model: SAC,
    observation: torch.Tensor,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """SAC probe loss: actor entropy-regularised Q maximisation (alpha=1)."""
    model.zero_grad()
    log_probs, probs = model.actor(observation)
    q1, q2 = model.critics(observation)
    min_q = torch.min(q1, q2)
    loss = (probs * (log_probs - min_q)).sum(dim=-1).mean()
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


def _client_gradient_sac(
    model: SAC,
    target_model: SAC,
    obs_b: torch.Tensor,
    act_b: torch.Tensor,
    rew_b: torch.Tensor,
    next_obs_b: torch.Tensor,
    done_b: torch.Tensor,
    gamma: float = 0.99,
    alpha: float = 0.2,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Exact SAC loss gradient (critic + actor) over a local batch."""
    model.zero_grad()

    with torch.no_grad():
        next_log_probs, next_probs = target_model.actor(next_obs_b)
        next_q1, next_q2 = target_model.critics(next_obs_b)
        next_v = (
            next_probs * (torch.min(next_q1, next_q2) - alpha * next_log_probs)
        ).sum(-1)
        target_q = rew_b + gamma * (1.0 - done_b) * next_v

    q1, q2 = model.critics(obs_b)
    q1_a = q1.gather(1, act_b.unsqueeze(1)).squeeze(1)
    q2_a = q2.gather(1, act_b.unsqueeze(1)).squeeze(1)
    critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

    log_probs, probs = model.actor(obs_b)
    with torch.no_grad():
        min_q = torch.min(q1, q2)
    actor_loss = (probs * (alpha * log_probs - min_q)).sum(-1).mean()

    (critic_loss + actor_loss).backward()
    return _extract_flat_gradient(model, gradient_layers)


def compute_sac_exact_gradient(
    model: SAC,
    target_model: SAC,
    obs_t: torch.Tensor,
    action: int,
    reward: float,
    next_obs_t: torch.Tensor,
    done: bool,
    gamma: float = 0.99,
    alpha: float = 0.2,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """Exact SAC gradient (critic + actor) for a single transition."""
    model.zero_grad()

    act_b = torch.tensor([action], dtype=torch.long, device=obs_t.device)
    rew_b = torch.tensor([reward], dtype=torch.float32, device=obs_t.device)
    done_b = torch.tensor([float(done)], dtype=torch.float32, device=obs_t.device)

    with torch.no_grad():
        next_log_probs, next_probs = target_model.actor(next_obs_t)
        next_q1, next_q2 = target_model.critics(next_obs_t)
        next_v = (
            next_probs * (torch.min(next_q1, next_q2) - alpha * next_log_probs)
        ).sum(-1)
        target_q = rew_b + gamma * (1.0 - done_b) * next_v

    q1, q2 = model.critics(obs_t)
    q1_a = q1.gather(1, act_b.unsqueeze(1)).squeeze(1)
    q2_a = q2.gather(1, act_b.unsqueeze(1)).squeeze(1)
    critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

    log_probs, probs = model.actor(obs_t)
    with torch.no_grad():
        min_q = torch.min(q1, q2)
    actor_loss = (probs * (alpha * log_probs - min_q)).sum(-1).mean()

    (critic_loss + actor_loss).backward()

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


def capture_sac_exact_gradients(
    model: SAC,
    target_model: SAC,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    steps_per_scene: int = 1000,
    max_steps_per_episode: int = 100,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    gamma: float = 0.99,
    alpha: float = 0.2,
) -> int:
    """Capture per-step exact SAC gradients (critic + actor); also stores next_images."""
    obs = env.reset(scene=scenes[0])
    next_obs_probe = env.reset(scene=scenes[0])
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    next_obs_t = torch.tensor(
        next_obs_probe, dtype=torch.float32, device=device
    ).unsqueeze(0)
    test_grads = compute_sac_exact_gradient(
        model,
        target_model,
        obs_t,
        0,
        0.0,
        next_obs_t,
        False,
        gamma=gamma,
        alpha=alpha,
        gradient_layers=gradient_layers,
    )
    gradient_size = len(flatten_gradients(test_grads))
    total_steps = steps_per_scene * len(scenes)

    print("SAC Gradient Capture (exact loss)")
    print(f"  gamma={gamma}, alpha={alpha}")
    print(f"  {steps_per_scene} steps x {len(scenes)} scenes = {total_steps:,} total")
    print(
        f"  Gradient size: {gradient_size:,} ({gradient_size * 2 / 1024:.1f} KB/step)"
    )

    create_hdf5_dataset(
        save_path,
        num_steps=total_steps,
        gradient_size=gradient_size,
        image_shape=obs.shape,
        compression=compression,
        with_next_images=True,
    )

    step_idx = 0
    scene_stats = {s: {"steps": 0, "episodes": 0, "rewards": []} for s in scenes}

    with h5py.File(save_path, "a") as f:
        for scene_idx, scene in enumerate(scenes):
            scene_steps = 0
            scene_episodes = 0
            print(
                f"\n[Scene {scene_idx + 1}/{len(scenes)}] {scene}: "
                f"capturing {steps_per_scene} steps..."
            )

            while scene_steps < steps_per_scene:
                obs = env.reset(scene=scene)
                done = False
                ep_steps = 0
                ep_reward = 0.0

                while (
                    not done
                    and ep_steps < max_steps_per_episode
                    and scene_steps < steps_per_scene
                ):
                    obs_t = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        _, probs = model.actor(obs_t)
                        action = (
                            torch.distributions.Categorical(probs=probs).sample().item()
                        )

                    try:
                        next_obs, reward, done, info = env.step(action)
                    except Exception as e:
                        print(
                            f"Critical error in scene {scene}: {e}. Skipping episode."
                        )
                        break

                    ep_reward += reward

                    next_obs_t = torch.tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    grads = compute_sac_exact_gradient(
                        model=model,
                        target_model=target_model,
                        obs_t=obs_t,
                        action=action,
                        reward=reward,
                        next_obs_t=next_obs_t,
                        done=done,
                        gamma=gamma,
                        alpha=alpha,
                        gradient_layers=gradient_layers,
                    )
                    flat_grads = flatten_gradients(grads)

                    episode_id = scene_idx * 10_000 + scene_episodes
                    f["images"][step_idx] = obs
                    f["next_images"][step_idx] = next_obs
                    f["gradients"][step_idx] = flat_grads
                    f["actions"][step_idx] = action
                    f["rewards"][step_idx] = reward
                    f["episode_ids"][step_idx] = episode_id
                    f["done"][step_idx] = done

                    obs = next_obs
                    step_idx += 1
                    ep_steps += 1
                    scene_steps += 1

                scene_episodes += 1
                scene_stats[scene]["rewards"].append(ep_reward)

            scene_stats[scene]["steps"] = scene_steps
            scene_stats[scene]["episodes"] = scene_episodes
            avg_reward = (
                np.mean(scene_stats[scene]["rewards"])
                if scene_stats[scene]["rewards"]
                else 0.0
            )
            print(
                f"  -> {scene}: {scene_steps} steps, {scene_episodes} episodes, "
                f"avg_reward={avg_reward:.2f}"
            )

        keys = [
            "images",
            "next_images",
            "gradients",
            "actions",
            "rewards",
            "episode_ids",
            "done",
        ]
        for key in keys:
            if step_idx < f[key].shape[0]:
                f[key].resize(step_idx, axis=0)

        f.attrs["num_scenes"] = len(scenes)
        f.attrs["steps_per_scene"] = steps_per_scene
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["capture_mode"] = "sac_exact"
        f.attrs["loss_type"] = "sac_exact"
        f.attrs["sac_gamma"] = gamma
        f.attrs["sac_alpha"] = alpha
        f.attrs["has_next_images"] = True
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    print(f"\nSAC capture complete. Total steps: {step_idx:,}")
    return step_idx

from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.buffers import PPOGradientBuffer
from victim.capture.utils import (
    create_hdf5_dataset,
    flatten_gradients,
)
from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import ActorCritic

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def compute_a2c_gradients(
    model: ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute per-step gradients under the exact A2C loss."""
    model.zero_grad()

    logits, value = model(observation)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_prob = dist.log_prob(action)

    policy_loss = -(log_prob * advantage).mean()
    value_loss = F.mse_loss(value, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()

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


def capture_a2c_gradients(
    model: ActorCritic,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    steps_per_scene: int = 1000,
    max_steps_per_episode: int = 100,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    gamma: float = 0.99,
    lam: float = 0.95,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> int:
    """Capture per-step exact A2C gradients with episode-buffered GAE."""
    # Use train mode so BatchNorm uses per-batch statistics; weights stay frozen.
    model.train()

    obs = env.reset(scene=scenes[0])
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    adv_t = torch.tensor([1.0], dtype=torch.float32, device=device)
    ret_t = torch.tensor([1.0], dtype=torch.float32, device=device)
    act_t = torch.tensor(0, dtype=torch.long, device=device)
    test_grads = compute_a2c_gradients(
        model,
        obs_tensor,
        act_t,
        adv_t,
        ret_t,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gradient_layers=gradient_layers,
    )
    gradient_size = len(flatten_gradients(test_grads))
    total_steps = steps_per_scene * len(scenes)

    print("A2C Gradient Capture")
    print(f"  GAE: gamma={gamma}, lam={lam} | vf_coef={vf_coef}, ent_coef={ent_coef}")
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
                a2c_buffer = PPOGradientBuffer(gamma=gamma, lam=lam)

                while (
                    not done
                    and ep_steps < max_steps_per_episode
                    and scene_steps + ep_steps < steps_per_scene
                ):
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits, value = model(obs_tensor)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()
                        log_prob = dist.log_prob(
                            torch.tensor(action, device=device)
                        ).item()

                    try:
                        next_obs, reward, done, info = env.step(action)
                    except Exception as e:
                        print(
                            f"Critical error in scene {scene}: {e}. Skipping episode."
                        )
                        break

                    ep_reward += reward
                    a2c_buffer.add(
                        obs=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value.item(),
                        log_prob=log_prob,
                    )
                    obs = next_obs
                    ep_steps += 1

                if len(a2c_buffer) == 0:
                    continue

                if done:
                    bootstrap_value = 0.0
                else:
                    with torch.no_grad():
                        next_tensor = torch.tensor(
                            obs, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        _, next_val = model(next_tensor)
                        bootstrap_value = next_val.item()

                advantages, returns = a2c_buffer.compute_gae_and_returns(
                    next_value=bootstrap_value
                )
                if len(advantages) > 1:
                    adv_arr = np.array(advantages)
                    advantages = list(
                        (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
                    )

                episode_id = scene_idx * 10_000 + scene_episodes

                for step_in_ep in range(len(a2c_buffer)):
                    step_obs_tensor = torch.tensor(
                        a2c_buffer.obs_list[step_in_ep],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    step_action_tensor = torch.tensor(
                        a2c_buffer.actions[step_in_ep],
                        dtype=torch.long,
                        device=device,
                    )
                    step_adv_tensor = torch.tensor(
                        advantages[step_in_ep],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    step_ret_tensor = torch.tensor(
                        returns[step_in_ep],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    grads = compute_a2c_gradients(
                        model=model,
                        observation=step_obs_tensor,
                        action=step_action_tensor,
                        advantage=step_adv_tensor,
                        returns=step_ret_tensor,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        gradient_layers=gradient_layers,
                    )
                    flat_grads = flatten_gradients(grads)

                    f["images"][step_idx] = a2c_buffer.obs_list[step_in_ep]
                    f["gradients"][step_idx] = flat_grads
                    f["actions"][step_idx] = a2c_buffer.actions[step_in_ep]
                    f["rewards"][step_idx] = a2c_buffer.rewards[step_in_ep]
                    f["episode_ids"][step_idx] = episode_id
                    f["done"][step_idx] = a2c_buffer.dones[step_in_ep]
                    step_idx += 1

                scene_steps += len(a2c_buffer)
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

        actual_size = step_idx
        for key in ["images", "gradients", "actions", "rewards", "episode_ids", "done"]:
            if actual_size < f[key].shape[0]:
                f[key].resize(actual_size, axis=0)

        f.attrs["num_scenes"] = len(scenes)
        f.attrs["steps_per_scene"] = steps_per_scene
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["capture_mode"] = "a2c_exact"
        f.attrs["loss_type"] = "a2c"
        f.attrs["gae_gamma"] = gamma
        f.attrs["gae_lambda"] = lam
        f.attrs["vf_coef"] = vf_coef
        f.attrs["ent_coef"] = ent_coef
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    print(f"\nA2C capture complete. Total steps: {step_idx:,}")
    return step_idx

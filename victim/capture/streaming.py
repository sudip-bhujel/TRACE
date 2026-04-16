from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.ppo import compute_gradients
from victim.capture.sac import compute_sac_gradients
from victim.capture.utils import create_hdf5_dataset, flatten_gradients
from victim.environment import AI2THORNavEnv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _sample_action(
    model: torch.nn.Module, obs_tensor: torch.Tensor, algorithm: str
) -> int:
    """Sample an action from the model (handles all three algorithms)."""
    with torch.no_grad():
        if algorithm in ("ppo", "a2c"):
            logits, _ = model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
        else:  # sac
            _, probs = model.actor(obs_tensor)
    return torch.distributions.Categorical(probs=probs).sample().item()


def _compute_probe_gradients(
    model: torch.nn.Module,
    obs_tensor: torch.Tensor,
    action: int,
    algorithm: str,
    gradient_layers: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Dispatch to the correct probe-loss gradient function."""
    if algorithm == "sac":
        return compute_sac_gradients(model, obs_tensor, gradient_layers)
    else:  # ppo, a2c
        return compute_gradients(model, obs_tensor, action, gradient_layers)


def capture_and_save_streaming(
    model: torch.nn.Module,
    env: AI2THORNavEnv,
    save_path: str,
    num_trajectories: int = 100,
    max_steps: int = 200,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    scenes: Optional[List[str]] = None,
    algorithm: str = "ppo",
) -> int:
    """Capture trajectories using the probe loss for the given algorithm."""
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = _compute_probe_gradients(model, obs_tensor, 0, algorithm, gradient_layers)
    gradient_size = len(flatten_gradients(test_grads))
    estimated_steps = num_trajectories * (max_steps // 2)

    print(
        f"Gradient size: {gradient_size:,} values ({gradient_size * 2 / 1024:.1f} KB per step as float16)"
    )
    print(f"Estimated total steps: ~{estimated_steps:,}")
    if scenes and len(scenes) > 1:
        print(f"Shuffling across {len(scenes)} scenes: {scenes}")

    create_hdf5_dataset(
        save_path,
        num_steps=estimated_steps,
        gradient_size=gradient_size,
        image_shape=obs.shape,
        compression=compression,
    )

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
                action = _sample_action(model, obs_tensor, algorithm)
                gradients = _compute_probe_gradients(
                    model, obs_tensor.clone(), action, algorithm, gradient_layers
                )
                flat_grads = flatten_gradients(gradients)

                next_obs, reward, done, info = env.step(action)
                ep_reward += reward

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

                f["images"][step_idx] = obs
                f["gradients"][step_idx] = flat_grads
                f["actions"][step_idx] = action
                f["rewards"][step_idx] = reward
                f["episode_ids"][step_idx] = traj_idx
                f["done"][step_idx] = done

                obs = next_obs
                step_idx += 1
                ep_steps += 1

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
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    return step_idx


def capture_uniform_per_scene(
    model: torch.nn.Module,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    steps_per_scene: int = 500,
    max_steps_per_episode: int = 100,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    algorithm: str = "ppo",
) -> int:
    """Capture a fixed number of steps from each scene for uniform distribution."""
    obs = env.reset(scene=scenes[0])
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = _compute_probe_gradients(model, obs_tensor, 0, algorithm, gradient_layers)
    gradient_size = len(flatten_gradients(test_grads))
    total_steps = steps_per_scene * len(scenes)

    print(
        f"Uniform capture mode: {steps_per_scene} steps x {len(scenes)} scenes = {total_steps:,} total steps"
    )
    print(
        f"Gradient size: {gradient_size:,} values ({gradient_size * 2 / 1024:.1f} KB per step)"
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
                f"\n[Scene {scene_idx + 1}/{len(scenes)}] {scene}: capturing {steps_per_scene} steps..."
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
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action = _sample_action(model, obs_tensor, algorithm)
                    gradients = _compute_probe_gradients(
                        model, obs_tensor.clone(), action, algorithm, gradient_layers
                    )
                    flat_grads = flatten_gradients(gradients)

                    try:
                        next_obs, reward, done, info = env.step(action)
                    except Exception as e:
                        print(f"Critical error taking step in scene {scene}: {e}")
                        print("Skipping remaining steps for this scene.")
                        break

                    ep_reward += reward
                    f["images"][step_idx] = obs
                    f["gradients"][step_idx] = flat_grads
                    f["actions"][step_idx] = action
                    f["rewards"][step_idx] = reward
                    f["episode_ids"][step_idx] = scene_idx * 1000 + scene_episodes
                    f["done"][step_idx] = done

                    obs = next_obs
                    step_idx += 1
                    ep_steps += 1
                    scene_steps += 1

                scene_episodes += 1
                scene_stats[scene]["rewards"].append(ep_reward)

            scene_stats[scene]["steps"] = scene_steps
            scene_stats[scene]["episodes"] = scene_episodes
            avg_reward = np.mean(scene_stats[scene]["rewards"])
            print(
                f"  -> {scene}: {scene_steps} steps, {scene_episodes} episodes, avg_reward={avg_reward:.2f}"
            )

        for key in ["images", "gradients", "actions", "rewards", "episode_ids", "done"]:
            if step_idx < f[key].shape[0]:
                f[key].resize(step_idx, axis=0)

        f.attrs["num_scenes"] = len(scenes)
        f.attrs["steps_per_scene"] = steps_per_scene
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["capture_mode"] = "uniform_per_scene"
        for scene in scenes:
            f.attrs[f"scene_{scene}_steps"] = scene_stats[scene]["steps"]
            f.attrs[f"scene_{scene}_episodes"] = scene_stats[scene]["episodes"]
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    print(f"\n{'=' * 60}")
    print("Uniform Capture Complete")
    print(f"{'=' * 60}")
    print(f"Total steps: {step_idx:,}")
    print(f"Steps per scene: {steps_per_scene}")
    print("Scene distribution:")
    for scene in scenes:
        print(
            f"  {scene}: {scene_stats[scene]['steps']} steps, {scene_stats[scene]['episodes']} eps"
        )

    return step_idx

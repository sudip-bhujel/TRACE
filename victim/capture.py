"""
Gradient Capture Script for PPO Agent (Efficient Storage)

This script captures gradients from a trained PPO model and saves them
using HDF5 format with compression.

Two modes:
1. Simple loss: per-step gradient capture with simple policy + value loss
2. PPO loss: PPO-style per-step capture from a frozen checkpoint using
   episode-buffered GAE/returns

For each step in a trajectory, it saves:
- Current observation (image)
- Action taken
- Gradients of the loss w.r.t. model parameters
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from victim.environment import AI2THORNavEnv
from victim.model import ActorCritic, compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, num_actions: int = 5) -> ActorCritic:
    """Load trained PPO model from checkpoint."""
    model = ActorCritic(in_channels=3, num_actions=num_actions).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from update {checkpoint.get('update', '?')}")
        print(f"  Global steps: {checkpoint.get('global_step', '?')}")
        print(f"  Episodes: {len(checkpoint.get('episode_rewards', []))}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")

    model.eval()
    model.requires_grad_(True)
    return model


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


def flatten_gradients(gradients: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all gradients into a single 1D array."""
    flat_grads = []
    for name in sorted(gradients.keys()):
        flat_grads.append(gradients[name].flatten())
    return np.concatenate(flat_grads)


class PPOGradientBuffer:
    """Buffer episode data needed to compute PPO-style per-step gradients."""

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
    ) -> None:
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae_and_returns(
        self, next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns with explicit bootstrap value."""
        if len(self.dones) == 0:
            return [], []
        bootstrap = 0.0 if self.dones[-1] else next_value
        values_for_gae = self.values + [bootstrap]
        advantages, returns = compute_gae(
            self.rewards, values_for_gae, self.dones, gamma=self.gamma, lam=self.lam
        )
        return advantages, returns

    def clear(self) -> None:
        self.obs_list: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

    def __len__(self) -> int:
        return len(self.obs_list)


def create_hdf5_dataset(
    save_path: str,
    num_steps: int,
    gradient_size: int,
    image_shape: Tuple[int, int, int] = (3, 84, 84),
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Create HDF5 file with pre-allocated datasets."""
    with h5py.File(save_path, "w") as f:
        f.create_dataset(
            "images",
            shape=(num_steps, *image_shape),
            maxshape=(None, *image_shape),
            dtype=np.uint8,
            chunks=(1, *image_shape),
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset(
            "gradients",
            shape=(num_steps, gradient_size),
            maxshape=(None, gradient_size),
            dtype=np.float16,
            chunks=(1, gradient_size),
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset(
            "actions",
            shape=(num_steps,),
            maxshape=(None,),
            dtype=np.int8,
            compression=compression,
        )
        f.create_dataset(
            "rewards",
            shape=(num_steps,),
            maxshape=(None,),
            dtype=np.float32,
            compression=compression,
        )
        f.create_dataset(
            "episode_ids",
            shape=(num_steps,),
            maxshape=(None,),
            dtype=np.int32,
            compression=compression,
        )
        f.create_dataset(
            "done",
            shape=(num_steps,),
            maxshape=(None,),
            dtype=np.bool_,
            compression=compression,
        )
    print(f"Created HDF5 file: {save_path}")


def capture_and_save_streaming(
    model: ActorCritic,
    env: AI2THORNavEnv,
    save_path: str,
    num_trajectories: int = 100,
    max_steps: int = 200,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    scenes: Optional[List[str]] = None,
) -> int:
    """Capture trajectories using the simple probe loss."""
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = compute_gradients(model, obs_tensor, 0, gradient_layers)
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
                with torch.no_grad():
                    logits, _ = model(obs_tensor)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                gradients = compute_gradients(
                    model, obs_tensor.clone(), action, gradient_layers
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
    model: ActorCritic,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    steps_per_scene: int = 500,
    max_steps_per_episode: int = 100,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
) -> int:
    """Capture a fixed number of steps from each scene for uniform distribution."""
    obs = env.reset(scene=scenes[0])
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = compute_gradients(model, obs_tensor, 0, gradient_layers)
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
                    with torch.no_grad():
                        logits, _ = model(obs_tensor)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()

                    gradients = compute_gradients(
                        model, obs_tensor.clone(), action, gradient_layers
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


def print_file_info(save_path: str) -> None:
    """Print information about the saved HDF5 file."""
    with h5py.File(save_path, "r") as f:
        print("\n" + "=" * 60)
        print("HDF5 File Information")
        print("=" * 60)
        print(f"File: {save_path}")
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Size: {file_size_mb:.1f} MB")
        print("\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print("\nDatasets:")
        for key in f.keys():
            ds = f[key]
            size_mb = ds.nbytes / (1024 * 1024)
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, size={size_mb:.1f} MB")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python capture.py <config_path>"

    cfg = OmegaConf.load(sys.argv[1])
    env_cfg = cfg.get("environment", {})
    model_cfg = cfg.get("model", {})
    capture_cfg = cfg.get("capture", {})

    if "scenes" in env_cfg:
        scenes = list(env_cfg.get("scenes"))
    elif "scene" in env_cfg:
        scenes = [env_cfg.get("scene")]
    else:
        scenes = ["FloorPlan1"]

    print("=" * 60)
    print("Efficient Gradient Capture for PPO Agent")
    print("=" * 60)
    print(f"Scenes: {scenes}")
    print(f"Trajectories: {capture_cfg.get('num_trajectories')}")
    print(f"Checkpoint: {model_cfg.get('checkpoint')}")

    os.makedirs(os.path.dirname(capture_cfg.get("save_path")) or ".", exist_ok=True)

    print(f"\nLoading model from: {model_cfg.get('checkpoint')}")
    model = load_model(
        model_cfg.get("checkpoint"), num_actions=model_cfg.get("num_actions", 5)
    )

    print(f"\nInitializing AI2-THOR environment (starting scene={scenes[0]})...")
    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=(84, 84),
        max_steps=env_cfg.get("max_steps"),
        headless=env_cfg.get("headless"),
    )

    use_ppo_loss = capture_cfg.get("use_ppo_loss", False)
    steps_per_scene = capture_cfg.get("steps_per_scene", None)
    ppo_params = {
        "gamma": capture_cfg.get("gae_gamma", 0.99),
        "lam": capture_cfg.get("gae_lambda", 0.95),
        "clip_eps": capture_cfg.get("ppo_clip_eps", 0.2),
        "vf_coef": capture_cfg.get("ppo_vf_coef", 0.5),
        "ent_coef": capture_cfg.get("ppo_ent_coef", 0.01),
    }

    try:
        if use_ppo_loss:
            print("\n[PPO LOSS MODE] Capturing with PPO-style per-step objective...")
            print(
                f"  PPO params: clip_eps={ppo_params['clip_eps']}, vf_coef={ppo_params['vf_coef']}, ent_coef={ppo_params['ent_coef']}"
            )
            print(
                f"  GAE params: gamma={ppo_params['gamma']}, lambda={ppo_params['lam']}"
            )
            total_steps = capture_ppo_gradients(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                num_trajectories=capture_cfg.get("num_trajectories", 100),
                max_steps=env_cfg.get("max_steps", 200),
                gradient_layers=capture_cfg.get("gradient_layers"),
                scenes=scenes,
                **ppo_params,
            )
        elif steps_per_scene and len(scenes) > 1:
            print(f"\n[UNIFORM MODE] Capturing {steps_per_scene} steps per scene...")
            if capture_cfg.get("gradient_layers"):
                print(f"Capturing layers: {capture_cfg.get('gradient_layers')}")
            total_steps = capture_uniform_per_scene(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                scenes=scenes,
                steps_per_scene=steps_per_scene,
                max_steps_per_episode=env_cfg.get("max_steps"),
                gradient_layers=capture_cfg.get("gradient_layers"),
            )
        else:
            print(
                f"\nCapturing {capture_cfg.get('num_trajectories')} trajectories across {len(scenes)} scene(s)..."
            )
            if capture_cfg.get("gradient_layers"):
                print(f"Capturing layers: {capture_cfg.get('gradient_layers')}")
            total_steps = capture_and_save_streaming(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                num_trajectories=capture_cfg.get("num_trajectories", 100),
                max_steps=env_cfg.get("max_steps", 200),
                gradient_layers=capture_cfg.get("gradient_layers"),
                scenes=scenes,
            )

        print_file_info(capture_cfg.get("save_path"))
    finally:
        env.close()
        print("\nDone!")

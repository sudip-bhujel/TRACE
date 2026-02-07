"""
Gradient Capture Script for PPO Agent (Efficient Storage)

This script captures gradients from a trained PPO model and saves them
efficiently using HDF5 format with compression.

For each step in a trajectory, it saves:
- Current observation (image)
- Action taken
- Gradients of the loss w.r.t. model parameters
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from victim.env import AI2THORNavEnv
from victim.model import ActorCritic

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
    return model


def compute_gradients(
    model: ActorCritic,
    observation: torch.Tensor,
    action: int,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute gradients of the loss w.r.t. model parameters.

    Args:
        model: The actor-critic model
        observation: Input observation tensor (1, C, H, W)
        action: Action taken
        gradient_layers: List of layer names to capture (None = all)
        use_float16: Store gradients as float16 to save space

    Returns:
        Dictionary mapping parameter names to gradient arrays
    """
    model.zero_grad()

    # Forward pass
    logits, value = model(observation)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    # Compute loss
    log_prob = dist.log_prob(torch.tensor([action], device=device))
    policy_loss = -log_prob.mean()
    value_loss = value.mean()
    loss = policy_loss + 0.5 * value_loss

    # Backward pass
    loss.backward()

    # Collect gradients
    gradients = {}
    dtype = np.float16 if use_float16 else np.float32

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Filter by layer names if specified
            if gradient_layers is not None:
                if not any(layer in name for layer in gradient_layers):
                    continue
            gradients[name] = param.grad.detach().cpu().numpy().astype(dtype)

    return gradients


def flatten_gradients(gradients: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all gradients into a single 1D array."""
    flat_grads = []
    for name in sorted(gradients.keys()):
        flat_grads.append(gradients[name].flatten())
    return np.concatenate(flat_grads)


def create_hdf5_dataset(
    save_path: str,
    num_steps: int,
    gradient_size: int,
    image_shape: Tuple[int, int, int] = (3, 84, 84),
    compression: str = "gzip",
    compression_level: int = 4,
):
    """Create HDF5 file with pre-allocated datasets."""
    with h5py.File(save_path, "w") as f:
        # Images - uint8 for efficiency
        f.create_dataset(
            "images",
            shape=(num_steps, *image_shape),
            maxshape=(None, *image_shape),  # Allow unlimited resize
            dtype=np.uint8,
            chunks=(1, *image_shape),
            compression=compression,
            compression_opts=compression_level,
        )

        # Gradients - float16 for efficiency
        f.create_dataset(
            "gradients",
            shape=(num_steps, gradient_size),
            maxshape=(None, gradient_size),  # Allow unlimited resize
            dtype=np.float16,
            chunks=(1, gradient_size),
            compression=compression,
            compression_opts=compression_level,
        )

        # Actions - int8 is enough for 5 actions
        f.create_dataset(
            "actions",
            shape=(num_steps,),
            maxshape=(None,),  # Allow unlimited resize
            dtype=np.int8,
            compression=compression,
        )

        # Rewards
        f.create_dataset(
            "rewards",
            shape=(num_steps,),
            maxshape=(None,),  # Allow unlimited resize
            dtype=np.float32,
            compression=compression,
        )

        # Episode indices (to know which episode each step belongs to)
        f.create_dataset(
            "episode_ids",
            shape=(num_steps,),
            maxshape=(None,),  # Allow unlimited resize
            dtype=np.int32,
            compression=compression,
        )

        # Done flags
        f.create_dataset(
            "done",
            shape=(num_steps,),
            maxshape=(None,),  # Allow unlimited resize
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
):
    """
    Capture trajectories and save directly to HDF5 (streaming).

    This avoids keeping all data in memory.

    Args:
        scenes: List of scenes to shuffle between episodes. If None, uses env's current scene.
    """
    # First, do a test step to get gradient size
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = compute_gradients(model, obs_tensor, 0, gradient_layers)
    flat_grads = flatten_gradients(test_grads)
    gradient_size = len(flat_grads)

    # Estimate total steps
    estimated_steps = num_trajectories * (max_steps // 2)  # Conservative estimate

    print(
        f"Gradient size: {gradient_size:,} values ({gradient_size * 2 / 1024:.1f} KB per step as float16)"
    )
    print(f"Estimated total steps: ~{estimated_steps:,}")
    if scenes and len(scenes) > 1:
        print(f"Shuffling across {len(scenes)} scenes: {scenes}")

    # Create HDF5 file
    create_hdf5_dataset(
        save_path,
        num_steps=estimated_steps,
        gradient_size=gradient_size,
        image_shape=obs.shape,
        compression=compression,
    )

    # Capture and save
    step_idx = 0
    episode_rewards = []

    with h5py.File(save_path, "a") as f:
        for traj_idx in range(num_trajectories):
            # Round-robin scene selection for uniform distribution
            if scenes and len(scenes) > 1:
                scene = scenes[traj_idx % len(scenes)]
                obs = env.reset(scene=scene)
            else:
                obs = env.reset()
            done = False
            ep_steps = 0
            ep_reward = 0.0

            while not done and ep_steps < max_steps:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)

                # Get action
                with torch.no_grad():
                    logits, value = model(obs_tensor)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                # Compute gradients
                gradients = compute_gradients(
                    model, obs_tensor.clone(), action, gradient_layers
                )
                flat_grads = flatten_gradients(gradients)

                # Take step
                next_obs, reward, done, info = env.step(action)
                ep_reward += reward

                # Resize datasets if needed
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

                # Save to HDF5
                f["images"][step_idx] = obs
                f["gradients"][step_idx] = flat_grads
                f["actions"][step_idx] = action
                f["rewards"][step_idx] = reward
                f["episode_ids"][step_idx] = traj_idx
                f["done"][step_idx] = done

                obs = next_obs
                step_idx += 1
                ep_steps += 1

            # Episode complete
            success = "✓" if info.get("distance", float("inf")) < 1.0 else "✗"
            episode_rewards.append(ep_reward)
            print(
                f"Trajectory {traj_idx + 1:4d}/{num_trajectories} | "
                f"Steps: {ep_steps:3d} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Success: {success} | "
                f"Total: {step_idx:,} steps"
            )

        # Trim to actual size
        for key in ["images", "gradients", "actions", "rewards", "episode_ids", "done"]:
            f[key].resize(step_idx, axis=0)

        # Save metadata
        f.attrs["num_trajectories"] = num_trajectories
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["avg_reward"] = np.mean(episode_rewards)
        f.attrs["success_rate"] = sum(1 for r in episode_rewards if r > 0) / len(
            episode_rewards
        )

        # Save gradient layer names for reconstruction
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
):
    """
    Capture a fixed number of steps from each scene for uniform distribution.

    This ensures equal representation of all scenes regardless of episode length.

    Args:
        model: Trained PPO model
        env: AI2THOR environment
        save_path: Path to save HDF5 file
        scenes: List of scene names to capture from
        steps_per_scene: Target number of steps to capture from each scene
        max_steps_per_episode: Maximum steps per episode before reset
        gradient_layers: Specific layers to capture (None = all)
        compression: HDF5 compression type
    """
    # First, do a test step to get gradient size
    obs = env.reset(scene=scenes[0])
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    test_grads = compute_gradients(model, obs_tensor, 0, gradient_layers)
    flat_grads = flatten_gradients(test_grads)
    gradient_size = len(flat_grads)

    total_steps = steps_per_scene * len(scenes)

    print(
        f"Uniform capture mode: {steps_per_scene} steps × {len(scenes)} scenes = {total_steps:,} total steps"
    )
    print(
        f"Gradient size: {gradient_size:,} values ({gradient_size * 2 / 1024:.1f} KB per step)"
    )

    # Create HDF5 file
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
                # Reset to this specific scene
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

                    # Get action from model
                    with torch.no_grad():
                        logits, value = model(obs_tensor)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()

                    # Compute gradients
                    gradients = compute_gradients(
                        model, obs_tensor.clone(), action, gradient_layers
                    )
                    flat_grads = flatten_gradients(gradients)

                    # Take step
                    try:
                        next_obs, reward, done, info = env.step(action)
                    except Exception as e:
                        print(f"Critical error taking step in scene {scene}: {e}")
                        print("Skipping remaining steps for this scene.")
                        break

                    ep_reward += reward

                    # Save to HDF5
                    f["images"][step_idx] = obs
                    f["gradients"][step_idx] = flat_grads
                    f["actions"][step_idx] = action
                    f["rewards"][step_idx] = reward
                    f["episode_ids"][step_idx] = (
                        scene_idx * 1000 + scene_episodes
                    )  # Unique episode ID
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
                f"  → {scene}: {scene_steps} steps, {scene_episodes} episodes, avg_reward={avg_reward:.2f}"
            )

        # Trim to actual size (should be exact, but just in case)
        for key in ["images", "gradients", "actions", "rewards", "episode_ids", "done"]:
            if step_idx < f[key].shape[0]:
                f[key].resize(step_idx, axis=0)

        # Save metadata
        f.attrs["num_scenes"] = len(scenes)
        f.attrs["steps_per_scene"] = steps_per_scene
        f.attrs["total_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["capture_mode"] = "uniform_per_scene"

        # Per-scene distribution summary
        for scene in scenes:
            f.attrs[f"scene_{scene}_steps"] = scene_stats[scene]["steps"]
            f.attrs[f"scene_{scene}_episodes"] = scene_stats[scene]["episodes"]

        # Save gradient layer names
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [n.encode() for n in grad_names]
        f.attrs["gradient_shapes"] = [str(test_grads[n].shape) for n in grad_names]

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Uniform Capture Complete")
    print(f"{'=' * 60}")
    print(f"Total steps: {step_idx:,}")
    print(f"Steps per scene: {steps_per_scene}")
    print(f"Scene distribution:")
    for scene in scenes:
        print(
            f"  {scene}: {scene_stats[scene]['steps']} steps, {scene_stats[scene]['episodes']} eps"
        )

    return step_idx


def print_file_info(save_path: str):
    """Print information about the saved HDF5 file."""
    with h5py.File(save_path, "r") as f:
        print("\n" + "=" * 60)
        print("HDF5 File Information")
        print("=" * 60)

        print(f"File: {save_path}")
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Size: {file_size_mb:.1f} MB")

        print(f"\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print(f"\nDatasets:")
        for key in f.keys():
            ds = f[key]
            size_mb = ds.nbytes / (1024 * 1024)
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, size={size_mb:.1f} MB")


# ------------------------------
# Main Entry
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture gradients (efficient)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_ai2thor_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to capture",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="trajectory_data/gradients.h5",
        help="Path to save HDF5 file",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help="Specific layers to capture (e.g., 'fc policy value'). Default: all",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run AI2-THOR in headless mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan1",
        help="AI2-THOR scene name (or first scene if using config with multiple scenes)",
    )
    args = parser.parse_args()

    # Default values
    scenes = [args.scene]  # Default to single scene from args

    # Load config from YAML if provided
    if args.config:
        try:
            if Path(args.config).exists():
                cfg = OmegaConf.load(args.config)
                print(f"Loaded config from {args.config}")

                # Extract settings from config
                env_cfg = cfg.get("environment", {})
                model_cfg = cfg.get("model", {})
                capture_cfg = cfg.get("capture", {})

                # Apply config values - support both single scene and scenes list
                if "scenes" in env_cfg:
                    scenes = list(env_cfg.get("scenes"))
                elif "scene" in env_cfg:
                    scenes = [env_cfg.get("scene")]

                args.max_steps = env_cfg.get("max_steps", args.max_steps)
                args.headless = env_cfg.get("headless", args.headless)
                args.checkpoint = model_cfg.get("checkpoint", args.checkpoint)
                args.num_trajectories = capture_cfg.get(
                    "num_trajectories", args.num_trajectories
                )
                args.save_path = capture_cfg.get("save_path", args.save_path)
                args.layers = capture_cfg.get("gradient_layers", args.layers)
            else:
                print(f"Config file not found: {args.config}, using CLI args")
        except ImportError:
            print("OmegaConf not installed, using CLI args")

    print("=" * 60)
    print("Efficient Gradient Capture for PPO Agent")
    print("=" * 60)
    print(f"Scenes: {scenes}")
    print(f"Trajectories: {args.num_trajectories}")
    print(f"Checkpoint: {args.checkpoint}")

    # Create directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)

    # Create environment with first scene (will switch during capture)
    print(f"\nInitializing AI2-THOR environment (starting scene={scenes[0]})...")
    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=(84, 84),
        max_steps=args.max_steps,
        headless=args.headless,
    )

    # Check for uniform capture mode
    steps_per_scene = None
    if args.config:
        try:
            if Path(args.config).exists():
                cfg = OmegaConf.load(args.config)
                capture_cfg = cfg.get("capture", {})
                steps_per_scene = capture_cfg.get("steps_per_scene", None)
        except ImportError:
            pass

    try:
        if steps_per_scene and len(scenes) > 1:
            # Uniform capture mode
            print(f"\n[UNIFORM MODE] Capturing {steps_per_scene} steps per scene...")
            if args.layers:
                print(f"Capturing layers: {args.layers}")

            total_steps = capture_uniform_per_scene(
                model=model,
                env=env,
                save_path=args.save_path,
                scenes=scenes,
                steps_per_scene=steps_per_scene,
                max_steps_per_episode=args.max_steps,
                gradient_layers=args.layers,
            )
        else:
            # Standard trajectory-based capture
            print(
                f"\nCapturing {args.num_trajectories} trajectories across {len(scenes)} scene(s)..."
            )
            if args.layers:
                print(f"Capturing layers: {args.layers}")

            total_steps = capture_and_save_streaming(
                model=model,
                env=env,
                save_path=args.save_path,
                num_trajectories=args.num_trajectories,
                max_steps=args.max_steps,
                gradient_layers=args.layers,
                scenes=scenes,
            )

        # Print summary
        print_file_info(args.save_path)

    finally:
        env.close()
        print("\nDone!")

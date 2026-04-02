"""
Augment Gradient Dataset - Pre-compute Augmented Gradients

This script loads an existing gradient HDF5 file, applies image augmentations,
recomputes gradients through the PointNav model, and appends the augmented data
to create an expanded training dataset.

In PPO mode, augmentation uses the same episode-buffered GAE/return targets as
PPO-style capture, rather than a single-step TD approximation.
"""

import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from victim.model import ActorCritic, compute_gae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pointnav_model(checkpoint_path: str, num_actions: int = 5) -> ActorCritic:
    """Load trained PointNav model from checkpoint."""
    model = ActorCritic(in_channels=3, num_actions=num_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from update {checkpoint.get('update', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")
    model.eval()
    model.requires_grad_(True)
    return model


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> np.ndarray:
    """Apply simple color jitter augmentation to an image."""
    img = image.astype(np.float32) / 255.0
    brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
    img = np.clip(img * brightness_factor, 0, 1)

    contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean = img.mean(axis=(1, 2), keepdims=True)
    img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

    if img.shape[0] == 3:
        saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        img = np.clip(gray + (img - gray) * saturation_factor, 0, 1)

    if img.shape[0] == 3:
        hue_shift = np.random.uniform(-hue, hue)
        if abs(hue_shift) > 0.01:
            shift_amount = hue_shift * 0.5
            r, g, b = img[0], img[1], img[2]
            img = np.stack(
                [
                    np.clip(r + shift_amount * (g - b), 0, 1),
                    np.clip(g + shift_amount * (b - r), 0, 1),
                    np.clip(b + shift_amount * (r - g), 0, 1),
                ]
            )

    return (img * 255).astype(np.uint8)


def _flatten_model_gradients(model: ActorCritic) -> np.ndarray:
    flat_grads = []
    for name in sorted(dict(model.named_parameters()).keys()):
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            flat_grads.append(param.grad.detach().cpu().numpy().flatten())
    return np.concatenate(flat_grads).astype(np.float16)


def compute_gradients_from_image(
    model: ActorCritic,
    image: np.ndarray,
    action: int,
) -> np.ndarray:
    """Compute simple probe-loss gradients for a single image."""
    model.zero_grad()
    obs_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor([action], device=device)

    logits, value = model(obs_tensor)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_prob = dist.log_prob(action_tensor)
    policy_loss = -log_prob.mean()
    value_loss = value.mean()
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    return _flatten_model_gradients(model)


def compute_ppo_gradients_from_targets(
    model: ActorCritic,
    image: np.ndarray,
    action: int,
    old_log_prob: float,
    advantage: float,
    returns: float,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> np.ndarray:
    """Compute PPO-style gradients from precomputed episode-level targets."""
    model.zero_grad()

    obs_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.long, device=device)
    old_logp_tensor = torch.tensor(
        old_log_prob, dtype=torch.float32, device=device
    ).unsqueeze(0)
    advantage_tensor = torch.tensor(
        advantage, dtype=torch.float32, device=device
    ).unsqueeze(0)
    return_tensor = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(
        0
    )

    logits, value = model(obs_tensor)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_logp = dist.log_prob(action_tensor)

    ratio = torch.exp(new_logp - old_logp_tensor)
    surr1 = ratio * advantage_tensor
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_tensor
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, return_tensor)
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    loss.backward()
    return _flatten_model_gradients(model)


class EpisodeBuffer:
    """Buffer episode data for GAE computation during augmentation."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self) -> None:
        self.obs_list: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

    def add(self, obs, action, reward, done, value, log_prob) -> None:
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae_and_returns(
        self, next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns with explicit bootstrap."""
        bootstrap = 0.0 if self.dones[-1] else next_value
        values_for_gae = self.values + [bootstrap]
        advantages, returns = compute_gae(
            self.rewards, values_for_gae, self.dones, gamma=self.gamma, lam=self.lam
        )
        return advantages, returns

    def __len__(self) -> int:
        return len(self.obs_list)


def augment_hdf5_dataset(
    input_path: str,
    output_path: str,
    model: ActorCritic,
    num_augmentations: int = 2,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    seed: int = 42,
    batch_size: int = 100,
    use_ppo_loss: bool = False,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> None:
    """Augment an HDF5 dataset with recomputed gradients."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'=' * 60}")
    print("Gradient Dataset Augmentation (Memory-Efficient Streaming)")
    print(f"{'=' * 60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Augmentations per sample: {num_augmentations}")
    print(f"Batch size: {batch_size} (lower = less memory)")
    print(
        f"Augmentation params: brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}"
    )

    with h5py.File(input_path, "r") as f_in:
        num_original = len(f_in["images"])
        gradient_size = f_in["gradients"].shape[1]
        image_shape = f_in["images"].shape[1:]
        metadata = dict(f_in.attrs)

    num_augmented = num_original * num_augmentations
    total_samples = num_original + num_augmented
    mem_per_sample_mb = (np.prod(image_shape) + gradient_size * 2) / (1024 * 1024)
    mem_batch_mb = mem_per_sample_mb * batch_size

    print("\nOriginal dataset:")
    print(f"  Samples: {num_original:,}")
    print(f"  Gradient size: {gradient_size:,}")
    print(f"  Image shape: {image_shape}")

    print("\nAugmented dataset:")
    print(f"  Original samples: {num_original:,}")
    print(f"  Augmented samples: {num_augmented:,}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Expansion factor: {total_samples / num_original:.1f}x")
    print("\nMemory estimate:")
    print(f"  Per sample: {mem_per_sample_mb:.2f} MB")
    print(f"  Per batch ({batch_size} samples): {mem_batch_mb:.1f} MB")

    print("\nCreating output file...")
    with h5py.File(output_path, "w") as f_out:
        ds_images = f_out.create_dataset(
            "images",
            shape=(total_samples, *image_shape),
            dtype=np.uint8,
            chunks=(1, *image_shape),
            compression="gzip",
            compression_opts=4,
        )
        ds_gradients = f_out.create_dataset(
            "gradients",
            shape=(total_samples, gradient_size),
            dtype=np.float16,
            chunks=(1, gradient_size),
            compression="gzip",
            compression_opts=4,
        )
        ds_actions = f_out.create_dataset(
            "actions", shape=(total_samples,), dtype=np.int8, compression="gzip"
        )
        ds_rewards = f_out.create_dataset(
            "rewards", shape=(total_samples,), dtype=np.float32, compression="gzip"
        )
        ds_episode_ids = f_out.create_dataset(
            "episode_ids", shape=(total_samples,), dtype=np.int32, compression="gzip"
        )
        ds_done = f_out.create_dataset(
            "done", shape=(total_samples,), dtype=np.bool_, compression="gzip"
        )

        with h5py.File(input_path, "r") as f_in:
            print("Copying original data in batches...")
            for start_idx in tqdm(range(0, num_original, batch_size), desc="Original"):
                end_idx = min(start_idx + batch_size, num_original)
                ds_images[start_idx:end_idx] = f_in["images"][start_idx:end_idx]
                ds_gradients[start_idx:end_idx] = f_in["gradients"][start_idx:end_idx]
                ds_actions[start_idx:end_idx] = f_in["actions"][start_idx:end_idx]
                ds_rewards[start_idx:end_idx] = f_in["rewards"][start_idx:end_idx]
                ds_episode_ids[start_idx:end_idx] = f_in["episode_ids"][
                    start_idx:end_idx
                ]
                ds_done[start_idx:end_idx] = f_in["done"][start_idx:end_idx]

            print(f"\nGenerating {num_augmented:,} augmented samples in batches...")
            out_idx = num_original

            if use_ppo_loss:
                print("\n[PPO MODE] Collecting episode data for GAE computation...")
                episode_buffers: Dict[int, EpisodeBuffer] = {}

                for start_idx in tqdm(
                    range(0, num_original), desc="Collecting episodes"
                ):
                    sample_obs = f_in["images"][start_idx]
                    sample_action = f_in["actions"][start_idx]
                    sample_reward = f_in["rewards"][start_idx]
                    sample_done = f_in["done"][start_idx]
                    sample_ep_id = int(f_in["episode_ids"][start_idx])

                    if sample_ep_id not in episode_buffers:
                        episode_buffers[sample_ep_id] = EpisodeBuffer(
                            gamma=gamma, lam=lam
                        )

                    obs_tensor = torch.tensor(
                        sample_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits, value = model(obs_tensor)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action_tensor = torch.tensor(
                            sample_action, dtype=torch.long, device=device
                        )
                        log_prob = dist.log_prob(action_tensor).item()

                    episode_buffers[sample_ep_id].add(
                        obs=sample_obs,
                        action=int(sample_action),
                        reward=float(sample_reward),
                        done=bool(sample_done),
                        value=value.item(),
                        log_prob=log_prob,
                    )

                print(f"  Collected {len(episode_buffers)} episodes")

                for aug_num in range(num_augmentations):
                    print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
                    for ep_id, ep_buffer in tqdm(
                        episode_buffers.items(), desc="Augmenting episodes"
                    ):
                        # For datasets captured by capture.py, episodes are terminal at the stored boundary.
                        # If a future dataset contains truncated non-terminal segments, the correct bootstrap
                        # value would require the next observation, which is not stored in this HDF5 schema.
                        bootstrap_value = (
                            0.0 if ep_buffer.dones[-1] else ep_buffer.values[-1]
                        )
                        advantages, returns = ep_buffer.compute_gae_and_returns(
                            next_value=bootstrap_value
                        )

                        if len(advantages) > 1:
                            adv_mean = np.mean(advantages)
                            adv_std = np.std(advantages) + 1e-8
                            advantages = [(a - adv_mean) / adv_std for a in advantages]

                        for step_idx in range(len(ep_buffer)):
                            aug_image = apply_color_jitter(
                                ep_buffer.obs_list[step_idx],
                                brightness=brightness,
                                contrast=contrast,
                                saturation=saturation,
                                hue=hue,
                            )
                            aug_gradients = compute_ppo_gradients_from_targets(
                                model=model,
                                image=aug_image,
                                action=ep_buffer.actions[step_idx],
                                old_log_prob=ep_buffer.log_probs[step_idx],
                                advantage=advantages[step_idx],
                                returns=returns[step_idx],
                                clip_eps=clip_eps,
                                vf_coef=vf_coef,
                                ent_coef=ent_coef,
                            )

                            ds_images[out_idx] = aug_image
                            ds_gradients[out_idx] = aug_gradients
                            ds_actions[out_idx] = ep_buffer.actions[step_idx]
                            ds_rewards[out_idx] = ep_buffer.rewards[step_idx]
                            ds_episode_ids[out_idx] = ep_id + (aug_num + 1) * 100000
                            ds_done[out_idx] = ep_buffer.dones[step_idx]
                            out_idx += 1
            else:
                for aug_num in range(num_augmentations):
                    print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
                    for start_idx in tqdm(
                        range(0, num_original, batch_size), desc="Augmenting"
                    ):
                        end_idx = min(start_idx + batch_size, num_original)
                        batch_len = end_idx - start_idx
                        batch_images = f_in["images"][start_idx:end_idx]
                        batch_actions = f_in["actions"][start_idx:end_idx]
                        batch_rewards = f_in["rewards"][start_idx:end_idx]
                        batch_episode_ids = f_in["episode_ids"][start_idx:end_idx]
                        batch_done = f_in["done"][start_idx:end_idx]

                        for i in range(batch_len):
                            aug_image = apply_color_jitter(
                                batch_images[i],
                                brightness=brightness,
                                contrast=contrast,
                                saturation=saturation,
                                hue=hue,
                            )
                            aug_gradients = compute_gradients_from_image(
                                model, aug_image, int(batch_actions[i].item())
                            )
                            ds_images[out_idx] = aug_image
                            ds_gradients[out_idx] = aug_gradients
                            ds_actions[out_idx] = batch_actions[i]
                            ds_rewards[out_idx] = batch_rewards[i]
                            ds_episode_ids[out_idx] = (
                                batch_episode_ids[i] + (aug_num + 1) * 100000
                            )
                            ds_done[out_idx] = batch_done[i]
                            out_idx += 1

        for key, value in metadata.items():
            f_out.attrs[key] = value
        f_out.attrs["augmented"] = True
        f_out.attrs["num_augmentations"] = num_augmentations
        f_out.attrs["original_samples"] = num_original
        f_out.attrs["augmented_samples"] = num_augmented
        f_out.attrs["total_samples"] = total_samples
        f_out.attrs["augmentation_params"] = (
            f"brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}"
        )
        if use_ppo_loss:
            f_out.attrs["loss_type"] = "ppo"
            f_out.attrs["ppo_clip_eps"] = clip_eps
            f_out.attrs["ppo_vf_coef"] = vf_coef
            f_out.attrs["ppo_ent_coef"] = ent_coef
            f_out.attrs["gae_gamma"] = gamma
            f_out.attrs["gae_lambda"] = lam

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print("Augmentation Complete!")
    print(f"{'=' * 60}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Total samples: {total_samples:,}")
    print("Ready for training!")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python augment.py <config.yaml>"
    cfg = OmegaConf.load(sys.argv[1])
    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)

    print(f"Loading PointNav model from: {cfg.checkpoint}")
    model = load_pointnav_model(cfg.checkpoint)

    use_ppo_loss = cfg.get("use_ppo_loss", False)
    ppo_params = {}
    if use_ppo_loss:
        ppo_params = {
            "gamma": cfg.get("gae_gamma", 0.99),
            "lam": cfg.get("gae_lambda", 0.95),
            "clip_eps": cfg.get("ppo_clip_eps", 0.2),
            "vf_coef": cfg.get("ppo_vf_coef", 0.5),
            "ent_coef": cfg.get("ppo_ent_coef", 0.01),
        }
        print("\n[PPO LOSS MODE] Using PPO-style objective for gradient computation")
        print(
            f"  PPO params: clip_eps={ppo_params['clip_eps']}, vf_coef={ppo_params['vf_coef']}, ent_coef={ppo_params['ent_coef']}"
        )
        print(f"  GAE params: gamma={ppo_params['gamma']}, lambda={ppo_params['lam']}")

    augment_hdf5_dataset(
        input_path=cfg.input,
        output_path=cfg.output,
        model=model,
        num_augmentations=cfg.num_augmentations,
        brightness=cfg.brightness,
        contrast=cfg.contrast,
        saturation=cfg.saturation,
        hue=cfg.hue,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        use_ppo_loss=use_ppo_loss,
        **ppo_params,
    )

    print("\nDone!")

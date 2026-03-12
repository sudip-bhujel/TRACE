"""
Augment Gradient Dataset - Pre-compute Augmented Gradients

This script loads an existing gradient HDF5 file, applies image augmentations,
recomputes gradients through the PointNav model, and appends the augmented data
to create an expanded training dataset.

Usage:
    python -m victim.augment_gradients victim/config/augment.yaml
"""

import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from victim.model import ActorCritic

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

    model.eval()  # Freeze weights
    model.requires_grad_(True)  # But allow gradients for backward pass
    return model


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> np.ndarray:
    """
    Apply color jitter augmentation to image.

    Args:
        image: (3, H, W) uint8 array in [0, 255]
        brightness, contrast, saturation, hue: Jitter ranges

    Returns:
        Augmented image (3, H, W) uint8
    """
    # Convert to float [0, 1]
    img = image.astype(np.float32) / 255.0

    # Random brightness
    brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
    img = np.clip(img * brightness_factor, 0, 1)

    # Random contrast
    contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean = img.mean(axis=(1, 2), keepdims=True)
    img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

    # Random saturation (RGB only)
    if img.shape[0] == 3:
        saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
        # Convert to grayscale
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        img = np.clip(gray + (img - gray) * saturation_factor, 0, 1)

    # Random hue shift (simple channel rotation)
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

    # Convert back to uint8
    return (img * 255).astype(np.uint8)


def compute_gradients_from_image(
    model: ActorCritic,
    image: np.ndarray,  # (3, H, W) uint8
    action: int,
) -> np.ndarray:
    """
    Compute gradients by forward + backward through PointNav model.

    Args:
        model: PointNav ActorCritic model
        image: (3, H, W) uint8 image
        action: Action taken

    Returns:
        Flattened gradient vector as float16
    """
    model.zero_grad()

    # Convert image to tensor
    obs_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor([action], device=device)

    # Forward pass
    logits, value = model(obs_tensor)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    # Compute loss (same as capture_gradients.py)
    log_prob = dist.log_prob(action_tensor)
    policy_loss = -log_prob.mean()
    value_loss = value.mean()
    loss = policy_loss + 0.5 * value_loss

    # Backward pass
    loss.backward()

    # Flatten gradients (in sorted order for consistency)
    flat_grads = []
    for name in sorted(dict(model.named_parameters()).keys()):
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            flat_grads.append(param.grad.detach().cpu().numpy().flatten())

    gradient_vector = np.concatenate(flat_grads).astype(np.float16)
    return gradient_vector


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
    batch_size: int = 100,  # Process in batches to avoid OOM
):
    """
    Augment existing HDF5 dataset with recomputed gradients.

    Uses streaming to avoid loading entire dataset into memory.

    Args:
        input_path: Path to existing gradients HDF5 file
        output_path: Path to save augmented HDF5 file
        model: PointNav model for gradient computation
        num_augmentations: Number of augmented versions per sample
        brightness, contrast, saturation, hue: Augmentation parameters
        seed: Random seed for reproducibility
        batch_size: Number of samples to process at once (lower = less memory)
    """
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
        f"Augmentation params: brightness={brightness}, contrast={contrast}, "
        f"saturation={saturation}, hue={hue}"
    )

    # Get dataset metadata WITHOUT loading data into memory
    with h5py.File(input_path, "r") as f_in:
        num_original = len(f_in["images"])
        gradient_size = f_in["gradients"].shape[1]
        image_shape = f_in["images"].shape[1:]

        print("\nOriginal dataset:")
        print(f"  Samples: {num_original:,}")
        print(f"  Gradient size: {gradient_size:,}")
        print(f"  Image shape: {image_shape}")

        # Copy metadata
        metadata = dict(f_in.attrs)

    # Calculate total size
    num_augmented = num_original * num_augmentations
    total_samples = num_original + num_augmented

    print("\nAugmented dataset:")
    print(f"  Original samples: {num_original:,}")
    print(f"  Augmented samples: {num_augmented:,}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Expansion factor: {total_samples / num_original:.1f}x")

    # Estimate memory usage
    mem_per_sample_mb = (np.prod(image_shape) + gradient_size * 2) / (1024 * 1024)
    mem_batch_mb = mem_per_sample_mb * batch_size
    print("\nMemory estimate:")
    print(f"  Per sample: {mem_per_sample_mb:.2f} MB")
    print(f"  Per batch ({batch_size} samples): {mem_batch_mb:.1f} MB")

    # Create output HDF5 file
    print("\nCreating output file...")
    with h5py.File(output_path, "w") as f_out:
        # Create datasets
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
            "actions",
            shape=(total_samples,),
            dtype=np.int8,
            compression="gzip",
        )
        ds_rewards = f_out.create_dataset(
            "rewards",
            shape=(total_samples,),
            dtype=np.float32,
            compression="gzip",
        )
        ds_episode_ids = f_out.create_dataset(
            "episode_ids",
            shape=(total_samples,),
            dtype=np.int32,
            compression="gzip",
        )
        ds_done = f_out.create_dataset(
            "done",
            shape=(total_samples,),
            dtype=np.bool_,
            compression="gzip",
        )

        # Open input file for streaming
        with h5py.File(input_path, "r") as f_in:
            # Copy original data in batches
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

            # Generate augmented data in batches
            print(f"\nGenerating {num_augmented:,} augmented samples in batches...")
            out_idx = num_original

            for aug_num in range(num_augmentations):
                print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")

                for start_idx in tqdm(
                    range(0, num_original, batch_size), desc="Augmenting"
                ):
                    end_idx = min(start_idx + batch_size, num_original)
                    batch_len = end_idx - start_idx

                    # Load batch from input
                    batch_images = f_in["images"][start_idx:end_idx]
                    batch_actions = f_in["actions"][start_idx:end_idx]
                    batch_rewards = f_in["rewards"][start_idx:end_idx]
                    batch_episode_ids = f_in["episode_ids"][start_idx:end_idx]
                    batch_done = f_in["done"][start_idx:end_idx]

                    # Process each sample in batch
                    for i in range(batch_len):
                        # Apply augmentation
                        aug_image = apply_color_jitter(
                            batch_images[i],
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                            hue=hue,
                        )

                        # Recompute gradients
                        aug_gradients = compute_gradients_from_image(
                            model, aug_image, batch_actions[i].item()
                        )

                        # Save augmented sample
                        ds_images[out_idx] = aug_image
                        ds_gradients[out_idx] = aug_gradients
                        ds_actions[out_idx] = batch_actions[i]
                        ds_rewards[out_idx] = batch_rewards[i]
                        # Mark as augmented with unique episode ID
                        ds_episode_ids[out_idx] = (
                            batch_episode_ids[i] + (aug_num + 1) * 100000
                        )
                        ds_done[out_idx] = batch_done[i]

                        out_idx += 1

        # Save metadata
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

    # Print summary
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

    # Create output directory
    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)

    # Load PointNav model
    print(f"Loading PointNav model from: {cfg.checkpoint}")
    model = load_pointnav_model(cfg.checkpoint)

    # Augment dataset
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
    )

    print("\nDone!")

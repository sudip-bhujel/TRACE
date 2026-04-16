import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

from victim.models.actor_critic import ActorCritic
from victim.models.sac import SAC

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_model(
    checkpoint_path: str,
    num_actions: int = 5,
    algorithm: str = "ppo",
) -> torch.nn.Module:
    """Load trained model from checkpoint.  Supports ppo, a2c, sac."""
    algorithm = algorithm.lower()
    if algorithm == "sac":
        model = SAC(in_channels=3, num_actions=num_actions).to(device)
    else:  # ppo, a2c — both use ActorCritic
        model = ActorCritic(in_channels=3, num_actions=num_actions).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {algorithm.upper()} checkpoint (step={checkpoint.get('step') or checkpoint.get('update', '?')})")
        print(f"  Episodes: {len(checkpoint.get('episode_rewards', []))}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")

    model.eval()
    model.requires_grad_(True)
    return model


def flatten_gradients(gradients: dict) -> np.ndarray:
    """Flatten all gradients into a single 1D array."""
    flat_grads = []
    for name in sorted(gradients.keys()):
        flat_grads.append(gradients[name].flatten())
    return np.concatenate(flat_grads)


def _extract_flat_gradient(
    model: torch.nn.Module,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Flatten all (filtered) gradients from a model into a float32 array."""
    parts = []
    for name, param in sorted(model.named_parameters()):
        if param.grad is None:
            continue
        if gradient_layers is not None and not any(layer in name for layer in gradient_layers):
            continue
        parts.append(param.grad.detach().cpu().float().numpy().flatten())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def create_hdf5_dataset(
    save_path: str,
    num_steps: int,
    gradient_size: int,
    image_shape: Tuple[int, int, int] = (3, 84, 84),
    compression: str = "gzip",
    compression_level: int = 4,
    with_next_images: bool = False,
) -> None:
    """Create HDF5 file with pre-allocated datasets.

    Args:
        save_path: Path to write the HDF5 file.
        num_steps: Initial (pre-allocated) number of rows.
        gradient_size: Flattened gradient length.
        image_shape: Shape of each observation image (C, H, W).
        compression: HDF5 compression codec.
        compression_level: Compression level (0–9 for gzip).
        with_next_images: When True, also create a ``next_images`` dataset
            (needed for SAC exact-loss capture/augmentation).
    """
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
        if with_next_images:
            f.create_dataset(
                "next_images",
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

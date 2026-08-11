import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

from victim.models.actor_critic import build_actor_critic

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_model(
    checkpoint_path: str,
    num_actions: Optional[int] = None,
    algorithm: str = "ppo",
    architecture: Optional[str] = None,
    device_name: str = "auto",
) -> torch.nn.Module:
    """Load a trained victim checkpoint. Supports ppo, a2c."""
    algorithm = algorithm.lower()
    if device_name != "auto":
        model_device = torch.device(device_name)
    else:
        model_device = device
    checkpoint = torch.load(checkpoint_path, map_location=model_device)
    checkpoint_model_cfg = (
        checkpoint.get("model_config", {})
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else {}
    )

    saved_architecture = checkpoint_model_cfg.get("architecture")
    saved_num_actions = checkpoint_model_cfg.get("num_actions")
    resolved_architecture = architecture or saved_architecture or "cnn"
    resolved_num_actions = num_actions or saved_num_actions or 5

    if (
        architecture is not None
        and saved_architecture is not None
        and architecture.lower() != saved_architecture.lower()
    ):
        raise ValueError(
            f"Config requests architecture '{architecture}', but checkpoint "
            f"contains '{saved_architecture}'"
        )
    if (
        num_actions is not None
        and saved_num_actions is not None
        and num_actions != saved_num_actions
    ):
        raise ValueError(
            f"Config requests {num_actions} actions, but checkpoint contains "
            f"{saved_num_actions}"
        )

    model = build_actor_critic(
        architecture=resolved_architecture,
        in_channels=3,
        num_actions=resolved_num_actions,
    ).to(model_device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded {algorithm.upper()} checkpoint (step={checkpoint.get('step') or checkpoint.get('update', '?')})"
        )
        print(f"  Episodes: {len(checkpoint.get('episode_rewards', []))}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")

    model.eval()
    model.requires_grad_(True)
    print(f"  Device: {model_device}")
    return model


def flatten_gradients(gradients: dict) -> np.ndarray:
    flat_grads = []
    for name in sorted(gradients.keys()):
        flat_grads.append(gradients[name].flatten())
    return np.concatenate(flat_grads)


def _extract_flat_gradient(
    model: torch.nn.Module,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    parts = []
    for name, param in sorted(model.named_parameters()):
        if param.grad is None:
            continue
        if gradient_layers is not None and not any(
            layer in name for layer in gradient_layers
        ):
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
    with_ppo_targets: bool = False,
    goal_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    """Pre-allocate HDF5 datasets for gradient capture."""
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
        if goal_shape is not None:
            f.create_dataset(
                "goals",
                shape=(num_steps, *goal_shape),
                maxshape=(None, *goal_shape),
                dtype=np.float32,
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
        if with_ppo_targets:
            for key in ("old_log_probs", "advantages", "returns"):
                f.create_dataset(
                    key,
                    shape=(num_steps,),
                    maxshape=(None,),
                    dtype=np.float32,
                    compression=compression,
                )
    print(f"Created HDF5 file: {save_path}")


def print_file_info(save_path: str) -> None:
    with h5py.File(save_path, "r") as f:
        print(f"\nHDF5: {save_path}")
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Size: {file_size_mb:.1f} MB")
        print("Metadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print("Datasets:")
        for key in f.keys():
            ds = f[key]
            size_mb = ds.nbytes / (1024 * 1024)
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, size={size_mb:.1f} MB")

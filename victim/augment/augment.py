"""
Augment Gradient Dataset — Pre-compute Augmented Gradients

Loads an existing gradient HDF5 file, applies colour-jitter augmentations,
recomputes gradients through the victim model using the exact training-loss
objective for each algorithm, and writes the expanded dataset.

Supported algorithms and gradient modes
----------------------------------------
ppo  — exact PPO loss (episode-buffered GAE advantages + clipped surrogate)
a2c  — exact A2C loss (episode-buffered GAE advantages, no clipping)
sac  — SAC probe loss  (actor entropy-regularised Q maximisation, per-step)

Usage
-----
    python -m victim.augment.augment victim/config/ppo/augment.yaml
"""

import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import copy

from victim.augment.a2c import augment_a2c
from victim.augment.ppo import augment_ppo
from victim.augment.sac import augment_sac, augment_sac_exact
from victim.capture.utils import load_model

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def augment_hdf5_dataset(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    algorithm: str,
    num_augmentations: int = 2,
    batch_size: int = 100,
    seed: int = 42,
    jitter_kwargs: dict = None,
    loss_kwargs: dict = None,
    target_model: torch.nn.Module = None,
) -> None:
    """Augment an HDF5 gradient dataset for the given RL algorithm.

    Shared I/O (open files, create output datasets, copy originals, write
    metadata) is handled here.  The per-algorithm augmentation loop is
    delegated to ``augment_ppo``, ``augment_a2c``, or ``augment_sac`` /
    ``augment_sac_exact``.

    For SAC, the exact training loss is used automatically when the input
    HDF5 contains a ``next_images`` dataset (written by
    ``capture_sac_exact_gradients``).  In that case ``target_model`` must be
    provided; it defaults to a deepcopy of ``model`` when omitted.

    Args:
        input_path: Path to the source HDF5 file.
        output_path: Path for the augmented output HDF5 file.
        model: Loaded, frozen victim model.
        algorithm: ``"ppo"``, ``"a2c"``, or ``"sac"``.
        num_augmentations: Number of colour-jitter copies per original step.
        batch_size: Samples per HDF5 read batch (controls memory usage).
        seed: RNG seed for reproducibility.
        jitter_kwargs: Colour-jitter parameters passed to apply_color_jitter.
        loss_kwargs: Algorithm-specific loss hyperparameters.
        target_model: Target network for SAC exact loss (defaults to a
            deepcopy of ``model`` when the input has ``next_images``).
    """
    if jitter_kwargs is None:
        jitter_kwargs = {}
    if loss_kwargs is None:
        loss_kwargs = {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    with h5py.File(input_path, "r") as f_in:
        num_original = len(f_in["images"])
        gradient_size = f_in["gradients"].shape[1]
        image_shape = f_in["images"].shape[1:]
        metadata = dict(f_in.attrs)
        has_next_images = "next_images" in f_in

    num_augmented = num_original * num_augmentations
    total_samples = num_original + num_augmented
    mem_per_sample_mb = (np.prod(image_shape) + gradient_size * 2) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Gradient Dataset Augmentation — {algorithm.upper()}")
    print(f"{'=' * 60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Augmentations per sample: {num_augmentations}")
    print(f"Batch size: {batch_size} (lower = less memory)")
    print(f"Jitter: {jitter_kwargs}")
    print(f"\nOriginal dataset:")
    print(f"  Samples:       {num_original:,}")
    print(f"  Gradient size: {gradient_size:,}")
    print(f"  Image shape:   {image_shape}")
    print(f"\nAugmented dataset:")
    print(f"  Augmented samples: {num_augmented:,}")
    print(f"  Total samples:     {total_samples:,}")
    print(f"  Expansion factor:  {total_samples / num_original:.1f}x")
    print(f"  Per-sample memory: {mem_per_sample_mb:.2f} MB")

    # SAC exact mode requires next_images and a target network
    sac_exact = algorithm == "sac" and has_next_images
    if sac_exact and target_model is None:
        target_model = copy.deepcopy(model)
        target_model.eval()
        target_model.requires_grad_(False)

    if sac_exact:
        print("SAC mode: exact loss (next_images detected)")
    elif algorithm == "sac":
        print("SAC mode: probe loss (no next_images in input)")

    print("\nCreating output file...")
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset(
            "images",
            shape=(total_samples, *image_shape),
            dtype=np.uint8,
            chunks=(1, *image_shape),
            compression="gzip",
            compression_opts=4,
        )
        if has_next_images:
            f_out.create_dataset(
                "next_images",
                shape=(total_samples, *image_shape),
                dtype=np.uint8,
                chunks=(1, *image_shape),
                compression="gzip",
                compression_opts=4,
            )
        f_out.create_dataset(
            "gradients",
            shape=(total_samples, gradient_size),
            dtype=np.float16,
            chunks=(1, gradient_size),
            compression="gzip",
            compression_opts=4,
        )
        f_out.create_dataset(
            "actions", shape=(total_samples,), dtype=np.int8, compression="gzip"
        )
        f_out.create_dataset(
            "rewards", shape=(total_samples,), dtype=np.float32, compression="gzip"
        )
        f_out.create_dataset(
            "episode_ids", shape=(total_samples,), dtype=np.int32, compression="gzip"
        )
        f_out.create_dataset(
            "done", shape=(total_samples,), dtype=np.bool_, compression="gzip"
        )

        with h5py.File(input_path, "r") as f_in:
            # Copy originals
            print("\nCopying original data in batches...")
            for start_idx in tqdm(range(0, num_original, batch_size), desc="Original"):
                end_idx = min(start_idx + batch_size, num_original)
                f_out["images"][start_idx:end_idx] = f_in["images"][start_idx:end_idx]
                f_out["gradients"][start_idx:end_idx] = f_in["gradients"][start_idx:end_idx]
                f_out["actions"][start_idx:end_idx] = f_in["actions"][start_idx:end_idx]
                f_out["rewards"][start_idx:end_idx] = f_in["rewards"][start_idx:end_idx]
                f_out["episode_ids"][start_idx:end_idx] = f_in["episode_ids"][start_idx:end_idx]
                f_out["done"][start_idx:end_idx] = f_in["done"][start_idx:end_idx]
                if has_next_images:
                    f_out["next_images"][start_idx:end_idx] = f_in["next_images"][start_idx:end_idx]

            print(f"\nGenerating {num_augmented:,} augmented samples...")

            # Dispatch to algorithm-specific augmentation
            if algorithm == "ppo":
                out_idx = augment_ppo(
                    f_in, f_out, model,
                    out_idx=num_original,
                    num_augmentations=num_augmentations,
                    jitter_kwargs=jitter_kwargs,
                    **loss_kwargs,
                )
            elif algorithm == "a2c":
                out_idx = augment_a2c(
                    f_in, f_out, model,
                    out_idx=num_original,
                    num_augmentations=num_augmentations,
                    jitter_kwargs=jitter_kwargs,
                    **loss_kwargs,
                )
            elif algorithm == "sac" and sac_exact:
                out_idx = augment_sac_exact(
                    f_in, f_out, model, target_model,
                    out_idx=num_original,
                    num_augmentations=num_augmentations,
                    jitter_kwargs=jitter_kwargs,
                    batch_size=batch_size,
                    **loss_kwargs,
                )
            elif algorithm == "sac":
                out_idx = augment_sac(
                    f_in, f_out, model,
                    out_idx=num_original,
                    num_augmentations=num_augmentations,
                    jitter_kwargs=jitter_kwargs,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: ppo, a2c, sac")

        # Write metadata
        for key, value in metadata.items():
            f_out.attrs[key] = value
        f_out.attrs["algorithm"] = algorithm
        f_out.attrs["augmented"] = True
        f_out.attrs["num_augmentations"] = num_augmentations
        f_out.attrs["original_samples"] = num_original
        f_out.attrs["augmented_samples"] = num_augmented
        f_out.attrs["total_samples"] = out_idx
        f_out.attrs["augmentation_params"] = str(jitter_kwargs)
        if algorithm in ("ppo", "a2c") or (algorithm == "sac" and sac_exact):
            f_out.attrs["loss_type"] = f"{algorithm}_exact"
            for k, v in loss_kwargs.items():
                f_out.attrs[k] = v
        elif algorithm == "sac":
            f_out.attrs["loss_type"] = "sac_probe"

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print("Augmentation Complete!")
    print(f"{'=' * 60}")
    print(f"Output file: {output_path}")
    print(f"File size:   {file_size_mb:.1f} MB")
    print(f"Total samples: {out_idx:,}")
    print("Ready for training!")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m victim.augment.augment <config.yaml>"
    cfg = OmegaConf.load(sys.argv[1])

    algorithm = cfg.get("algorithm", "ppo").lower()
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})
    loss_cfg = cfg.get("loss", {})

    os.makedirs(os.path.dirname(data_cfg.get("output")) or ".", exist_ok=True)

    print(f"Loading {algorithm.upper()} model from: {model_cfg.get('checkpoint')}")
    model = load_model(
        model_cfg.get("checkpoint"),
        num_actions=model_cfg.get("num_actions", 5),
        algorithm=algorithm,
    )

    jitter_kwargs = {
        "brightness": aug_cfg.get("brightness", 0.2),
        "contrast": aug_cfg.get("contrast", 0.2),
        "saturation": aug_cfg.get("saturation", 0.2),
        "hue": aug_cfg.get("hue", 0.1),
    }

    # Build loss_kwargs per algorithm
    if algorithm == "ppo":
        loss_kwargs = {
            "gamma":    loss_cfg.get("gamma", 0.99),
            "lam":      loss_cfg.get("lam", 0.95),
            "clip_eps": loss_cfg.get("clip_eps", 0.2),
            "vf_coef":  loss_cfg.get("vf_coef", 0.5),
            "ent_coef": loss_cfg.get("ent_coef", 0.01),
        }
        print(
            f"[PPO EXACT LOSS] clip_eps={loss_kwargs['clip_eps']}, "
            f"vf_coef={loss_kwargs['vf_coef']}, ent_coef={loss_kwargs['ent_coef']}, "
            f"gamma={loss_kwargs['gamma']}, lam={loss_kwargs['lam']}"
        )
    elif algorithm == "a2c":
        loss_kwargs = {
            "gamma":    loss_cfg.get("gamma", 0.99),
            "lam":      loss_cfg.get("lam", 0.95),
            "vf_coef":  loss_cfg.get("vf_coef", 0.5),
            "ent_coef": loss_cfg.get("ent_coef", 0.01),
        }
        print(
            f"[A2C EXACT LOSS] vf_coef={loss_kwargs['vf_coef']}, "
            f"ent_coef={loss_kwargs['ent_coef']}, "
            f"gamma={loss_kwargs['gamma']}, lam={loss_kwargs['lam']}"
        )
    else:  # sac
        loss_kwargs = {
            "gamma": loss_cfg.get("gamma", 0.99),
            "alpha": loss_cfg.get("alpha", 0.2),
        }
        print(
            f"[SAC EXACT LOSS] gamma={loss_kwargs['gamma']}, alpha={loss_kwargs['alpha']} "
            f"(exact only when input contains next_images; probe otherwise)"
        )

    augment_hdf5_dataset(
        input_path=data_cfg.get("input"),
        output_path=data_cfg.get("output"),
        model=model,
        algorithm=algorithm,
        num_augmentations=aug_cfg.get("num_augmentations", 2),
        batch_size=data_cfg.get("batch_size", 100),
        seed=data_cfg.get("seed", 42),
        jitter_kwargs=jitter_kwargs,
        loss_kwargs=loss_kwargs,
    )

    print("\nDone!")

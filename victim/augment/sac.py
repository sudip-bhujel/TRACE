import h5py
import numpy as np
import torch
from tqdm import tqdm

from victim.augment.image import apply_color_jitter
from victim.capture.sac import compute_sac_exact_gradient, compute_sac_gradients
from victim.capture.utils import flatten_gradients
from victim.models.sac import SAC

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def augment_sac(
    f_in: h5py.File,
    f_out: h5py.File,
    model: SAC,
    out_idx: int,
    num_augmentations: int,
    jitter_kwargs: dict,
    batch_size: int = 100,
) -> int:
    """Augment a SAC dataset using the SAC probe loss.

    SAC augmentation always uses the probe loss (actor entropy-regularised Q
    maximisation) rather than an exact training loss.  The SAC exact loss
    requires ``next_obs`` and a target network, neither of which is available
    in the per-step HDF5 schema.

    Colour-jitter is applied per step and the gradient is recomputed from the
    augmented observation only — no episode context is needed.

    Args:
        f_in: Open input HDF5 file.
        f_out: Open output HDF5 file (datasets already created and originals copied).
        model: Frozen SAC checkpoint.
        out_idx: Write cursor — index of the first augmented slot in f_out.
        num_augmentations: Number of colour-jitter copies per original step.
        jitter_kwargs: kwargs forwarded to apply_color_jitter.
        batch_size: Number of samples to load from HDF5 at once.

    Returns:
        Updated out_idx after all augmented samples are written.
    """
    num_original = f_in["images"].shape[0]

    for aug_num in range(num_augmentations):
        print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
        for start_idx in tqdm(range(0, num_original, batch_size), desc="Augmenting"):
            end_idx = min(start_idx + batch_size, num_original)
            batch_len = end_idx - start_idx

            batch_images = f_in["images"][start_idx:end_idx]
            batch_actions = f_in["actions"][start_idx:end_idx]
            batch_rewards = f_in["rewards"][start_idx:end_idx]
            batch_episode_ids = f_in["episode_ids"][start_idx:end_idx]
            batch_done = f_in["done"][start_idx:end_idx]

            for i in range(batch_len):
                aug_image = apply_color_jitter(batch_images[i], **jitter_kwargs)
                obs_tensor = torch.tensor(
                    aug_image, dtype=torch.float32, device=device
                ).unsqueeze(0)
                grad_dict = compute_sac_gradients(model, obs_tensor)
                flat_grads = flatten_gradients(grad_dict)

                f_out["images"][out_idx] = aug_image
                f_out["gradients"][out_idx] = flat_grads.astype(np.float16)
                f_out["actions"][out_idx] = batch_actions[i]
                f_out["rewards"][out_idx] = batch_rewards[i]
                f_out["episode_ids"][out_idx] = batch_episode_ids[i] + (aug_num + 1) * 100_000
                f_out["done"][out_idx] = batch_done[i]
                out_idx += 1

    return out_idx


def augment_sac_exact(
    f_in: h5py.File,
    f_out: h5py.File,
    model: SAC,
    target_model: SAC,
    out_idx: int,
    num_augmentations: int,
    jitter_kwargs: dict,
    gamma: float = 0.99,
    alpha: float = 0.2,
    batch_size: int = 100,
) -> int:
    """Augment a SAC dataset using the exact SAC training loss.

    Requires ``next_images`` in the input HDF5 (written by
    ``capture_sac_exact_gradients``).  Colour-jitter is applied to both
    ``images`` and ``next_images`` and the exact critic + actor gradient is
    recomputed for each augmented transition.

    Args:
        f_in: Open input HDF5 file (must contain ``next_images``).
        f_out: Open output HDF5 file (datasets already created and originals copied).
        model: Frozen SAC checkpoint (online network).
        target_model: Target SAC network (typically same weights; no_grad).
        out_idx: Write cursor — index of the first augmented slot in f_out.
        num_augmentations: Number of colour-jitter copies per original step.
        jitter_kwargs: kwargs forwarded to apply_color_jitter.
        gamma: SAC discount factor.
        alpha: SAC entropy regularisation coefficient.
        batch_size: Number of samples to load from HDF5 at once.

    Returns:
        Updated out_idx after all augmented samples are written.
    """
    num_original = f_in["images"].shape[0]

    for aug_num in range(num_augmentations):
        print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
        for start_idx in tqdm(range(0, num_original, batch_size), desc="Augmenting"):
            end_idx = min(start_idx + batch_size, num_original)
            batch_len = end_idx - start_idx

            batch_images = f_in["images"][start_idx:end_idx]
            batch_next_images = f_in["next_images"][start_idx:end_idx]
            batch_actions = f_in["actions"][start_idx:end_idx]
            batch_rewards = f_in["rewards"][start_idx:end_idx]
            batch_done = f_in["done"][start_idx:end_idx]
            batch_episode_ids = f_in["episode_ids"][start_idx:end_idx]

            for i in range(batch_len):
                aug_image = apply_color_jitter(batch_images[i], **jitter_kwargs)
                aug_next_image = apply_color_jitter(batch_next_images[i], **jitter_kwargs)

                obs_t = torch.tensor(
                    aug_image, dtype=torch.float32, device=device
                ).unsqueeze(0)
                next_obs_t = torch.tensor(
                    aug_next_image, dtype=torch.float32, device=device
                ).unsqueeze(0)

                grad_dict = compute_sac_exact_gradient(
                    model=model,
                    target_model=target_model,
                    obs_t=obs_t,
                    action=int(batch_actions[i]),
                    reward=float(batch_rewards[i]),
                    next_obs_t=next_obs_t,
                    done=bool(batch_done[i]),
                    gamma=gamma,
                    alpha=alpha,
                )
                flat_grads = flatten_gradients(grad_dict)

                f_out["images"][out_idx] = aug_image
                f_out["next_images"][out_idx] = aug_next_image
                f_out["gradients"][out_idx] = flat_grads.astype(np.float16)
                f_out["actions"][out_idx] = batch_actions[i]
                f_out["rewards"][out_idx] = batch_rewards[i]
                f_out["episode_ids"][out_idx] = batch_episode_ids[i] + (aug_num + 1) * 100_000
                f_out["done"][out_idx] = batch_done[i]
                out_idx += 1

    return out_idx

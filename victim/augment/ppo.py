from typing import Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from victim.augment.image import apply_color_jitter
from victim.capture.buffers import PPOGradientBuffer
from victim.capture.utils import _extract_flat_gradient
from victim.models.actor_critic import ActorCritic

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _compute_ppo_gradient_from_targets(
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
    """Compute exact PPO-style gradient from precomputed episode-level targets."""
    model.zero_grad()

    obs_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.long, device=device)
    old_logp_tensor = torch.tensor(old_log_prob, dtype=torch.float32, device=device).unsqueeze(0)
    advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=device).unsqueeze(0)
    return_tensor = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(0)

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
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()

    return _extract_flat_gradient(model)


def augment_ppo(
    f_in: h5py.File,
    f_out: h5py.File,
    model: ActorCritic,
    out_idx: int,
    num_augmentations: int,
    jitter_kwargs: dict,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> int:
    """Augment a PPO dataset using the exact PPO loss with episode-buffered GAE targets.

    First pass: re-runs the model on every stored observation to collect
    values and log-probs, then groups steps by episode_id into
    PPOGradientBuffers.  Second pass: for each augmentation, applies colour
    jitter to each step and recomputes the exact per-step PPO gradient using
    the buffered GAE advantages and returns.

    Args:
        f_in: Open input HDF5 file.
        f_out: Open output HDF5 file (datasets already created and originals copied).
        model: Frozen ActorCritic checkpoint.
        out_idx: Write cursor — index of the first augmented slot in f_out.
        num_augmentations: Number of colour-jitter copies per original step.
        jitter_kwargs: kwargs forwarded to apply_color_jitter.
        gamma, lam: GAE discount and lambda.
        clip_eps, vf_coef, ent_coef: PPO loss coefficients.

    Returns:
        Updated out_idx after all augmented samples are written.
    """
    num_original = f_in["images"].shape[0]

    # --- Pass 1: collect episode buffers ---
    print("\n[PPO] Collecting episode data for GAE computation...")
    episode_buffers: Dict[int, PPOGradientBuffer] = {}

    for i in tqdm(range(num_original), desc="Collecting episodes"):
        sample_obs = f_in["images"][i]
        sample_action = int(f_in["actions"][i])
        sample_reward = float(f_in["rewards"][i])
        sample_done = bool(f_in["done"][i])
        sample_ep_id = int(f_in["episode_ids"][i])

        if sample_ep_id not in episode_buffers:
            episode_buffers[sample_ep_id] = PPOGradientBuffer(gamma=gamma, lam=lam)

        obs_tensor = torch.tensor(sample_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = torch.tensor(sample_action, dtype=torch.long, device=device)
            log_prob = dist.log_prob(action_tensor).item()

        episode_buffers[sample_ep_id].add(
            obs=sample_obs,
            action=sample_action,
            reward=sample_reward,
            done=sample_done,
            value=value.item(),
            log_prob=log_prob,
        )

    print(f"  Collected {len(episode_buffers)} episodes")

    # --- Pass 2: augment ---
    for aug_num in range(num_augmentations):
        print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
        for ep_id, ep_buf in tqdm(episode_buffers.items(), desc="Augmenting episodes"):
            bootstrap = 0.0 if ep_buf.dones[-1] else ep_buf.values[-1]
            advantages, returns = ep_buf.compute_gae_and_returns(next_value=bootstrap)

            if len(advantages) > 1:
                adv_arr = np.array(advantages)
                advantages = list((adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8))

            for step_i in range(len(ep_buf)):
                aug_image = apply_color_jitter(ep_buf.obs_list[step_i], **jitter_kwargs)
                flat_grads = _compute_ppo_gradient_from_targets(
                    model,
                    aug_image,
                    ep_buf.actions[step_i],
                    ep_buf.log_probs[step_i],
                    advantages[step_i],
                    returns[step_i],
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                )

                f_out["images"][out_idx] = aug_image
                f_out["gradients"][out_idx] = flat_grads.astype(np.float16)
                f_out["actions"][out_idx] = ep_buf.actions[step_i]
                f_out["rewards"][out_idx] = ep_buf.rewards[step_i]
                f_out["episode_ids"][out_idx] = ep_id + (aug_num + 1) * 100_000
                f_out["done"][out_idx] = ep_buf.dones[step_i]
                out_idx += 1

    return out_idx

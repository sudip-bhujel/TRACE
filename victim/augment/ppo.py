from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from victim.augment.image import apply_color_jitter
from victim.capture.buffers import PPOGradientBuffer
from victim.capture.utils import _extract_flat_gradient
from victim.models.actor_critic import (
    ActorCritic,
    forward_actor_critic,
    initial_recurrent_state,
)
from victim.observations import NumpyObservation, observation_to_torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _compute_ppo_gradient_from_targets(
    model: ActorCritic,
    observation: NumpyObservation,
    action: int,
    old_log_prob: float,
    advantage: float,
    returns: float,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    recurrent_state: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
    model.zero_grad()
    model_device = next(model.parameters()).device

    obs_tensor = observation_to_torch(observation, model_device)
    action_tensor = torch.tensor(action, dtype=torch.long, device=model_device)
    old_logp_tensor = torch.tensor(
        old_log_prob, dtype=torch.float32, device=model_device
    ).unsqueeze(0)
    advantage_tensor = torch.tensor(
        advantage, dtype=torch.float32, device=model_device
    ).unsqueeze(0)
    return_tensor = torch.tensor(
        returns, dtype=torch.float32, device=model_device
    ).unsqueeze(0)

    logits, value, next_recurrent_state = forward_actor_critic(
        model, obs_tensor, recurrent_state
    )
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

    return (
        _extract_flat_gradient(model),
        next_recurrent_state.detach()
        if next_recurrent_state is not None
        else None,
    )


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
    """Augment a PPO dataset using exact PPO targets."""
    num_original = f_in["images"].shape[0]
    target_keys = ("old_log_probs", "advantages", "returns")
    available_targets = [key in f_in for key in target_keys]

    if any(available_targets) and not all(available_targets):
        raise ValueError(
            "PPO target data is incomplete; expected old_log_probs, advantages, "
            "and returns"
        )

    if all(available_targets):
        print("\n[PPO] Using exact PPO targets stored during capture")
        model_device = next(model.parameters()).device
        has_goals = "goals" in f_in
        for aug_num in range(num_augmentations):
            print(f"\n[Augmentation {aug_num + 1}/{num_augmentations}]")
            current_episode_id = None
            recurrent_state = None
            for i in tqdm(range(num_original), desc="Augmenting samples"):
                image = apply_color_jitter(f_in["images"][i], **jitter_kwargs)
                action = int(f_in["actions"][i])
                episode_id = int(f_in["episode_ids"][i])
                if episode_id != current_episode_id:
                    recurrent_state = initial_recurrent_state(model, 1, model_device)
                    current_episode_id = episode_id

                observation: NumpyObservation = image
                if has_goals:
                    observation = {
                        "rgb": image,
                        "goal": f_in["goals"][i],
                    }

                flat_grads, recurrent_state = _compute_ppo_gradient_from_targets(
                    model,
                    observation,
                    action,
                    float(f_in["old_log_probs"][i]),
                    float(f_in["advantages"][i]),
                    float(f_in["returns"][i]),
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    recurrent_state=recurrent_state,
                )

                f_out["images"][out_idx] = image
                if has_goals:
                    f_out["goals"][out_idx] = f_in["goals"][i]
                f_out["gradients"][out_idx] = flat_grads.astype(np.float16)
                f_out["actions"][out_idx] = action
                f_out["rewards"][out_idx] = f_in["rewards"][i]
                f_out["episode_ids"][out_idx] = (
                    int(f_in["episode_ids"][i]) + (aug_num + 1) * 100_000
                )
                f_out["done"][out_idx] = f_in["done"][i]
                out_idx += 1
        return out_idx

    print("\n[PPO] Legacy dataset: reconstructing PPO targets from episode data")
    if getattr(model, "is_recurrent", False) or "goals" in f_in:
        raise ValueError(
            "Recurrent and RGB-goal augmentation require exact PPO targets "
            "stored during capture"
        )
    episode_buffers: Dict[int, PPOGradientBuffer] = {}

    for i in tqdm(range(num_original), desc="Collecting episodes"):
        sample_obs = f_in["images"][i]
        sample_action = int(f_in["actions"][i])
        sample_reward = float(f_in["rewards"][i])
        sample_done = bool(f_in["done"][i])
        sample_ep_id = int(f_in["episode_ids"][i])

        if sample_ep_id not in episode_buffers:
            episode_buffers[sample_ep_id] = PPOGradientBuffer(gamma=gamma, lam=lam)

        obs_tensor = torch.tensor(
            sample_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
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
                flat_grads, _ = _compute_ppo_gradient_from_targets(
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

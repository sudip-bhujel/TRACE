import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.buffers import PPOGradientBuffer
from victim.capture.utils import (
    _extract_flat_gradient,
    create_hdf5_dataset,
    flatten_gradients,
)
from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import (
    ActorCritic,
    TensorObservation,
    forward_actor_critic,
    initial_recurrent_state,
)
from victim.observations import observation_image, observation_to_torch


def _infer_written_steps(gradients: h5py.Dataset) -> int:
    """Find the first unwritten row in a contiguous, preallocated gradient set."""
    total_rows = gradients.shape[0]
    if total_rows == 0 or not np.any(gradients[0]):
        return 0
    if np.any(gradients[total_rows - 1]):
        return total_rows

    low, high = 1, total_rows - 1
    while low < high:
        mid = (low + high) // 2
        if np.any(gradients[mid]):
            low = mid + 1
        else:
            high = mid
    return low


def compute_gradients(
    model: ActorCritic,
    observation: TensorObservation,
    action: int,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
    recurrent_state: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Probe-loss gradient: -log pi(a|s) + 0.5 * V(s)."""
    model.zero_grad()

    logits, value, _ = forward_actor_critic(model, observation, recurrent_state)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    log_prob = dist.log_prob(torch.tensor([action], device=logits.device))
    loss = -log_prob.mean() + 0.5 * value.mean()
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
    observation: TensorObservation,
    action: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
    use_float16: bool = True,
    recurrent_state: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Per-step gradient under the PPO clipped surrogate loss."""
    model.zero_grad()

    logits, value, _ = forward_actor_critic(model, observation, recurrent_state)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    new_logp = dist.log_prob(action)
    ratio = torch.exp(new_logp - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()

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


def _client_gradient_ppo(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Exact PPO loss gradient over a local rollout."""
    model.zero_grad()
    logits, values = model(obs)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()
    return _extract_flat_gradient(model, gradient_layers)


def _client_gradient_a2c(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gradient_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """Exact A2C loss gradient over a local rollout."""
    model.zero_grad()
    logits, values = model(obs)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    (policy_loss + vf_coef * value_loss - ent_coef * entropy).backward()
    return _extract_flat_gradient(model, gradient_layers)


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
    """Capture per-step PPO gradients from a frozen checkpoint."""
    model_device = next(model.parameters()).device
    obs = env.reset()
    obs_tensor = observation_to_torch(obs, model_device)
    action_tensor = torch.tensor(0, dtype=torch.long, device=model_device)
    test_grads = compute_ppo_gradients(
        model=model,
        observation=obs_tensor,
        action=action_tensor,
        old_log_prob=torch.tensor(0.0, device=model_device).unsqueeze(0),
        advantage=torch.tensor(1.0, device=model_device).unsqueeze(0),
        returns=torch.tensor(1.0, device=model_device).unsqueeze(0),
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gradient_layers=gradient_layers,
    )
    gradient_size = len(flatten_gradients(test_grads))
    estimated_steps = num_trajectories * (max_steps // 2)

    print("PPO Gradient Capture (per-step)")
    print(
        f"  clip_eps={clip_eps}, vf_coef={vf_coef}, ent_coef={ent_coef}, gamma={gamma}, lam={lam}"
    )
    print(
        f"  Gradient size: {gradient_size:,} ({gradient_size * 2 / 1024:.1f} KB/step)"
    )
    print(f"  Estimated total steps: ~{estimated_steps:,}")

    create_hdf5_dataset(
        save_path,
        num_steps=estimated_steps,
        gradient_size=gradient_size,
        image_shape=observation_image(obs).shape,
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
            recurrent_state = initial_recurrent_state(model, 1, model_device)

            while not done and ep_steps < max_steps:
                obs_tensor = observation_to_torch(obs, model_device)
                state_before = recurrent_state
                with torch.no_grad():
                    logits, value, next_recurrent_state = forward_actor_critic(
                        model, obs_tensor, recurrent_state
                    )
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()
                    log_prob = dist.log_prob(
                        torch.tensor(action, device=model_device)
                    ).item()

                next_obs, reward, done, info = env.step(action)
                ep_reward += reward
                ppo_buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value.item(),
                    log_prob=log_prob,
                    recurrent_state=state_before.squeeze(0).cpu().numpy()
                    if state_before is not None
                    else None,
                )
                if next_recurrent_state is not None:
                    recurrent_state = next_recurrent_state.detach()
                obs = next_obs
                ep_steps += 1

            if done:
                bootstrap_value = 0.0
            else:
                with torch.no_grad():
                    next_obs_tensor = observation_to_torch(obs, model_device)
                    _, next_val, _ = forward_actor_critic(
                        model, next_obs_tensor, recurrent_state
                    )
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
                step_obs_tensor = observation_to_torch(
                    ppo_buffer.obs_list[step_in_episode], model_device
                )
                step_action_tensor = torch.tensor(
                    ppo_buffer.actions[step_in_episode],
                    dtype=torch.long,
                    device=model_device,
                )
                step_old_logp_tensor = torch.tensor(
                    ppo_buffer.log_probs[step_in_episode],
                    dtype=torch.float32,
                    device=model_device,
                ).unsqueeze(0)
                step_advantage_tensor = torch.tensor(
                    advantages[step_in_episode],
                    dtype=torch.float32,
                    device=model_device,
                ).unsqueeze(0)
                step_return_tensor = torch.tensor(
                    returns[step_in_episode],
                    dtype=torch.float32,
                    device=model_device,
                ).unsqueeze(0)
                stored_state = ppo_buffer.recurrent_states[step_in_episode]
                step_recurrent_state = (
                    torch.tensor(
                        stored_state,
                        dtype=torch.float32,
                        device=model_device,
                    ).unsqueeze(0)
                    if stored_state is not None
                    else None
                )

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
                    recurrent_state=step_recurrent_state,
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

                f["images"][step_idx] = observation_image(
                    ppo_buffer.obs_list[step_in_episode]
                )
                f["gradients"][step_idx] = step_flat_grads
                f["actions"][step_idx] = ppo_buffer.actions[step_in_episode]
                f["rewards"][step_idx] = ppo_buffer.rewards[step_in_episode]
                f["episode_ids"][step_idx] = traj_idx
                f["done"][step_idx] = ppo_buffer.dones[step_in_episode]
                step_idx += 1

            ppo_buffer.clear()
            success = "ok" if info.get("distance", float("inf")) < 1.0 else "miss"
            episode_rewards.append(ep_reward)
            print(
                f"Trajectory {traj_idx + 1:4d}/{num_trajectories} | Steps: {ep_steps:3d} | Reward: {ep_reward:7.2f} | {success} | Total: {step_idx:,}"
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


def capture_ppo_uniform_per_scene(
    model: ActorCritic,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    steps_per_scene: int = 1000,
    max_steps_per_episode: int = 100,
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    resume: bool = False,
) -> int:
    """Capture a fixed number of exact per-step PPO gradients from each scene."""
    model_device = next(model.parameters()).device
    obs = env.reset(scene=scenes[0])
    obs_tensor = observation_to_torch(obs, model_device)
    test_grads = compute_ppo_gradients(
        model=model,
        observation=obs_tensor,
        action=torch.tensor(0, dtype=torch.long, device=model_device),
        old_log_prob=torch.tensor(0.0, device=model_device).unsqueeze(0),
        advantage=torch.tensor(1.0, device=model_device).unsqueeze(0),
        returns=torch.tensor(1.0, device=model_device).unsqueeze(0),
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gradient_layers=gradient_layers,
    )
    gradient_size = len(flatten_gradients(test_grads))
    total_steps = steps_per_scene * len(scenes)

    print("Exact PPO Gradient Capture (uniform per scene)")
    print(
        f"  {steps_per_scene} steps x {len(scenes)} scenes = {total_steps:,} total"
    )
    print(f"  Gradient size: {gradient_size:,}")

    if resume:
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"Cannot resume capture because '{save_path}' does not exist"
            )
        with h5py.File(save_path, "r") as existing:
            required_datasets = {
                "images",
                "gradients",
                "actions",
                "rewards",
                "episode_ids",
                "done",
                "old_log_probs",
                "advantages",
                "returns",
            }
            missing = required_datasets.difference(existing.keys())
            if missing:
                raise ValueError(
                    "Cannot resume capture; HDF5 file is missing datasets: "
                    + ", ".join(sorted(missing))
                )
            if existing["gradients"].shape != (total_steps, gradient_size):
                raise ValueError(
                    "Cannot resume capture; existing gradient shape "
                    f"{existing['gradients'].shape} does not match "
                    f"{(total_steps, gradient_size)}"
                )

            recorded_steps = existing.attrs.get("completed_steps")
            if recorded_steps is not None:
                recorded_steps = int(recorded_steps)
            if (
                recorded_steps is None
                or recorded_steps < 0
                or recorded_steps > total_steps
                or (
                    recorded_steps < total_steps
                    and np.any(existing["gradients"][recorded_steps])
                )
            ):
                step_idx = _infer_written_steps(existing["gradients"])
            else:
                step_idx = recorded_steps

            episode_idx = (
                int(existing["episode_ids"][step_idx - 1]) + 1
                if step_idx > 0
                else 0
            )
        print(
            f"  Resuming at row {step_idx:,}/{total_steps:,} "
            f"(scene {step_idx // steps_per_scene + 1}, "
            f"step {step_idx % steps_per_scene})"
        )
    else:
        create_hdf5_dataset(
            save_path,
            num_steps=total_steps,
            gradient_size=gradient_size,
            image_shape=observation_image(obs).shape,
            compression=compression,
            with_ppo_targets=True,
        )
        step_idx = 0
        episode_idx = 0

    episode_rewards = []

    with h5py.File(save_path, "a") as f:
        f.attrs["completed_steps"] = step_idx
        start_scene_idx = min(step_idx // steps_per_scene, len(scenes))

        for scene_idx in range(start_scene_idx, len(scenes)):
            scene = scenes[scene_idx]
            scene_steps = max(0, step_idx - scene_idx * steps_per_scene)
            print(
                f"\n[Scene {scene_idx + 1}/{len(scenes)}] {scene}: "
                f"capturing {steps_per_scene - scene_steps} remaining exact PPO steps..."
            )

            while scene_steps < steps_per_scene:
                remaining = steps_per_scene - scene_steps
                rollout_limit = min(max_steps_per_episode, remaining)
                ppo_buffer = PPOGradientBuffer(gamma=gamma, lam=lam)
                obs = env.reset(scene=scene)
                done = False
                ep_reward = 0.0
                recurrent_state = initial_recurrent_state(model, 1, model_device)

                while not done and len(ppo_buffer) < rollout_limit:
                    obs_tensor = observation_to_torch(obs, model_device)
                    state_before = recurrent_state
                    with torch.no_grad():
                        logits, value, next_recurrent_state = forward_actor_critic(
                            model, obs_tensor, recurrent_state
                        )
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()
                        log_prob = dist.log_prob(
                            torch.tensor(action, device=model_device)
                        ).item()

                    next_obs, reward, done, _info = env.step(action)
                    ppo_buffer.add(
                        obs=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value.item(),
                        log_prob=log_prob,
                        recurrent_state=state_before.squeeze(0).cpu().numpy()
                        if state_before is not None
                        else None,
                    )
                    if next_recurrent_state is not None:
                        recurrent_state = next_recurrent_state.detach()
                    obs = next_obs
                    ep_reward += reward

                if done:
                    bootstrap_value = 0.0
                else:
                    with torch.no_grad():
                        next_obs_tensor = observation_to_torch(obs, model_device)
                        _, next_value, _ = forward_actor_critic(
                            model, next_obs_tensor, recurrent_state
                        )
                        bootstrap_value = next_value.item()

                advantages, returns = ppo_buffer.compute_gae_and_returns(
                    next_value=bootstrap_value
                )
                if len(advantages) > 1:
                    adv_array = np.asarray(advantages)
                    advantages = list(
                        (adv_array - adv_array.mean()) / (adv_array.std() + 1e-8)
                    )

                for step_in_episode in range(len(ppo_buffer)):
                    step_obs = observation_to_torch(
                        ppo_buffer.obs_list[step_in_episode], model_device
                    )
                    step_action = torch.tensor(
                        ppo_buffer.actions[step_in_episode],
                        dtype=torch.long,
                        device=model_device,
                    )
                    old_log_prob = ppo_buffer.log_probs[step_in_episode]
                    advantage = advantages[step_in_episode]
                    return_value = returns[step_in_episode]
                    stored_state = ppo_buffer.recurrent_states[step_in_episode]
                    step_recurrent_state = (
                        torch.tensor(
                            stored_state,
                            dtype=torch.float32,
                            device=model_device,
                        ).unsqueeze(0)
                        if stored_state is not None
                        else None
                    )

                    grads = compute_ppo_gradients(
                        model=model,
                        observation=step_obs,
                        action=step_action,
                        old_log_prob=torch.tensor(
                            old_log_prob, dtype=torch.float32, device=model_device
                        ).unsqueeze(0),
                        advantage=torch.tensor(
                            advantage, dtype=torch.float32, device=model_device
                        ).unsqueeze(0),
                        returns=torch.tensor(
                            return_value, dtype=torch.float32, device=model_device
                        ).unsqueeze(0),
                        clip_eps=clip_eps,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        gradient_layers=gradient_layers,
                        recurrent_state=step_recurrent_state,
                    )

                    f["images"][step_idx] = observation_image(
                        ppo_buffer.obs_list[step_in_episode]
                    )
                    f["gradients"][step_idx] = flatten_gradients(grads)
                    f["actions"][step_idx] = ppo_buffer.actions[step_in_episode]
                    f["rewards"][step_idx] = ppo_buffer.rewards[step_in_episode]
                    f["episode_ids"][step_idx] = episode_idx
                    f["done"][step_idx] = ppo_buffer.dones[step_in_episode]
                    f["old_log_probs"][step_idx] = old_log_prob
                    f["advantages"][step_idx] = advantage
                    f["returns"][step_idx] = return_value
                    step_idx += 1
                    scene_steps += 1

                episode_rewards.append(ep_reward)
                episode_idx += 1
                f.attrs["completed_steps"] = step_idx

            print(f"  -> {scene}: {scene_steps} steps")
            f.flush()

        f.attrs["num_scenes"] = len(scenes)
        f.attrs["steps_per_scene"] = steps_per_scene
        f.attrs["total_steps"] = step_idx
        f.attrs["completed_steps"] = step_idx
        f.attrs["gradient_size"] = gradient_size
        f.attrs["capture_mode"] = "ppo_exact_uniform_per_scene"
        f.attrs["loss_type"] = "ppo"
        f.attrs["ppo_clip_eps"] = clip_eps
        f.attrs["ppo_vf_coef"] = vf_coef
        f.attrs["ppo_ent_coef"] = ent_coef
        f.attrs["gae_gamma"] = gamma
        f.attrs["gae_lambda"] = lam
        if episode_rewards:
            f.attrs["avg_reward"] = np.mean(episode_rewards)
        grad_names = sorted(test_grads.keys())
        f.attrs["gradient_names"] = [name.encode() for name in grad_names]
        f.attrs["gradient_shapes"] = [
            str(test_grads[name].shape) for name in grad_names
        ]

    return step_idx

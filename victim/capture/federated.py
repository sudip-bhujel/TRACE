import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from victim.capture.ppo import _client_gradient_a2c, _client_gradient_ppo
from victim.environment import AI2THORNavEnv
from victim.models.actor_critic import ActorCritic, compute_gae

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _collect_ppo_rollout(
    model: ActorCritic,
    env: AI2THORNavEnv,
    rollout_steps: int,
    scene: Optional[str] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    obs = env.reset(scene=scene) if scene else env.reset()
    obs_list, act_list, logp_list, rew_list, done_list, val_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    images: List[np.ndarray] = []

    for _ in range(rollout_steps):
        images.append(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=device)).item()

        next_obs, reward, done, _ = env.step(action)
        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(logp)
        rew_list.append(reward)
        done_list.append(done)
        val_list.append(value.item())
        obs = (
            next_obs if not done else (env.reset(scene=scene) if scene else env.reset())
        )

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_val = model(obs_t)

    advantages, returns = compute_gae(
        rew_list, val_list + [last_val.item()], done_list, gamma=gamma, lam=lam
    )
    adv = np.array(advantages, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return (
        torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device),
        torch.tensor(act_list, dtype=torch.long, device=device),
        torch.tensor(logp_list, dtype=torch.float32, device=device),
        torch.tensor(adv, dtype=torch.float32, device=device),
        torch.tensor(returns, dtype=torch.float32, device=device),
        np.stack(images),
    )


def capture_federated(
    model: torch.nn.Module,
    env: AI2THORNavEnv,
    save_path: str,
    scenes: List[str],
    num_rounds: int = 100,
    num_clients: int = 10,
    rollout_steps: int = 200,
    algorithm: str = "ppo",
    gradient_layers: Optional[List[str]] = None,
    compression: str = "gzip",
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> int:
    """Simulate federated rounds and store one client gradient per record."""
    print(f"[FL Capture] {algorithm.upper()} probe")
    scene0 = scenes[0]
    if algorithm == "ppo":
        obs_t, act_t, logp_t, adv_t, ret_t, imgs = _collect_ppo_rollout(
            model, env, rollout_steps, scene=scene0, gamma=gamma, lam=lam
        )
        grad_probe = _client_gradient_ppo(
            model,
            obs_t,
            act_t,
            logp_t,
            adv_t,
            ret_t,
            clip_eps,
            vf_coef,
            ent_coef,
            gradient_layers,
        )
    elif algorithm == "a2c":
        obs_t, act_t, logp_t, adv_t, ret_t, imgs = _collect_ppo_rollout(
            model, env, rollout_steps, scene=scene0, gamma=gamma, lam=lam
        )
        grad_probe = _client_gradient_a2c(
            model,
            obs_t,
            act_t,
            adv_t,
            ret_t,
            vf_coef,
            ent_coef,
            gradient_layers,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: ppo, a2c")

    grad_dim = grad_probe.shape[0]
    img_shape = imgs.shape[1:]
    total_records = num_rounds * num_clients

    print(f"  grad_dim={grad_dim:,} ({grad_dim * 2 / 1024:.1f} KB)")
    print(f"  records={total_records:,} ({num_rounds} rounds x {num_clients} clients)")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset(
            "gradients",
            shape=(total_records, grad_dim),
            dtype=np.float16,
            compression=compression,
        )
        f.create_dataset(
            "images",
            shape=(total_records, rollout_steps, *img_shape),
            dtype=np.uint8,
            compression=compression,
        )
        f.create_dataset(
            "actions",
            shape=(total_records, rollout_steps),
            dtype=np.int8,
            compression=compression,
        )
        f.create_dataset(
            "rewards",
            shape=(total_records, rollout_steps),
            dtype=np.float32,
            compression=compression,
        )
        f.create_dataset(
            "client_ids",
            shape=(total_records,),
            dtype=np.int32,
            compression=compression,
        )
        f.create_dataset(
            "round_ids", shape=(total_records,), dtype=np.int32, compression=compression
        )

    record_idx = 0
    with h5py.File(save_path, "a") as f:
        for round_idx in range(num_rounds):
            for client_idx in range(num_clients):
                scene = scenes[client_idx % len(scenes)]

                if algorithm == "ppo":
                    obs_t, act_t, logp_t, adv_t, ret_t, images = _collect_ppo_rollout(
                        model, env, rollout_steps, scene=scene, gamma=gamma, lam=lam
                    )
                    grad = _client_gradient_ppo(
                        model,
                        obs_t,
                        act_t,
                        logp_t,
                        adv_t,
                        ret_t,
                        clip_eps,
                        vf_coef,
                        ent_coef,
                        gradient_layers,
                    )
                    actions_np = act_t.cpu().numpy().astype(np.int8)
                    rewards_np = ret_t.cpu().numpy().astype(np.float32)
                elif algorithm == "a2c":
                    obs_t, act_t, logp_t, adv_t, ret_t, images = _collect_ppo_rollout(
                        model, env, rollout_steps, scene=scene, gamma=gamma, lam=lam
                    )
                    grad = _client_gradient_a2c(
                        model,
                        obs_t,
                        act_t,
                        adv_t,
                        ret_t,
                        vf_coef,
                        ent_coef,
                        gradient_layers,
                    )
                    actions_np = act_t.cpu().numpy().astype(np.int8)
                    rewards_np = ret_t.cpu().numpy().astype(np.float32)

                f["gradients"][record_idx] = grad.astype(np.float16)
                f["images"][record_idx] = images
                f["actions"][record_idx] = actions_np
                f["rewards"][record_idx] = rewards_np
                f["client_ids"][record_idx] = client_idx
                f["round_ids"][record_idx] = round_idx
                record_idx += 1

            if (round_idx + 1) % 10 == 0:
                print(
                    f"  Round {round_idx + 1:4d}/{num_rounds} | Records: {record_idx:,}"
                )

        f.attrs["algorithm"] = algorithm
        f.attrs["capture_mode"] = "federated"
        f.attrs["num_rounds"] = num_rounds
        f.attrs["num_clients"] = num_clients
        f.attrs["rollout_steps"] = rollout_steps
        f.attrs["grad_dim"] = grad_dim
        f.attrs["total_records"] = record_idx
        f.attrs["scenes"] = [s.encode() for s in scenes]

    print(f"\nFederated capture done. {record_idx:,} records -> {save_path}")
    return record_idx

import json
import os
import sys

import h5py

from victim.capture.a2c import capture_a2c_gradients
from victim.capture.federated import capture_federated
from victim.capture.ppo import (
    capture_ppo_gradients,
    capture_ppo_uniform_per_scene,
)
from victim.capture.streaming import (
    capture_and_save_streaming,
    capture_uniform_per_scene,
)
from victim.capture.utils import load_model, print_file_info
from victim.config_utils import load_config
from victim.environment import AI2THORNavEnv

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m victim.capture.capture <config_path>"

    cfg = load_config(sys.argv[1], sys.argv[2:])
    env_cfg = cfg.get("environment", {})
    model_cfg = cfg.get("model", {})
    capture_cfg = cfg.get("capture", {})

    algorithm = model_cfg.get("algorithm", "ppo").lower()

    if "scenes" in env_cfg:
        scenes = list(env_cfg.get("scenes"))
    elif "scene" in env_cfg:
        scenes = [env_cfg.get("scene")]
    else:
        scenes = ["FloorPlan1"]

    print(
        f"Gradient Capture | algorithm={algorithm.upper()} | scenes={len(scenes)} | "
        f"checkpoint={model_cfg.get('checkpoint')}"
    )

    os.makedirs(os.path.dirname(capture_cfg.get("save_path")) or ".", exist_ok=True)

    model = load_model(
        model_cfg.get("checkpoint"),
        num_actions=model_cfg.get("num_actions"),
        algorithm=algorithm,
        architecture=model_cfg.get("architecture"),
        device_name=cfg.get("device", "auto"),
    )

    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=tuple(env_cfg.get("image_size", [84, 84])),
        max_steps=env_cfg.get("max_steps", 200),
        headless=env_cfg.get("headless", False),
        action_set=env_cfg.get("action_set", "nav5"),
    )
    if env.action_space_n != model.num_actions:
        raise ValueError(
            f"Environment action set '{env.action_set}' has {env.action_space_n} "
            f"actions, but the victim model has {model.num_actions}"
        )

    capture_mode = capture_cfg.get("mode", "uniform")
    steps_per_scene = capture_cfg.get("steps_per_scene", None)

    ppo_params = {
        "gamma": capture_cfg.get("gae_gamma", 0.99),
        "lam": capture_cfg.get("gae_lambda", 0.95),
        "clip_eps": capture_cfg.get("ppo_clip_eps", 0.2),
        "vf_coef": capture_cfg.get("ppo_vf_coef", 0.5),
        "ent_coef": capture_cfg.get("ppo_ent_coef", 0.01),
    }
    a2c_params = {
        "gamma": capture_cfg.get("gae_gamma", 0.99),
        "lam": capture_cfg.get("gae_lambda", 0.95),
        "vf_coef": capture_cfg.get("vf_coef", 0.5),
        "ent_coef": capture_cfg.get("ent_coef", 0.01),
    }

    try:
        if algorithm == "a2c":
            total_steps = capture_a2c_gradients(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                scenes=scenes,
                steps_per_scene=steps_per_scene or 1000,
                max_steps_per_episode=env_cfg.get("max_steps", 100),
                gradient_layers=capture_cfg.get("gradient_layers"),
                compression=capture_cfg.get("compression", "gzip"),
                **a2c_params,
            )

        elif capture_mode == "federated":
            total_steps = capture_federated(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                scenes=scenes,
                num_rounds=capture_cfg.get("num_rounds", 100),
                num_clients=capture_cfg.get("num_clients", 10),
                rollout_steps=capture_cfg.get("rollout_steps", 200),
                algorithm=algorithm,
                gradient_layers=capture_cfg.get("gradient_layers"),
                compression=capture_cfg.get("compression", "gzip"),
                gamma=ppo_params["gamma"],
                lam=ppo_params["lam"],
                clip_eps=ppo_params["clip_eps"],
                vf_coef=ppo_params["vf_coef"],
                ent_coef=ppo_params["ent_coef"],
            )

        elif capture_mode == "ppo":
            total_steps = capture_ppo_gradients(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                num_trajectories=capture_cfg.get("num_trajectories", 100),
                max_steps=env_cfg.get("max_steps", 200),
                gradient_layers=capture_cfg.get("gradient_layers"),
                scenes=scenes,
                **ppo_params,
            )

        elif steps_per_scene and len(scenes) > 1:
            if capture_cfg.get("use_ppo_loss", True):
                total_steps = capture_ppo_uniform_per_scene(
                    model=model,
                    env=env,
                    save_path=capture_cfg.get("save_path"),
                    scenes=scenes,
                    steps_per_scene=steps_per_scene,
                    max_steps_per_episode=env_cfg.get("max_steps", 100),
                    gradient_layers=capture_cfg.get("gradient_layers"),
                    compression=capture_cfg.get("compression", "gzip"),
                    resume=capture_cfg.get("resume", False),
                    **ppo_params,
                )
            else:
                total_steps = capture_uniform_per_scene(
                    model=model,
                    env=env,
                    save_path=capture_cfg.get("save_path"),
                    scenes=scenes,
                    steps_per_scene=steps_per_scene,
                    max_steps_per_episode=env_cfg.get("max_steps", 100),
                    gradient_layers=capture_cfg.get("gradient_layers"),
                    compression=capture_cfg.get("compression", "gzip"),
                    algorithm=algorithm,
                )

        else:
            total_steps = capture_and_save_streaming(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                num_trajectories=capture_cfg.get("num_trajectories", 100),
                max_steps=env_cfg.get("max_steps", 200),
                gradient_layers=capture_cfg.get("gradient_layers"),
                scenes=scenes,
                algorithm=algorithm,
            )

        save_path = capture_cfg.get("save_path")
        with h5py.File(save_path, "a") as h5_file:
            h5_file.attrs["victim_architecture"] = model.architecture
            h5_file.attrs["num_actions"] = model.num_actions
            h5_file.attrs["action_set"] = env.action_set
            h5_file.attrs["action_names"] = json.dumps(env.action_names)

        print_file_info(save_path)
    finally:
        env.close()

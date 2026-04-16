import copy
import os
import sys

from omegaconf import OmegaConf

from victim.capture.a2c import capture_a2c_gradients
from victim.capture.federated import capture_federated
from victim.capture.ppo import capture_ppo_gradients
from victim.capture.sac import capture_sac_exact_gradients
from victim.capture.streaming import capture_and_save_streaming, capture_uniform_per_scene
from victim.capture.utils import load_model, print_file_info
from victim.environment import AI2THORNavEnv

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m victim.capture.capture <config_path>"

    cfg = OmegaConf.load(sys.argv[1])
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

    print("=" * 60)
    print(f"Gradient Capture — {algorithm.upper()} Agent")
    print("=" * 60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Scenes: {len(scenes)} scene(s)")
    print(f"Checkpoint: {model_cfg.get('checkpoint')}")

    os.makedirs(os.path.dirname(capture_cfg.get("save_path")) or ".", exist_ok=True)

    print(f"\nLoading model from: {model_cfg.get('checkpoint')}")
    model = load_model(
        model_cfg.get("checkpoint"),
        num_actions=model_cfg.get("num_actions", 5),
        algorithm=algorithm,
    )

    print(f"\nInitializing AI2-THOR environment (starting scene={scenes[0]})...")
    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=(84, 84),
        max_steps=env_cfg.get("max_steps"),
        headless=env_cfg.get("headless"),
    )

    capture_mode = capture_cfg.get("mode", "uniform")
    steps_per_scene = capture_cfg.get("steps_per_scene", None)
    use_exact_loss = capture_cfg.get("use_exact_loss", False)

    # Common GAE / PPO params (used by PPO and federated modes)
    ppo_params = {
        "gamma":    capture_cfg.get("gae_gamma", 0.99),
        "lam":      capture_cfg.get("gae_lambda", 0.95),
        "clip_eps": capture_cfg.get("ppo_clip_eps", 0.2),
        "vf_coef":  capture_cfg.get("ppo_vf_coef", 0.5),
        "ent_coef": capture_cfg.get("ppo_ent_coef", 0.01),
    }

    # A2C exact-loss params
    a2c_params = {
        "gamma":   capture_cfg.get("gae_gamma", 0.99),
        "lam":     capture_cfg.get("gae_lambda", 0.95),
        "vf_coef": capture_cfg.get("vf_coef", 0.5),
        "ent_coef": capture_cfg.get("ent_coef", 0.01),
    }

    # SAC exact-loss params
    sac_params = {
        "gamma": capture_cfg.get("sac_gamma", 0.99),
        "alpha": capture_cfg.get("sac_alpha", 0.2),
    }

    try:
        # ------------------------------------------------------------------ #
        # A2C — always uses episode-buffered exact A2C loss                  #
        # ------------------------------------------------------------------ #
        if algorithm == "a2c":
            print(
                f"\n[A2C EXACT MODE] Episode-buffered GAE, "
                f"gamma={a2c_params['gamma']}, lam={a2c_params['lam']}, "
                f"vf_coef={a2c_params['vf_coef']}, ent_coef={a2c_params['ent_coef']}"
            )
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

        # ------------------------------------------------------------------ #
        # SAC — exact loss (critic + actor) when use_exact_loss: true        #
        # ------------------------------------------------------------------ #
        elif algorithm == "sac" and use_exact_loss:
            print(
                f"\n[SAC EXACT MODE] Critic + actor loss, "
                f"gamma={sac_params['gamma']}, alpha={sac_params['alpha']}"
            )
            # Target network = copy of the frozen checkpoint (same weights)
            target_model = copy.deepcopy(model)
            target_model.eval()
            target_model.requires_grad_(False)

            total_steps = capture_sac_exact_gradients(
                model=model,
                target_model=target_model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                scenes=scenes,
                steps_per_scene=steps_per_scene or 1000,
                max_steps_per_episode=env_cfg.get("max_steps", 100),
                gradient_layers=capture_cfg.get("gradient_layers"),
                compression=capture_cfg.get("compression", "gzip"),
                **sac_params,
            )

        # ------------------------------------------------------------------ #
        # Federated mode (all algorithms)                                     #
        # ------------------------------------------------------------------ #
        elif capture_mode == "federated":
            print(f"\n[FEDERATED MODE] {algorithm.upper()} — exact training-loss gradients")
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
                alpha=capture_cfg.get("sac_alpha", 0.2),
            )

        # ------------------------------------------------------------------ #
        # PPO per-step exact mode                                             #
        # ------------------------------------------------------------------ #
        elif capture_mode == "ppo":
            print("\n[PPO MODE] Capturing with per-step PPO-style objective...")
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

        # ------------------------------------------------------------------ #
        # Uniform per-scene (probe loss — PPO / SAC without exact loss)      #
        # ------------------------------------------------------------------ #
        elif steps_per_scene and len(scenes) > 1:
            print(f"\n[UNIFORM MODE] Capturing {steps_per_scene} steps per scene...")
            total_steps = capture_uniform_per_scene(
                model=model,
                env=env,
                save_path=capture_cfg.get("save_path"),
                scenes=scenes,
                steps_per_scene=steps_per_scene,
                max_steps_per_episode=env_cfg.get("max_steps"),
                gradient_layers=capture_cfg.get("gradient_layers"),
                algorithm=algorithm,
            )

        # ------------------------------------------------------------------ #
        # Streaming probe mode (fallback)                                     #
        # ------------------------------------------------------------------ #
        else:
            print(f"\n[STREAMING MODE] {capture_cfg.get('num_trajectories')} trajectories...")
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

        print_file_info(capture_cfg.get("save_path"))
    finally:
        env.close()

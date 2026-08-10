import os
import random
import sys

import numpy as np
import torch

from victim.config_utils import load_config
from victim.environment import AI2THORNavEnv
from victim.training.a2c import train_a2c
from victim.training.ppo import train
from victim.training.utils import plot_results

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m victim.training.train <config_path>"

    cfg = load_config(sys.argv[1], sys.argv[2:])

    algorithm = cfg.get("algorithm", "ppo").lower()
    env_cfg = cfg.get("environment", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    if "scenes" in env_cfg:
        scenes = list(env_cfg.get("scenes"))
    elif "scene" in env_cfg:
        scenes = [env_cfg.get("scene")]
    else:
        scenes = ["FloorPlan1"]

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = output_cfg.get("save_dir", f"ckpts/victim/{algorithm}")
    plot_path = output_cfg.get("plot_path", f"output/victim_{algorithm}_results.png")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    print(
        f"Victim {algorithm.upper()} training | architecture={model_cfg.get('architecture', 'cnn')} "
        f"| action_set={env_cfg.get('action_set', 'nav5')} | scenes={scenes} "
        f"| save_dir={save_dir}"
    )

    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=tuple(env_cfg.get("image_size", [84, 84])),
        max_steps=env_cfg.get("max_steps", 200),
        headless=env_cfg.get("headless", False),
        action_set=env_cfg.get("action_set", "nav5"),
        observation_mode=env_cfg.get("observation_mode", "rgb"),
    )

    try:
        if algorithm == "ppo":
            alg_cfg = cfg.get("ppo", {})
            episode_rewards = train(
                env,
                total_updates=alg_cfg.get("total_updates", 500),
                steps_per_update=alg_cfg.get("steps_per_update", 1024),
                mini_batch_size=alg_cfg.get("mini_batch_size", 64),
                ppo_epochs=alg_cfg.get("ppo_epochs", 4),
                gamma=alg_cfg.get("gamma", 0.99),
                lam=alg_cfg.get("gae_lambda", 0.95),
                clip_eps=alg_cfg.get("clip_eps", 0.2),
                lr=alg_cfg.get("learning_rate", 2.5e-4),
                ent_coef=alg_cfg.get("entropy_coef", 0.01),
                vf_coef=alg_cfg.get("value_coef", 0.5),
                max_grad_norm=alg_cfg.get("max_grad_norm", 0.5),
                save_dir=save_dir,
                resume_from=train_cfg.get("resume_from"),
                scenes=scenes,
                architecture=model_cfg.get("architecture", "cnn"),
                device_name=cfg.get("device", "auto"),
            )
        elif algorithm == "a2c":
            alg_cfg = cfg.get("a2c", {})
            episode_rewards = train_a2c(
                env,
                total_updates=alg_cfg.get("total_updates", 500),
                steps_per_update=alg_cfg.get("steps_per_update", 512),
                gamma=alg_cfg.get("gamma", 0.99),
                lam=alg_cfg.get("gae_lambda", 0.95),
                lr=alg_cfg.get("learning_rate", 7e-4),
                ent_coef=alg_cfg.get("entropy_coef", 0.01),
                vf_coef=alg_cfg.get("value_coef", 0.5),
                max_grad_norm=alg_cfg.get("max_grad_norm", 0.5),
                save_dir=save_dir,
                resume_from=train_cfg.get("resume_from"),
                scenes=scenes,
            )
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: ppo, a2c")
    finally:
        env.close()

    if episode_rewards:
        plot_results(episode_rewards, plot_path)

    print("\nTraining complete!")

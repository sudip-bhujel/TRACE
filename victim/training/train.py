import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from victim.environment import AI2THORNavEnv
from victim.training.a2c import train_a2c
from victim.training.ppo import train
from victim.training.sac import train_sac
from victim.training.utils import plot_results

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m victim.training.train <config_path>"

    cfg = OmegaConf.load(sys.argv[1])

    algorithm = cfg.get("algorithm", "ppo").lower()
    env_cfg = cfg.get("environment", {})
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
        f"Victim {algorithm.upper()} training | scenes={scenes} | save_dir={save_dir}"
    )

    env = AI2THORNavEnv(
        scene=scenes[0],
        image_size=tuple(env_cfg.get("image_size", [84, 84])),
        max_steps=env_cfg.get("max_steps", 200),
        headless=env_cfg.get("headless", False),
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
            )
        elif algorithm == "sac":
            alg_cfg = cfg.get("sac", {})
            episode_rewards = train_sac(
                env,
                total_steps=alg_cfg.get("total_steps", 500_000),
                batch_size=alg_cfg.get("batch_size", 256),
                buffer_size=alg_cfg.get("buffer_size", 100_000),
                learning_starts=alg_cfg.get("learning_starts", 5_000),
                gamma=alg_cfg.get("gamma", 0.99),
                tau=alg_cfg.get("tau", 0.005),
                lr=alg_cfg.get("learning_rate", 3e-4),
                target_entropy_ratio=alg_cfg.get("target_entropy_ratio", 0.98),
                train_freq=alg_cfg.get("train_freq", 1),
                save_dir=save_dir,
                resume_from=train_cfg.get("resume_from"),
                scenes=scenes,
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
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: ppo, a2c, sac")
    finally:
        env.close()

    if episode_rewards:
        plot_results(episode_rewards, plot_path)

    print("\nTraining complete!")

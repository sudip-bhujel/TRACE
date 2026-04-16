import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_results(returns: List[float], save_path: str = "output/ppo_thor_results.png"):
    """Plot training results."""
    if not returns:
        print("No episodes completed, skipping plot.")
        return

    # Calculate running average
    running_avg = []
    for i in range(len(returns)):
        start_idx = max(0, i - 99)
        running_avg.append(np.mean(returns[start_idx : i + 1]))

    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Episode Reward", alpha=0.3, color="blue", lw=1)
    plt.plot(running_avg, label="Average Reward (100 episodes)", color="red", lw=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved as '{save_path}'")

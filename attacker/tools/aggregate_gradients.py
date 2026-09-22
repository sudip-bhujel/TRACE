"""Average consecutive captured gradients on CPU; never overwrite source files."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


def aggregate_file(source, destination, group_size=4, num_actions=5):
    """Stream disjoint groups, retaining the last RGB frame and action histogram.

    This averages stored per-step gradients, not recomputed PPO minibatch updates.
    Short episode tails are dropped. Source row ranges are end-exclusive.
    """
    source, destination = Path(source), Path(destination)
    if group_size < 2 or num_actions < 2:
        raise ValueError("group_size and num_actions must both be at least 2")
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    with h5py.File(source, "r") as src:
        required = {"gradients", "images", "actions", "episode_ids", "done"}
        if not required.issubset(src):
            raise ValueError(f"Missing datasets: {sorted(required.difference(src))}")
        n = len(src["gradients"])
        if not n or any(len(src[key]) != n for key in required):
            raise ValueError("Source datasets must have equal, nonzero row counts")
        for key in ("completed_steps", "total_steps"):
            if key in src.attrs and int(src.attrs[key]) != n:
                raise ValueError(f"Source {key} disagrees with row count; check capture completion")
        if src["gradients"].ndim != 2 or src["images"].ndim != 4:
            raise ValueError("Expected gradients [N,D] and images [N,C,H,W]")
        if src["images"].shape[1] != 3 or src["images"].dtype != np.uint8:
            raise ValueError("Expected uint8 RGB images in channel-first layout")
        if int(src.attrs.get("aggregation_size", 1)) != 1:
            raise ValueError("Input is already aggregated; use the original capture")
        if int(src.attrs.get("num_actions", num_actions)) != num_actions:
            raise ValueError("num_actions disagrees with source metadata")

        actions = src["actions"][:]
        episodes = src["episode_ids"][:]
        done = src["done"][:].astype(bool)
        if any(x.shape != (n,) for x in (actions, episodes, done)):
            raise ValueError("Expected per-step action labels, episode IDs and done flags")
        if not np.issubdtype(actions.dtype, np.integer) or np.any(
            (actions < 0) | (actions >= num_actions)
        ):
            raise ValueError("Source actions must be integer labels within num_actions")
        if not np.issubdtype(episodes.dtype, np.integer):
            raise ValueError("Source episode IDs must be integers")

        # Split on both recorded resets and terminal flags, even if an ID repeats.
        boundaries = np.r_[0, np.flatnonzero((episodes[1:] != episodes[:-1]) | done[:-1]) + 1, n]
        count = int(np.sum(np.diff(boundaries) // group_size))
        if not count:
            raise ValueError("No episode segment is long enough for one aggregate")
        loss_type = src.attrs.get("loss_type", "unknown")
        print(f"{source}: {n:,} steps -> {count:,} aggregates; source loss_type={loss_type}")
        if loss_type != "ppo":
            print("WARNING: PPO loss provenance is not verified by this file's metadata.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "x") as dst:
            # Keep source provenance separate from the new aggregate row counts.
            for key, value in src.attrs.items():
                dst.attrs[f"source_{key}"] = value
            for key in ("action_names", "victim_architecture", "gradient_names", "gradient_shapes"):
                if key in src.attrs:
                    dst.attrs[key] = src.attrs[key]
            dst.attrs.update(
                source_path=str(source.resolve()),
                aggregation_size=group_size,
                aggregation_reduction="mean",
                aggregation_complete=False,
                image_target="window_last",
                action_target="histogram",
                num_actions=num_actions,
                total_steps=count,
                completed_steps=0,
                gradient_size=src["gradients"].shape[1],
                dropped_steps=n - count * group_size,
            )
            dst.create_dataset("gradients", (count, src["gradients"].shape[1]), dtype="f4",
                               chunks=(1, src["gradients"].shape[1]), compression="lzf")
            dst.create_dataset("images", (count, *src["images"].shape[1:]), dtype="u1",
                               compression="lzf")
            dst.create_dataset("actions", (count, num_actions), dtype="f4")
            for key in ("episode_ids", "source_episode_ids", "source_start", "source_stop"):
                dst.create_dataset(key, (count,), dtype="i8")
            dst.create_dataset("done", (count,), dtype="bool")

            row = 0
            with tqdm(total=count, desc=destination.name) as progress:
                for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    for first in range(int(start), int(stop) - group_size + 1, group_size):
                        last = first + group_size
                        gradients = src["gradients"][first:last].astype(np.float32)
                        if not np.isfinite(gradients).all():
                            raise ValueError(f"Nonfinite source gradients at rows {first}:{last}")
                        dst["gradients"][row] = gradients.mean(axis=0)
                        dst["images"][row] = src["images"][last - 1]
                        dst["actions"][row] = np.bincount(actions[first:last], minlength=num_actions) / group_size
                        dst["episode_ids"][row] = segment
                        dst["source_episode_ids"][row] = episodes[first]
                        dst["source_start"][row] = first
                        dst["source_stop"][row] = last
                        dst["done"][row] = done[last - 1]
                        row += 1
                        progress.update(1)
            dst.attrs["completed_steps"] = row
            dst.attrs["aggregation_complete"] = True
    print(f"Saved {destination}; dropped {n - count * group_size} trailing steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Aggregation attacker YAML")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    pairs = [
        (cfg.aggregation.train_source, cfg.data.h5_path),
        (cfg.aggregation.test_source, cfg.eval.h5_path),
    ]
    paths = [Path(path).resolve() for pair in pairs for path in pair]
    if len(set(paths)) != len(paths):
        raise ValueError("Train/test sources and outputs must be four distinct paths")
    for source, output in pairs:
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        if Path(output).exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    for source, output in pairs:
        aggregate_file(source, output, cfg.aggregation.group_size, cfg.model.num_actions)


if __name__ == "__main__":
    main()

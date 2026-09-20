import json
from typing import List, Optional, Tuple, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class TemporalGradientDataset(Dataset):
    """Sliding-window sequences of (gradient, image, action) tuples drawn from episodes."""

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 8,
        stride: int = 4,
        gradient_dim: Optional[int] = None,
        gradient_layers: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        num_actions: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.gradient_dim = gradient_dim
        self.gradient_layers = gradient_layers
        self.h5_path = h5_path

        self._layer_slices: Optional[List[Tuple[int, int]]] = None
        if gradient_layers is not None:
            if gradient_dim is not None:
                print(
                    "Warning: Both gradient_layers and gradient_dim are set. "
                    "gradient_layers takes priority; gradient_dim is ignored."
                )
            try:
                from attacker.analyze_gradients import get_layer_map
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "gradient_layers requires attacker.analyze_gradients, which "
                    "is not present in this checkout. Use the full gradient by "
                    "setting gradient_layers: null."
                ) from exc
            layer_map, _total = get_layer_map()

            with h5py.File(h5_path, "r") as h5_tmp:
                gradients_ds = cast(h5py.Dataset, h5_tmp["gradients"])
                actual_grad_dim = gradients_ds.shape[1]

            self._layer_slices = []
            for name in gradient_layers:
                if name not in layer_map:
                    raise ValueError(
                        f"Layer '{name}' not found in model. "
                        f"Available layers: {list(layer_map.keys())}"
                    )
                info = layer_map[name]
                start = info["start"]
                end = min(info["end"], actual_grad_dim)
                if start >= actual_grad_dim:
                    print(
                        f"Warning: Layer '{name}' is beyond captured gradient "
                        f"dim ({actual_grad_dim:,}), skipping."
                    )
                    continue
                self._layer_slices.append((start, end))
            self._effective_gradient_dim = sum(
                end - start for start, end in self._layer_slices
            )
        else:
            self._effective_gradient_dim = None

        print(f"Loading data from {h5_path} (lazy loading for gradients)...")

        with h5py.File(h5_path, "r") as h5_file:
            gradients_ds = cast(h5py.Dataset, h5_file["gradients"])
            self.gradient_shape = gradients_ds.shape

            images_ds = cast(h5py.Dataset, h5_file["images"])
            actions_ds = cast(h5py.Dataset, h5_file["actions"])
            episode_ids_ds = cast(h5py.Dataset, h5_file["episode_ids"])
            done_ds = cast(h5py.Dataset, h5_file["done"])

            self.images = torch.tensor(images_ds[:], dtype=torch.float32) / 255.0
            self.action_target = "histogram" if actions_ds.ndim == 2 else "class"
            self.actions = torch.tensor(
                actions_ds[:],
                dtype=torch.float32 if self.action_target == "histogram" else torch.long,
            )
            self.episode_ids = torch.tensor(episode_ids_ds[:], dtype=torch.long)
            self.split_episode_ids = torch.tensor(
                h5_file["source_episode_ids"][:] if "source_episode_ids" in h5_file else episode_ids_ds[:],
                dtype=torch.long,
            )
            self.done = torch.tensor(done_ds[:], dtype=torch.bool)
            self.aggregation_size = int(h5_file.attrs.get("aggregation_size", 1))
            if self.aggregation_size > 1 and (
                not h5_file.attrs.get("aggregation_complete", False)
                or int(h5_file.attrs.get("completed_steps", -1)) != self.gradient_shape[0]
            ):
                raise ValueError("Aggregated dataset is incomplete; rerun preprocessing to a new file")
            stored_num_actions = h5_file.attrs.get("num_actions")
            stored_action_names = h5_file.attrs.get("action_names")

        inferred_num_actions = (
            self.actions.shape[1] if self.action_target == "histogram"
            else int(self.actions.max().item()) + 1 if len(self.actions) > 0 else 0
        )
        if (
            num_actions is not None
            and stored_num_actions is not None
            and num_actions != int(stored_num_actions)
        ):
            raise ValueError(
                f"Config requests {num_actions} actions, but {h5_path} declares "
                f"{int(stored_num_actions)}"
            )
        self.num_actions = int(
            num_actions
            if num_actions is not None
            else stored_num_actions
            if stored_num_actions is not None
            else inferred_num_actions
        )
        if inferred_num_actions > self.num_actions:
            raise ValueError(
                f"Dataset contains action index {inferred_num_actions - 1}, but "
                f"num_actions is {self.num_actions}"
            )

        if self.action_target == "histogram":
            if (
                self.actions.shape != (self.gradient_shape[0], self.num_actions)
                or not torch.isfinite(self.actions).all()
                or (self.actions < 0).any()
                or not torch.allclose(self.actions.sum(-1), torch.ones(len(self.actions)), atol=1e-5)
            ):
                raise ValueError("Action histograms must be finite, nonnegative and sum to one")
            print(f"  Targets: window-last RGB and action histograms; aggregation_size={self.aggregation_size}")

        self.action_names: Optional[List[str]] = None
        if stored_action_names is not None:
            if isinstance(stored_action_names, bytes):
                stored_action_names = stored_action_names.decode()
            try:
                parsed_names = json.loads(str(stored_action_names))
                if isinstance(parsed_names, list):
                    self.action_names = [str(name) for name in parsed_names]
            except json.JSONDecodeError:
                pass

        if self._effective_gradient_dim is None:
            if gradient_dim is not None and gradient_dim < self.gradient_shape[1]:
                self._effective_gradient_dim = gradient_dim
            else:
                self._effective_gradient_dim = self.gradient_shape[1]

        self.sequence_indices = self._build_sequence_indices()

        if max_sequences is not None and len(self.sequence_indices) > max_sequences:
            import random as _random

            rng = _random.Random(42)
            self.sequence_indices = rng.sample(self.sequence_indices, max_sequences)
            print(
                f"  Capped to {max_sequences} sequences "
                f"({max_sequences * sequence_length:,} total frames)"
            )

        if gradient_layers is not None:
            print(
                f"  Using {len(gradient_layers)} selected layers "
                f"({self._effective_gradient_dim:,} dims)"
            )
        elif gradient_dim is not None:
            print(f"  Gradient dim limited to {gradient_dim:,}")

        print(
            f"Dataset initialized: {len(self)} sequences, "
            f"seq_len={sequence_length}, stride={stride}"
        )

        # File handle is opened lazily per-worker for multiprocessing safety.
        self._h5_file = None
        self._gradients_dataset = None

    def _ensure_h5_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._gradients_dataset = cast(h5py.Dataset, self._h5_file["gradients"])

    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._gradients_dataset = None

    def _build_sequence_indices(self) -> List[Tuple[int, int]]:
        indices = []
        unique_episodes = self.episode_ids.unique()

        for ep_id in unique_episodes:
            ep_mask = self.episode_ids == ep_id
            ep_indices = torch.where(ep_mask)[0]

            if len(ep_indices) < self.sequence_length:
                continue

            for start in range(
                0, len(ep_indices) - self.sequence_length + 1, self.stride
            ):
                window_indices = ep_indices[start : start + self.sequence_length]

                if len(window_indices) == self.sequence_length:
                    diffs = window_indices[1:] - window_indices[:-1]
                    if torch.all(diffs == 1):
                        start_idx = window_indices[0].item()
                        end_idx = window_indices[-1].item()
                        indices.append((start_idx, end_idx + 1))

        return indices

    @property
    def effective_gradient_dim(self) -> int:
        assert self._effective_gradient_dim is not None
        return self._effective_gradient_dim

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx, end_idx = self.sequence_indices[idx]

        self._ensure_h5_open()
        assert self._gradients_dataset is not None

        gradients_np = self._gradients_dataset[start_idx:end_idx]

        if self._layer_slices is not None:
            slices = [gradients_np[:, s:e] for s, e in self._layer_slices]
            gradients_np = np.concatenate(slices, axis=1)
        elif (
            self.gradient_dim is not None and self.gradient_dim < gradients_np.shape[1]
        ):
            gradients_np = gradients_np[:, : self.gradient_dim]

        gradients = torch.tensor(gradients_np, dtype=torch.float32)
        images = self.images[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]

        return gradients, images, actions


def episode_split(dataset, val_fraction=0.05, seed=42, train_fraction=1.0):
    """Hold validation fixed and select nested fractions of training episodes."""
    import random

    groups = [int(dataset.split_episode_ids[start]) for start, _ in dataset.sequence_indices]
    episodes = sorted(set(groups))
    if len(episodes) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Episode splitting needs at least two eligible episodes and 0 < val_fraction < 1")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    random.Random(seed).shuffle(episodes)
    n_val = min(len(episodes) - 1, max(1, int(len(episodes) * val_fraction)))
    val_episodes = set(episodes[:n_val])
    n_train = max(1, int((len(episodes) - n_val) * train_fraction))
    train_episodes = set(episodes[n_val:n_val + n_train])
    train_indices = [i for i, ep in enumerate(groups) if ep in train_episodes]
    val_indices = [i for i, ep in enumerate(groups) if ep in val_episodes]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

from typing import List, Optional, Tuple, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from attacker.analyze_gradients import get_layer_map


class TemporalGradientDataset(Dataset):
    """
    Dataset that creates sequences of consecutive frames from episodes.

    Each sample is a window of T consecutive (gradient, image, action) tuples
    from the same episode.
    """

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 8,
        stride: int = 4,
        gradient_dim: Optional[int] = None,
        gradient_layers: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
    ):
        """
        Initialize dataset with lazy loading for gradients.

        Args:
            h5_path: Path to HDF5 file with gradients
            sequence_length: Number of consecutive frames per sample (T)
            stride: Step size between consecutive windows
            gradient_dim: Optional limit on gradient dimensions
            gradient_layers: Optional list of layer names to select. When set,
                only gradients from these layers are extracted and concatenated.
                Takes priority over gradient_dim.
            max_sequences: Optional cap on number of sequences. When set,
                a deterministic random subset is selected. Useful for
                equalizing total frames across sequence-length ablations.
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.gradient_dim = gradient_dim
        self.gradient_layers = gradient_layers
        self.h5_path = h5_path

        # Build layer index slices if gradient_layers is specified
        self._layer_slices: Optional[List[Tuple[int, int]]] = None
        if gradient_layers is not None:
            if gradient_dim is not None:
                print(
                    "Warning: Both gradient_layers and gradient_dim are set. "
                    "gradient_layers takes priority; gradient_dim is ignored."
                )
            layer_map, _total = get_layer_map()

            # Get actual gradient dim from H5 to clamp partially-captured layers
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
                # Clamp end to actual gradient dim in H5 (handles truncated captures)
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
            self._effective_gradient_dim = None  # computed after loading

        print(f"Loading data from {h5_path} (lazy loading for gradients)...")

        # Open HDF5 file temporarily to get metadata and load small arrays
        with h5py.File(h5_path, "r") as h5_file:
            # Keep gradient dataset shape for later
            gradients_ds = cast(h5py.Dataset, h5_file["gradients"])
            self.gradient_shape = gradients_ds.shape

            images_ds = cast(h5py.Dataset, h5_file["images"])
            actions_ds = cast(h5py.Dataset, h5_file["actions"])
            episode_ids_ds = cast(h5py.Dataset, h5_file["episode_ids"])
            done_ds = cast(h5py.Dataset, h5_file["done"])

            # Load smaller arrays into memory (images, actions, etc.)
            # Images: 23k * 3 * 84 * 84 * 1 byte = ~470 MB - fits in memory
            self.images = torch.tensor(images_ds[:], dtype=torch.float32) / 255.0
            self.actions = torch.tensor(actions_ds[:], dtype=torch.long)
            self.episode_ids = torch.tensor(episode_ids_ds[:], dtype=torch.long)
            self.done = torch.tensor(done_ds[:], dtype=torch.bool)

        # Compute effective gradient dim for non-layer-selection mode
        if self._effective_gradient_dim is None:
            if gradient_dim is not None and gradient_dim < self.gradient_shape[1]:
                self._effective_gradient_dim = gradient_dim
            else:
                self._effective_gradient_dim = self.gradient_shape[1]

        # Build sequence indices: (start_idx, end_idx) for valid windows
        self.sequence_indices = self._build_sequence_indices()

        # Cap to max_sequences if set (deterministic subsample)
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

        # File handle will be opened per-worker (for multiprocessing)
        self._h5_file = None
        self._gradients_dataset = None

    def _ensure_h5_open(self):
        """Open HDF5 file if not already open (per-worker lazy initialization)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._gradients_dataset = cast(h5py.Dataset, self._h5_file["gradients"])

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._gradients_dataset = None

    def _build_sequence_indices(self) -> List[Tuple[int, int]]:
        """Build list of valid sequence start/end indices."""
        indices = []

        # Group by episode
        unique_episodes = self.episode_ids.unique()

        for ep_id in unique_episodes:
            # Find all steps belonging to this episode
            ep_mask = self.episode_ids == ep_id
            ep_indices = torch.where(ep_mask)[0]

            if len(ep_indices) < self.sequence_length:
                continue  # Episode too short

            # Create windows with stride
            for start in range(
                0, len(ep_indices) - self.sequence_length + 1, self.stride
            ):
                # Get the actual global indices for this window
                window_indices = ep_indices[start : start + self.sequence_length]

                # Check if indices are consecutive (no gaps in episode)
                if len(window_indices) == self.sequence_length:
                    # Check consecutive
                    diffs = window_indices[1:] - window_indices[:-1]
                    if torch.all(diffs == 1):
                        start_idx = window_indices[0].item()
                        end_idx = window_indices[-1].item()
                        indices.append((start_idx, end_idx + 1))

        return indices

    @property
    def effective_gradient_dim(self) -> int:
        """The actual gradient dimension after layer selection or truncation."""
        assert self._effective_gradient_dim is not None
        return self._effective_gradient_dim

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence of T consecutive frames.

        Returns:
            gradients: (T, gradient_dim)
            images: (T, 3, H, W)
            actions: (T,)
        """
        start_idx, end_idx = self.sequence_indices[idx]

        # Ensure HDF5 file is open in this worker
        self._ensure_h5_open()
        assert self._gradients_dataset is not None

        # Lazy load gradients from HDF5 (only this slice, not entire array)
        gradients_np = self._gradients_dataset[start_idx:end_idx]

        if self._layer_slices is not None:
            # Layer-name selection: extract and concatenate selected slices
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

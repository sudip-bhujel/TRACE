from typing import List, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset


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
    ):
        """
        Initialize dataset with lazy loading for gradients.

        Args:
            h5_path: Path to HDF5 file with gradients
            sequence_length: Number of consecutive frames per sample (T)
            stride: Step size between consecutive windows
            gradient_dim: Optional limit on gradient dimensions
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.gradient_dim = gradient_dim
        self.h5_path = h5_path

        print(f"Loading data from {h5_path} (lazy loading for gradients)...")

        # Open HDF5 file temporarily to get metadata and load small arrays
        with h5py.File(h5_path, "r") as h5_file:
            # Keep gradient dataset shape for later
            self.gradient_shape = h5_file["gradients"].shape

            # Load smaller arrays into memory (images, actions, etc.)
            # Images: 23k * 3 * 84 * 84 * 1 byte = ~470 MB - fits in memory
            self.images = (
                torch.tensor(h5_file["images"][:], dtype=torch.float32) / 255.0
            )
            self.actions = torch.tensor(h5_file["actions"][:], dtype=torch.long)
            self.episode_ids = torch.tensor(h5_file["episode_ids"][:], dtype=torch.long)
            self.done = torch.tensor(h5_file["done"][:], dtype=torch.bool)

        # Build sequence indices: (start_idx, end_idx) for valid windows
        self.sequence_indices = self._build_sequence_indices()

        print(
            f"Dataset initialized: {len(self)} sequences, "
            f"seq_len={sequence_length}, stride={stride}"
        )
        if gradient_dim is not None:
            print(f"  Gradient dim limited to {gradient_dim:,}")

        # File handle will be opened per-worker (for multiprocessing)
        self._h5_file = None
        self._gradients_dataset = None

    def _ensure_h5_open(self):
        """Open HDF5 file if not already open (per-worker lazy initialization)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._gradients_dataset = self._h5_file["gradients"]

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

        # Lazy load gradients from HDF5 (only this slice, not entire array)
        gradients_np = self._gradients_dataset[start_idx:end_idx]
        if self.gradient_dim is not None and self.gradient_dim < gradients_np.shape[1]:
            gradients_np = gradients_np[:, : self.gradient_dim]
        gradients = torch.tensor(gradients_np, dtype=torch.float32)

        images = self.images[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]

        return gradients, images, actions

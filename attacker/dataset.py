"""
Dataset for loading gradient sequences and corresponding images.

This module provides a PyTorch Dataset that loads gradient sequences
from the collected features and pairs them with the original images.
"""

import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path to import from lsr
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_image(image_path: str, size: Tuple[int, int] = (120, 120)) -> torch.Tensor:
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("L")
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array)


def load_images_batch(
    image_paths: List[str], size: Tuple[int, int] = (120, 120), num_workers: int = 8
) -> List[torch.Tensor]:
    """Load multiple images in parallel."""

    def load_single(path):
        try:
            img = Image.open(path).convert("L")
            img = img.resize(size)
            return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_single, image_paths))

    return images


class GradientSequenceDataset(Dataset):
    """
    Dataset that yields subsequences of gradients and corresponding images.

    Each sample contains:
    - gradients: Tensor of shape (seq_len, 64, 512) - normalized gradients
    - images: Tensor of shape (seq_len, 120, 120) - grayscale images

    All data is preloaded into memory to avoid multiprocessing serialization issues.
    """

    def __init__(
        self,
        features_path: str,
        use_normalized_gradients: bool = True,
        preload_images: bool = True,
    ):
        """
        Initialize dataset from collected features.

        Args:
            features_path: Path to the pickle file containing collected features
            use_normalized_gradients: If True, use normalized gradients (weight/bias)
                                     If False, use weight gradients only
            preload_images: If True, preload all images into memory (faster but uses more RAM)
        """
        self.use_normalized_gradients = use_normalized_gradients
        self.preload_images = preload_images

        # Load the features data
        print(f"Loading features from {features_path}...")
        with open(features_path, "rb") as f:
            data = pickle.load(f)

        # Handle both FeatureCollector object and raw dict
        if hasattr(data, "metadata"):
            # It's a FeatureCollector object
            metadata = data.metadata
            scenes = data.scenes
            image_paths_dict = data.image_paths
            subsequence_length = data.subsequence_length
        else:
            # It's a raw dictionary
            metadata = data.get("metadata", {})
            scenes = data.get("scenes", {})
            image_paths_dict = data.get(
                "image_paths", {}
            )  # Separate dict for image paths
            subsequence_length = metadata.get("subsequence_length", 10)

        print(f"Loaded features from {features_path}")
        print(f"  - Total scenes: {metadata.get('num_scenes', len(scenes))}")
        print(f"  - Total samples: {metadata.get('total_samples', 'unknown')}")

        # Debug: Show structure of first scene
        if scenes:
            first_scene_id = list(scenes.keys())[0]
            first_scene = scenes[first_scene_id]
            print(
                f"\n  DEBUG - First scene '{first_scene_id}' keys: {list(first_scene.keys())}"
            )
            for key, value in first_scene.items():
                if isinstance(value, list):
                    print(f"    {key}: list of {len(value)} items")
                    if value and hasattr(value[0], "shape"):
                        print(f"      First item shape: {value[0].shape}")
                else:
                    print(f"    {key}: {type(value)}")

            # Check image_paths_dict
            if first_scene_id in image_paths_dict:
                print(
                    f"    image_paths (from separate dict): list of {len(image_paths_dict[first_scene_id])} items"
                )

        # Preload all subsequences into memory to avoid multiprocessing serialization issues
        self.subsequences = []
        self.subsequence_length = subsequence_length

        print("Preloading subsequences into memory...")
        print(f"  Using {16} threads for parallel image loading")
        scene_ids = list(scenes.keys())
        skipped_scenes = 0

        for scene_id in tqdm(scene_ids, desc="Loading scenes"):
            scene_data = scenes[scene_id]

            # Get gradients for this scene
            weight_gradients = scene_data.get("weight_gradients", [])
            normalized_gradients = scene_data.get("normalized_gradients", [])

            # Image paths are stored in a separate dictionary
            image_paths = image_paths_dict.get(scene_id, [])

            if not weight_gradients:
                skipped_scenes += 1
                continue

            if not image_paths:
                skipped_scenes += 1
                continue

            # Make sure we have matching lengths
            num_gradients = len(weight_gradients)
            num_images = len(image_paths)
            num_to_use = min(num_gradients, num_images)

            if num_to_use < subsequence_length:
                skipped_scenes += 1
                continue

            # Load ALL images for this scene in parallel (much faster!)
            if preload_images:
                all_images = load_images_batch(
                    image_paths[:num_to_use], size=(120, 120), num_workers=16
                )
                # Check for failed loads
                if any(img is None for img in all_images):
                    skipped_scenes += 1
                    continue

            # Split into subsequences
            num_subsequences = num_to_use // subsequence_length

            for subseq_idx in range(num_subsequences):
                start_idx = subseq_idx * subsequence_length
                end_idx = start_idx + subsequence_length

                # Extract gradients for this subsequence
                if use_normalized_gradients and normalized_gradients:
                    gradients = normalized_gradients[start_idx:end_idx]
                else:
                    gradients = weight_gradients[start_idx:end_idx]

                subseq_paths = image_paths[start_idx:end_idx]

                # Skip if we don't have enough data
                if (
                    len(gradients) != subsequence_length
                    or len(subseq_paths) != subsequence_length
                ):
                    continue

                # Stack gradients into tensor and detach/clone to avoid shared storage
                gradient_tensor = torch.stack(
                    [
                        (
                            g.detach().clone()
                            if isinstance(g, torch.Tensor)
                            else torch.tensor(g)
                        )
                        for g in gradients
                    ]
                )  # (seq_len, 64, 512)

                # Get preloaded images for this subsequence
                if preload_images:
                    subseq_images = all_images[start_idx:end_idx]
                    image_tensor = torch.stack(subseq_images)  # (seq_len, 120, 120)
                    self.subsequences.append(
                        {
                            "gradients": gradient_tensor,
                            "images": image_tensor,
                        }
                    )
                else:
                    self.subsequences.append(
                        {
                            "gradients": gradient_tensor,
                            "image_paths": subseq_paths,
                        }
                    )

        if skipped_scenes > 0:
            print(f"  Skipped {skipped_scenes} scenes with insufficient data")

        print(f"Loaded {len(self.subsequences)} subsequences from {features_path}")
        if use_normalized_gradients:
            print("  Using normalized gradients")
        else:
            print("  Using weight gradients only")
        print(f"  Subsequence length: {self.subsequence_length}")

    def __len__(self) -> int:
        return len(self.subsequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a subsequence.

        Returns:
            gradients: Tensor of shape (seq_len, 64, 512)
            images: Tensor of shape (seq_len, 120, 120)
        """
        subseq = self.subsequences[idx]
        gradients = subseq["gradients"]

        if self.preload_images:
            images = subseq["images"]
        else:
            # Load images on-the-fly
            images = torch.stack([load_image(p) for p in subseq["image_paths"]])

        return gradients, images


def create_dataloaders(
    features_path: str,
    batch_size: int = 4,
    train_split: float = 0.9,
    num_workers: int = 0,  # Default to 0 to avoid multiprocessing issues
    use_normalized_gradients: bool = True,
    preload_images: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        features_path: Path to collected features
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers (0 = main process only)
        use_normalized_gradients: Whether to use normalized gradients
        preload_images: Whether to preload all images into memory

    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = GradientSequenceDataset(
        features_path=features_path,
        use_normalized_gradients=use_normalized_gradients,
        preload_images=preload_images,
    )

    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\nDataset split:")
    print(f"  Training subsequences: {len(train_dataset)}")
    print(f"  Validation subsequences: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")

    # Create dataloaders
    # Use num_workers=0 by default to avoid multiprocessing serialization issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers == 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers == 0 else False,
    )

    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Gradient Sequence Dataset")
    print("=" * 80)

    # Load dataset
    features_path = "features/all_scenes.pkl"

    try:
        dataset = GradientSequenceDataset(
            features_path,
            use_normalized_gradients=True,
            preload_images=True,
        )

        print(f"\nDataset statistics:")
        print(f"  Total subsequences: {len(dataset)}")

        # Get first sample
        gradients, images = dataset[0]

        print(f"\nSample data shapes:")
        print(f"  Gradients: {gradients.shape}")
        print(f"    (sequence_length, 64, 512) - normalized gradients")
        print(f"  Images: {images.shape}")
        print(f"    (sequence_length, 120, 120)")

        # Create dataloaders
        print("\n" + "=" * 80)
        print("Creating DataLoaders")
        print("=" * 80)

        train_loader, val_loader = create_dataloaders(
            features_path,
            batch_size=4,
            train_split=0.8,
            num_workers=0,  # Use 0 for testing
            use_normalized_gradients=True,
            preload_images=True,
        )

        # Test a batch
        for batch_gradients, batch_images in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Gradients: {batch_gradients.shape}")
            print(f"    (batch_size, sequence_length, 64, 512)")
            print(f"  Images: {batch_images.shape}")
            print(f"    (batch_size, sequence_length, 120, 120)")
            break

        print("\n✅ Dataset and DataLoader working correctly!")

    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {features_path}")
        print("   Please run the gradient collection first:")
        print("   uv run python -m lsr.lsr")

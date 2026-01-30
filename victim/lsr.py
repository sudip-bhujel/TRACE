import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from victim.nnet import FeatureCollector, NNetwork


def load_image(image_path: Path, size=(150, 150)) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file
        size: Target size (height, width)

    Returns:
        Normalized tensor of shape (1, 1, height, width)
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    # Add channel dimension: (H, W) -> (1, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    return img_tensor


def load_batch_images(
    image_paths: List[Path], size=(150, 150), num_workers: int = 4
) -> torch.Tensor:
    """
    Load multiple images in parallel using ThreadPoolExecutor.

    Args:
        image_paths: List of paths to images
        size: Target size (height, width)
        num_workers: Number of parallel workers for loading

    Returns:
        Batched tensor of shape (batch_size, 1, height, width)
    """

    def load_single(path):
        img = Image.open(path).convert("L")
        img = img.resize(size)
        return np.array(img, dtype=np.float32) / 255.0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_single, image_paths))

    # Stack into batch tensor
    batch = np.stack(images, axis=0)  # (batch, H, W)
    batch_tensor = torch.from_numpy(batch).unsqueeze(1)  # (batch, 1, H, W)
    return batch_tensor


def get_scene_image_paths(
    data_dir: str = "data", num_images_per_scene: int = None
) -> Dict[str, List[Path]]:
    """
    Get all image paths organized by scene (FloorPlan).

    Args:
        data_dir: Root data directory
        num_images_per_scene: Maximum number of images to load per scene (None = all images)

    Returns:
        Dictionary mapping scene_id -> list of image paths
        Example: {"Kitchen/FloorPlan1": [img1.png, img2.png, ...]}
    """
    data_path = Path(data_dir)
    scenes = {}

    for room_dir in data_path.iterdir():
        if not room_dir.is_dir():
            continue

        room_name = room_dir.name

        # Iterate through FloorPlans
        for floor_plan_dir in room_dir.iterdir():
            if not floor_plan_dir.is_dir():
                continue

            floor_plan_name = floor_plan_dir.name
            scene_id = f"{room_name}/{floor_plan_name}"

            # Get all images in this FloorPlan
            if floor_plan_dir.exists():
                image_files = sorted(floor_plan_dir.glob("*.png")) + sorted(
                    floor_plan_dir.glob("*.jpg")
                )
                if image_files:
                    # Limit images per scene if specified
                    if num_images_per_scene is not None:
                        scenes[scene_id] = image_files[:num_images_per_scene]
                    else:
                        scenes[scene_id] = image_files

    return scenes


def collect_scene_features(
    model: NNetwork,
    scene_id: str,
    image_paths: List[Path],
) -> Dict:
    """
    Collect features for all images in a scene.

    Args:
        model: Trained or training model
        scene_id: Scene identifier
        image_paths: List of image paths for this scene

    Returns:
        Dictionary with all features for the scene
    """
    scene_features = {
        "gradients": [],
        # "hidden_activations": [],
        # "output_activations": [],
        # "cnn_features": [],
        "image_paths": [],
    }

    for img_path in image_paths:
        image = load_image(img_path).unsqueeze(0)

        output = model(image)

        pseudo_loss = output.sum()

        pseudo_loss.backward()

        features = model.get_last_layer_gradients()

        # Store features
        if features is not None:
            scene_features["gradients"].append(features.cpu())
        # if features["hidden_activations"] is not None:
        #     scene_features["hidden_activations"].append(
        #         features["hidden_activations"].cpu()
        #     )
        # if features["output_activations"] is not None:
        #     scene_features["output_activations"].append(
        #         features["output_activations"].cpu()
        #     )
        # if features["cnn_features"]:
        #     scene_features["cnn_features"].append(
        #         {k: v.cpu() for k, v in features["cnn_features"].items()}
        #     )
        scene_features["image_paths"].append(str(img_path))

        # Zero gradients for next iteration
        model.zero_grad()

    return scene_features


def collect_all_features(
    model: NNetwork,
    data_dir: str = "data",
    save_path: str = "features/all_scenes.pkl",
    subsequence_length: int = 10,
    device: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    num_capture_layers: int = 1,
    store_mode: str = "both",
):
    """
    Collect features for all scenes in the dataset with batch processing.

    Each scene is saved to a separate temp file immediately after processing,
    allowing resume if the job is killed. Final merge happens at the end.

    Args:
        model: Model to use for feature extraction
        data_dir: Root data directory
        save_path: Path to save collected features
        subsequence_length: Length of each subsequence (default: 10)
        device: Device to run the model on (cuda/cpu)
        batch_size: Number of images to process at once (default: 32)
        num_workers: Number of workers for parallel image loading (default: 4)
        num_capture_layers: Number of layers to capture gradients from (default: 1).
                           Use -1 to capture all layers.
        store_mode: What gradients to store:
                   - "normalized_only": Only store normalized gradients (saves ~50% memory)
                   - "both": Store both weight and normalized gradients

    Returns:
        FeatureCollector with all collected features
    """
    import pickle
    from pathlib import Path

    # Move model to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model = model.to(device_obj)
    print(f"Using device: {device_obj}")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    print(f"Capturing gradients from {num_capture_layers} layer(s)")
    print(f"Store mode: {store_mode}")

    model.eval()
    model.register_gradient_hook(num_layers=num_capture_layers)

    # Setup paths - use a temp directory for scene files
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = save_path_obj.parent / f".{save_path_obj.stem}_scenes"
    temp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = temp_dir / "checkpoint.pkl"

    # Load checkpoint (just scene IDs - very fast!)
    processed_scenes = set()
    if checkpoint_path.exists():
        try:
            print("\n📂 Found checkpoint, loading...")
            with open(checkpoint_path, "rb") as f:
                processed_scenes = pickle.load(f)
            print(f"   Resuming: {len(processed_scenes)} scenes already done")
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint: {e}, starting fresh...")
            processed_scenes = set()

    # Get all scenes
    scenes = get_scene_image_paths(data_dir)

    print(f"Found {len(scenes)} scenes")
    remaining_scenes = len(scenes) - len(processed_scenes)
    print(f"Scenes to process: {remaining_scenes}")
    print("Processing scenes...")

    # Process each scene
    for scene_idx, (scene_id, image_paths) in enumerate(scenes.items()):
        # Skip already processed scenes
        if scene_id in processed_scenes:
            continue

        num_images = len(image_paths)
        print(
            f"\n[{scene_idx + 1}/{len(scenes)}] Processing {scene_id} ({num_images} images)"
        )

        # Create a temporary collector for just this scene
        scene_collector = FeatureCollector(
            subsequence_length=subsequence_length, store_mode=store_mode
        )

        # Process in batches
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_paths = image_paths[batch_start:batch_end]

            # Load batch of images in parallel
            batch_images = load_batch_images(
                batch_paths, size=(150, 150), num_workers=num_workers
            ).to(device_obj)

            # Process each image in batch individually (required for per-image gradients)
            for i, img_path in enumerate(batch_paths):
                # Get single image from batch
                image = batch_images[i : i + 1]  # Keep batch dimension

                # Forward pass
                output = model(image)

                # Use output sum as pseudo-loss
                pseudo_loss = output.sum()

                # Backward pass to generate gradients
                pseudo_loss.backward()

                # Collect features with scene_id - use multi-layer gradients
                features = model.get_layer_gradients()
                scene_collector.add_features(
                    features, scene_id=scene_id, image_path=str(img_path)
                )

                # Zero gradients
                model.zero_grad()

            # Progress update per batch
            print(f"  Processed {batch_end}/{num_images} images")

        # Save this scene to its own file (fast - just one scene at a time)
        scene_file = temp_dir / f"{scene_id.replace('/', '_')}.pkl"
        scene_data = {
            "scene_id": scene_id,
            "scene_data": scene_collector.scenes[scene_id],
            "image_paths": scene_collector.image_paths[scene_id],
        }
        print(f"  💾 Saving scene...")
        with open(scene_file, "wb") as f:
            pickle.dump(scene_data, f)

        # Update checkpoint (very fast - just scene IDs)
        processed_scenes.add(scene_id)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(processed_scenes, f)

        # Free memory
        del scene_collector
        del scene_data

    # Merge all scene files into final output
    print(f"\n📦 Merging {len(processed_scenes)} scenes into {save_path}...")
    collector = FeatureCollector(
        subsequence_length=subsequence_length, store_mode=store_mode
    )

    for scene_file in temp_dir.glob("*.pkl"):
        if scene_file.name == "checkpoint.pkl":
            continue
        with open(scene_file, "rb") as f:
            scene_data = pickle.load(f)
        collector.scenes[scene_data["scene_id"]] = scene_data["scene_data"]
        collector.image_paths[scene_data["scene_id"]] = scene_data["image_paths"]
        collector.metadata["num_scenes"] += 1
        collector.metadata["total_samples"] += len(scene_data["image_paths"])

    # Save final result
    collector.save(save_path)

    # Clean up temp directory
    import shutil

    shutil.rmtree(temp_dir)
    print("   Removed temporary scene files")

    print("\n✅ Feature collection complete!")
    print(f"   Total scenes: {collector.metadata['num_scenes']}")
    print(f"   Total images: {collector.metadata['total_samples']}")

    return collector


def create_scene_dataloader(collector: FeatureCollector, batch_size: int = 1):
    """
    Create a dataloader that yields sequences of features for each scene.

    This is useful for training a transformer that processes sequences of
    gradients from related images.

    Args:
        collector: FeatureCollector with loaded features
        batch_size: Number of scenes per batch

    Yields:
        Tuple of (scene_id, scene_features, scene_image_paths)
    """
    scene_ids = collector.get_all_scene_ids()

    for i in range(0, len(scene_ids), batch_size):
        batch_scene_ids = scene_ids[i : i + batch_size]

        for scene_id in batch_scene_ids:
            scene_features = collector.get_scene_features(scene_id)
            scene_image_paths = collector.get_scene_image_paths(scene_id)

            yield scene_id, scene_features, scene_image_paths


def create_subsequence_dataloader(collector: FeatureCollector, batch_size: int = 1):
    """
    Create a dataloader that yields subsequences of features for each scene.

    Each scene is split into multiple subsequences of fixed length (e.g., 10 images each).
    This maintains the temporal sequence order while creating smaller, manageable chunks.

    Args:
        collector: FeatureCollector with loaded features
        batch_size: Number of subsequences per batch

    Yields:
        Tuple of (scene_id, subsequence_index, subsequence_data)
        where subsequence_data contains:
            - weight_gradients: List of weight gradient tensors
            - bias_gradients: List of bias gradient tensors
            - image_paths: List of image paths
            - start_index, end_index: Indices in the original sequence
    """
    scene_ids = collector.get_all_scene_ids()

    all_subsequences = []
    for scene_id in scene_ids:
        subsequences = collector.get_scene_subsequences(scene_id)
        if subsequences:
            for subseq in subsequences:
                all_subsequences.append((scene_id, subseq))

    # Yield in batches
    for i in range(0, len(all_subsequences), batch_size):
        batch = all_subsequences[i : i + batch_size]
        for scene_id, subseq_data in batch:
            yield scene_id, subseq_data["subsequence_index"], subseq_data


def main():
    parser = argparse.ArgumentParser(
        description="Collect features from images using NNetwork"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/data120x120",
        help="Root directory containing scene images",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="features/all_scenes.pkl",
        help="Path to save collected features",
    )
    parser.add_argument(
        "--num_images_per_scene",
        type=int,
        default=None,
        help="Maximum number of images to process per scene (default: all)",
    )
    parser.add_argument(
        "--subsequence_length",
        type=int,
        default=10,
        help="Length of each subsequence (default: 10)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=150,
        help="Height of input images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=150,
        help="Width of input images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to load per batch (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel image loading (default: 4)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to load (default: None, uses untrained model)",
    )
    parser.add_argument(
        "--num_capture_layers",
        type=int,
        default=1,
        help="Number of layers to capture gradients from (default: 1, use -1 for all layers)",
    )
    parser.add_argument(
        "--store_mode",
        type=str,
        default="normalized_only",
        choices=["normalized_only", "both"],
        help="What gradients to store: 'normalized_only' (saves ~50%% memory) or 'both' (default: normalized_only)",
    )
    args = parser.parse_args()

    scenes = get_scene_image_paths(args.data_dir, args.num_images_per_scene)

    if scenes:
        model = NNetwork(output_size=64)

        # Load checkpoint if provided
        if args.model_checkpoint is not None:
            checkpoint_path = Path(args.model_checkpoint)
            if checkpoint_path.exists():
                model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                print(f"✅ Loaded model checkpoint from {checkpoint_path}")
            else:
                print(
                    f"⚠️  Checkpoint not found: {checkpoint_path}, using untrained model"
                )

        collector = collect_all_features(
            model=model,
            data_dir=args.data_dir,
            save_path=args.save_path,
            subsequence_length=args.subsequence_length,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_capture_layers=args.num_capture_layers,
            store_mode=args.store_mode,
        )

        # Show full scene statistics
        print("\n" + "=" * 60)
        print("Full Scene Statistics:")
        print("=" * 60)
        for scene_id, features, image_paths in create_scene_dataloader(collector):
            if features is None or image_paths is None:
                continue
            print(f"\n   Scene: {scene_id}")
            print(f"     Total images: {len(image_paths)}")
            if features.get("weight_gradients"):
                print(
                    f"     Weight gradients: {len(features['weight_gradients'])} x {features['weight_gradients'][0].shape}"
                )
            if features.get("bias_gradients"):
                print(
                    f"     Bias gradients: {len(features['bias_gradients'])} x {features['bias_gradients'][0].shape}"
                )
            break  # Just show first scene

        # Show subsequence statistics
        print("\n" + "=" * 60)
        print(f"Subsequence Statistics (length={args.subsequence_length}):")
        print("=" * 60)

        count = 0
        for scene_id, subseq_idx, subseq_data in create_subsequence_dataloader(
            collector
        ):
            print(f"\n   Scene: {scene_id} | Subsequence: {subseq_idx}")
            print(
                f"     Images: {subseq_data['start_index']}-{subseq_data['end_index'] - 1} (total: {len(subseq_data['image_paths'])})"
            )
            print(
                f"     Weight gradients: {len(subseq_data['weight_gradients'])} x {subseq_data['weight_gradients'][0].shape}"
            )
            print(
                f"     Bias gradients: {len(subseq_data['bias_gradients'])} x {subseq_data['bias_gradients'][0].shape}"
            )
            if subseq_data.get("normalized_gradients"):
                print(
                    f"     Normalized gradients: {len(subseq_data['normalized_gradients'])} x {subseq_data['normalized_gradients'][0].shape}"
                )

            count += 1
            if count >= 3:  # Show first 3 subsequences
                break

        # Show total count
        total_subsequences = sum(1 for _ in create_subsequence_dataloader(collector))
        print(f"\n   Total subsequences across all scenes: {total_subsequences}")


if __name__ == "__main__":
    main()

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class NNetwork(nn.Module):
    """
    Improved Multi-Layer Perceptron for image processing with CNN layers.

    This network processes 150x150 images with CNN layers to preserve spatial structure,
    then uses MLP layers for final processing. It captures gradients, activations, and
    intermediate features for comprehensive gradient-based inversion.

    Args:
        input_size (int): Size of flattened input image (default: 14400 for 120x120)
        hidden_size (int): Size of hidden layer (default: 512)
        output_size (int): Size of output (default: 64 for richer gradients)
    """

    def __init__(
        self,
        input_size: int = 120 * 120,  # 14,400 for 120x120 image
        hidden_size: int = 512,
        output_size: int = 64,
    ):
        super(NNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 75, 75)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch, 32, 37, 37)
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # (batch, 64, 19, 19)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch, 64, 9, 9)
        )

        self.cnn_output_size = 64 * 9 * 9  # 5,184

        self.fc1 = nn.Linear(self.cnn_output_size, hidden_size)

        self.activation = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, output_size)

        # Storage for gradients - now supports multiple layers
        self.layer_weight_gradients: Dict[str, torch.Tensor] = {}
        self.layer_bias_gradients: Dict[str, torch.Tensor] = {}
        self.gradient_hook_handles: List[Any] = []
        self.num_capture_layers: int = 1  # Default: capture only last layer

        # Get ordered list of layers with weights (for indexing from last)
        self._weight_layers = self._get_weight_layers()

        # Storage for intermediate features
        self.capture_features = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with feature capture.

        Args:
            x: Input tensor of shape (batch_size, 1, 120, 120) or (batch_size, 14400)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = self.encoder(x)  # Pass through CNN layers
        x = x.view(x.size(0), -1)  # Flatten for MLP

        # MLP Layer 1
        x = self.fc1(x)
        x = self.activation(x)

        # MLP Layer 2 (last layer)
        x = self.fc2(x)

        return x

    def _get_weight_layers(self) -> List[tuple]:
        """
        Get ordered list of layers that have weights.
        Returns list of (layer_name, layer_module) tuples in forward order.
        """
        weight_layers = []
        for name, module in self.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                # Skip the top-level module and batch norm layers (they have weight but not trainable in this context)
                if name and not isinstance(module, nn.BatchNorm2d):
                    weight_layers.append((name, module))
        return weight_layers

    def _create_weight_hook(self, layer_name: str):
        """
        Create a hook function for capturing weight gradients for a specific layer.
        """

        def hook(grad: torch.Tensor):
            self.layer_weight_gradients[layer_name] = grad.detach().clone()

        return hook

    def _create_bias_hook(self, layer_name: str):
        """
        Create a hook function for capturing bias gradients for a specific layer.
        """

        def hook(grad: torch.Tensor):
            self.layer_bias_gradients[layer_name] = grad.detach().clone()

        return hook

    def register_gradient_hook(self, num_layers: int = 1):
        """
        Register hooks to capture gradients from the specified number of layers.
        Layers are counted from the last layer backwards.

        Args:
            num_layers: Number of layers to capture gradients from (counting from last).
                       Use -1 to capture all layers with weights.
        """
        # Remove existing hooks
        self.remove_gradient_hook()

        # Store the number of layers to capture
        self.num_capture_layers = num_layers

        # Refresh layer list (in case model structure changed)
        self._weight_layers = self._get_weight_layers()

        # Determine which layers to capture
        if num_layers == -1:
            layers_to_capture = self._weight_layers
        else:
            # Take the last num_layers
            layers_to_capture = self._weight_layers[-num_layers:]

        # Register hooks for each layer
        for layer_name, module in layers_to_capture:
            # Register hook on weight
            handle = module.weight.register_hook(self._create_weight_hook(layer_name))
            self.gradient_hook_handles.append(handle)

            # Register hook on bias if it exists
            if hasattr(module, "bias") and module.bias is not None:
                handle = module.bias.register_hook(self._create_bias_hook(layer_name))
                self.gradient_hook_handles.append(handle)

    def remove_gradient_hook(self):
        """Remove all gradient hooks if they exist."""
        for handle in self.gradient_hook_handles:
            handle.remove()
        self.gradient_hook_handles = []
        self.layer_weight_gradients = {}
        self.layer_bias_gradients = {}

    def get_layer_gradients(
        self,
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Get the captured gradients from all registered layers.

        Returns:
            Dictionary mapping layer_name -> {'weight': tensor, 'bias': tensor or None},
            or None if no gradients captured yet.

            Example:
                {
                    'fc2': {'weight': tensor, 'bias': tensor},
                    'fc1': {'weight': tensor, 'bias': tensor},
                    'encoder.4': {'weight': tensor, 'bias': None},
                }
        """
        if not self.layer_weight_gradients:
            return None

        result = {}
        for layer_name in self.layer_weight_gradients:
            result[layer_name] = {
                "weight": self.layer_weight_gradients[layer_name],
                "bias": self.layer_bias_gradients.get(layer_name),
            }
        return result

    def get_last_layer_gradients(self) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """
        Get the captured gradients from the last layer only.
        This is a convenience method for backward compatibility.

        Returns:
            Dictionary containing 'weight' and 'bias' gradients, or None if not captured yet
        """
        all_gradients = self.get_layer_gradients()
        if all_gradients is None:
            return None

        # Get the last layer (fc2 by default)
        if "fc2" in all_gradients:
            return all_gradients["fc2"]

        # If fc2 not found, return the last captured layer
        if all_gradients:
            last_layer_name = list(all_gradients.keys())[-1]
            return all_gradients[last_layer_name]

        return None

    def enable_feature_capture(self):
        """Enable capturing of activations and intermediate features."""
        self.capture_features = True

    def disable_feature_capture(self):
        """Disable capturing of activations and intermediate features."""
        self.capture_features = False

    def __del__(self):
        """Cleanup hook on deletion."""
        self.remove_gradient_hook()


class FeatureCollector:
    """
    Enhanced collector for gradients.

    This class collects features grouped by scene (FloorPlan) to maintain
    relationships between images from the same scene with different actions.

    Attributes:
        scenes: Dict mapping scene_id -> list of features for that scene
        image_paths: Dict mapping scene_id -> list of image paths
        metadata: General metadata about the collection
    """

    def __init__(self, subsequence_length: int = 10, store_mode: str = "both"):
        """
        Initialize the FeatureCollector.

        Args:
            subsequence_length: Length of each subsequence
            store_mode: What gradients to store:
                       - "normalized_only": Only store normalized gradients (saves ~50% memory)
                       - "both": Store both weight and normalized gradients
        """
        # Group features by scene (FloorPlan)
        self.scenes: Dict[str, Dict[str, List]] = {}
        # Store image paths for each scene
        self.image_paths: Dict[str, List[str]] = {}
        self.subsequence_length = subsequence_length
        self.store_mode = store_mode
        self.metadata: Dict = {
            "image_size": (120, 120),
            "num_scenes": 0,
            "total_samples": 0,
            "subsequence_length": subsequence_length,
            "gradient_format": "multi_layer",  # New format: gradients stored as {layer_name: tensor, ...}
            "store_mode": store_mode,
        }

    def add_features(
        self,
        features: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        scene_id: str,
        image_path: Optional[str] = None,
    ):
        """
        Add features from a forward/backward pass, grouped by scene.

        Args:
            features: Dictionary from model.get_layer_gradients() with structure:
                     {layer_name: {'weight': tensor, 'bias': tensor}, ...}
                     Example: {'fc2': {'weight': tensor, 'bias': tensor}, 'fc1': {...}}
            scene_id: Scene identifier (e.g., "Kitchen/FloorPlan1")
            image_path: Optional path to the source image for reference
        """
        # Initialize scene if not exists
        if scene_id not in self.scenes:
            self.scenes[scene_id] = {
                "weight_gradients": [],
                "bias_gradients": [],
                "normalized_gradients": [],
            }
            self.image_paths[scene_id] = []
            self.metadata["num_scenes"] += 1

        # Store on CPU to save GPU memory
        if features is not None:
            # Multi-layer format: features is {layer_name: {'weight': tensor, 'bias': tensor}, ...}
            # Store as dict per sample: {'fc2': weight_tensor, 'fc1': weight_tensor, ...}
            weight_grads_dict = {}
            bias_grads_dict = {}
            normalized_grads_dict = {}

            for layer_name, layer_grads in features.items():
                weight_grad = layer_grads.get("weight")
                bias_grad = layer_grads.get("bias")

                # Store weight gradients only if mode is "both"
                if self.store_mode == "both":
                    if weight_grad is not None:
                        weight_grads_dict[layer_name] = weight_grad.cpu()
                    if bias_grad is not None:
                        bias_grads_dict[layer_name] = bias_grad.cpu()

                # Compute normalized gradient: weight_grad / (bias_grad + 1e-8)
                if weight_grad is not None and bias_grad is not None:
                    bias_expanded = bias_grad.unsqueeze(1)
                    normalized_grad = weight_grad / (bias_expanded + 1e-8)
                    normalized_grads_dict[layer_name] = normalized_grad.cpu()

            if weight_grads_dict:
                self.scenes[scene_id]["weight_gradients"].append(weight_grads_dict)
            if bias_grads_dict:
                self.scenes[scene_id]["bias_gradients"].append(bias_grads_dict)
            if normalized_grads_dict:
                self.scenes[scene_id]["normalized_gradients"].append(
                    normalized_grads_dict
                )

        # Store image path if provided
        if image_path is not None:
            self.image_paths[scene_id].append(image_path)

        self.metadata["total_samples"] += 1

    def get_scene_features(self, scene_id: str) -> Optional[Dict[str, List]]:
        """
        Get all features for a specific scene.

        Args:
            scene_id: Scene identifier

        Returns:
            Dictionary with lists of features for that scene, or None if scene not found
        """
        return self.scenes.get(scene_id)

    def get_scene_image_paths(self, scene_id: str) -> Optional[List[str]]:
        """
        Get all image paths for a specific scene.

        Args:
            scene_id: Scene identifier

        Returns:
            List of image paths for that scene, or None if scene not found
        """
        return self.image_paths.get(scene_id)

    def get_all_scene_ids(self) -> List[str]:
        """Get list of all scene identifiers."""
        return list(self.scenes.keys())

    def get_scene_sequence_length(self, scene_id: str) -> int:
        """
        Get the number of samples in a scene sequence.

        Args:
            scene_id: Scene identifier

        Returns:
            Number of images/features in that scene
        """
        if scene_id in self.scenes:
            return len(self.scenes[scene_id]["weight_gradients"])
        return 0

    def get_scene_subsequences(self, scene_id: str) -> Optional[List[Dict[str, List]]]:
        """
        Get scene features grouped into subsequences of fixed length.

        Args:
            scene_id: Scene identifier

        Returns:
            List of dictionaries, each containing a subsequence of features,
            or None if scene not found
        """
        if scene_id not in self.scenes:
            return None

        weight_grads = self.scenes[scene_id]["weight_gradients"]
        bias_grads = self.scenes[scene_id]["bias_gradients"]
        normalized_grads = self.scenes[scene_id]["normalized_gradients"]
        image_paths = self.image_paths[scene_id]

        subsequences = []
        num_samples = len(weight_grads)

        # Group into subsequences
        for i in range(0, num_samples, self.subsequence_length):
            end_idx = min(i + self.subsequence_length, num_samples)
            subsequence = {
                "weight_gradients": weight_grads[i:end_idx],
                "bias_gradients": bias_grads[i:end_idx],
                "normalized_gradients": normalized_grads[i:end_idx],
                "image_paths": image_paths[i:end_idx],
                "subsequence_index": i // self.subsequence_length,
                "start_index": i,
                "end_index": end_idx,
            }
            subsequences.append(subsequence)

        return subsequences

    def save(self, filepath: str):
        """
        Save all collected features to a file.

        Args:
            filepath: Path to save the features (will use pickle format)
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "scenes": self.scenes,
            "image_paths": self.image_paths,
            "metadata": self.metadata,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved features from {self.metadata['num_scenes']} scenes to {filepath}")
        print(f"  - Total scenes: {self.metadata['num_scenes']}")
        print(f"  - Total samples: {self.metadata['total_samples']}")

        # Show sample of scene statistics
        if self.scenes:
            print(f"\n  Scene statistics (first 3):")
            for i, scene_id in enumerate(list(self.scenes.keys())[:3]):
                seq_len = self.get_scene_sequence_length(scene_id)
                print(f"    {scene_id}: {seq_len} images")

    def load(self, filepath: str):
        """
        Load features from a file.

        Args:
            filepath: Path to load the features from
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.scenes = data.get("scenes", {})
        self.image_paths = data.get("image_paths", {})
        self.metadata = data.get("metadata", {})

        print(f"Loaded features from {filepath}")
        print(f"  - Total scenes: {self.metadata.get('num_scenes', 0)}")
        print(f"  - Total samples: {self.metadata.get('total_samples', 0)}")

    def clear(self):
        """Clear all collected features."""
        self.scenes = {}
        self.image_paths = {}
        self.metadata["num_scenes"] = 0
        self.metadata["total_samples"] = 0


class GradientCollector:
    """
    Utility class to collect and save gradients during training.

    This class helps manage the collection of gradients from multiple
    forward/backward passes and save them for later use.
    """

    def __init__(self):
        self.gradients: List[torch.Tensor] = []
        self.metadata: Dict = {"image_size": (150, 150), "num_samples": 0}

    def add_gradient(self, gradient: torch.Tensor):
        """
        Add a gradient tensor to the collection.

        Args:
            gradient: Gradient tensor to store
        """
        # Store on CPU to save GPU memory
        self.gradients.append(gradient.cpu())
        self.metadata["num_samples"] += 1

    def save(self, filepath: str):
        """
        Save collected gradients to a file.

        Args:
            filepath: Path to save the gradients (will use pickle format)
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"gradients": self.gradients, "metadata": self.metadata}

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {len(self.gradients)} gradient samples to {filepath}")

    def load(self, filepath: str):
        """
        Load gradients from a file.

        Args:
            filepath: Path to load the gradients from
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.gradients = data["gradients"]
        self.metadata = data["metadata"]

        print(f"Loaded {len(self.gradients)} gradient samples from {filepath}")

    def clear(self):
        """Clear all collected gradients."""
        self.gradients = []
        self.metadata["num_samples"] = 0

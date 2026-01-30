"""
Temporal Gradient Inversion Model

This module implements a sequence-to-sequence model that takes consecutive
gradients and reconstructs consecutive images, leveraging temporal dependencies
through a causal transformer architecture.

Architecture:
    [g_1, g_2, ..., g_T] → Encoder → Temporal Transformer → Decoder → [img_1, ..., img_T]

Key features:
    - Causal (autoregressive) attention: frame t only sees frames 1..t
    - Shared encoder/decoder across timesteps
    - Per-frame reconstruction with temporal context
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

# =============================================================================
# Dataset
# =============================================================================


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
        Initialize dataset.

        Args:
            h5_path: Path to HDF5 file with gradients
            sequence_length: Number of consecutive frames per sample (T)
            stride: Step size between consecutive windows
            gradient_dim: Optional limit on gradient dimensions
        """
        self.sequence_length = sequence_length
        self.stride = stride

        print(f"Loading data from {h5_path}...")

        with h5py.File(h5_path, "r") as f:
            self.gradients = torch.tensor(f["gradients"][:], dtype=torch.float32)
            self.images = torch.tensor(f["images"][:], dtype=torch.float32) / 255.0
            self.actions = torch.tensor(f["actions"][:], dtype=torch.long)
            self.episode_ids = torch.tensor(f["episode_ids"][:], dtype=torch.long)
            self.done = torch.tensor(f["done"][:], dtype=torch.bool)

        # Limit gradient dimensions if specified
        if gradient_dim is not None and gradient_dim < self.gradients.shape[1]:
            self.gradients = self.gradients[:, :gradient_dim]

        # Build sequence indices: (start_idx, end_idx) for valid windows
        self.sequence_indices = self._build_sequence_indices()

        print(f"  Total steps: {len(self.gradients)}")
        print(f"  Unique episodes: {len(self.episode_ids.unique())}")
        print(
            f"  Valid sequences (T={sequence_length}, stride={stride}): {len(self.sequence_indices)}"
        )
        print(f"  Gradient shape: {self.gradients.shape}")
        print(f"  Image shape: {self.images.shape}")

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
                start_idx = ep_indices[start].item()
                end_idx = ep_indices[start + self.sequence_length - 1].item()

                # Verify all indices are consecutive and same episode
                window = ep_indices[start : start + self.sequence_length]
                if len(window) == self.sequence_length:
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

        gradients = self.gradients[start_idx:end_idx]
        images = self.images[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]

        return gradients, images, actions


# =============================================================================
# Model Components
# =============================================================================


class GradientEncoder(nn.Module):
    """Encode a single gradient vector into a latent representation."""

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        hidden_dims: List[int] = [4096, 2048, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = gradient_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, gradient_dim) or (B, gradient_dim)
        Returns:
            (B, T, latent_dim) or (B, latent_dim)
        """
        return self.encoder(x)


class CausalTemporalTransformer(nn.Module):
    """
    Transformer with causal (autoregressive) attention.

    Each position can only attend to previous positions.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, latent_dim) * 0.02
        )

        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(latent_dim)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, latent_dim)
        Returns:
            (B, T, latent_dim)
        """
        B, T, D = x.shape

        # Add positional embeddings
        x = x + self.pos_embedding[:, :T, :]

        # Generate causal mask
        causal_mask = self.generate_causal_mask(T, x.device)

        # Apply transformer
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)

        return x


class ImageDecoder(nn.Module):
    """Decode latent representation to image and action."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
    ):
        super().__init__()

        self.image_size = image_size

        # Compute sizes for transposed convolution
        # 84 -> 21 -> 42 -> 84
        self.init_size = image_size // 4  # 21

        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # Transposed convolutions for image generation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 21 -> 42
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 42 -> 84
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),  # 84 -> 84
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Action prediction head
        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, latent_dim) or (B, latent_dim)
        Returns:
            images: (B, T, 3, H, W) or (B, 3, H, W)
            actions: (B, T, num_actions) or (B, num_actions)
        """
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Generate image
        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)
        images = self.decoder(h)

        # Predict action
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class TemporalGradientInversion(nn.Module):
    """
    Full temporal gradient inversion model.

    Takes a sequence of gradients and produces a sequence of images,
    using causal attention for temporal modeling.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        encoder_hidden_dims: List[int] = [4096, 2048, 1024],
        dropout: float = 0.1,
        image_size: int = 84,
    ):
        super().__init__()

        self.gradient_encoder = GradientEncoder(
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )

        self.temporal_transformer = CausalTemporalTransformer(
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
        )

    def forward(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            gradients: (B, T, gradient_dim)
        Returns:
            images: (B, T, 3, H, W)
            actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
        """
        # Encode each gradient
        latents = self.gradient_encoder(gradients)  # (B, T, latent_dim)

        # Apply temporal transformer (causal attention)
        latents = self.temporal_transformer(latents)  # (B, T, latent_dim)

        # Decode to images and actions
        images, actions = self.image_decoder(
            latents
        )  # (B, T, 3, H, W), (B, T, num_actions)

        return images, actions, latents


# =============================================================================
# Loss Functions
# =============================================================================


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss for sharper reconstructions.

    Computes feature-space loss using VGG16 pretrained on ImageNet.
    """

    def __init__(
        self,
        layers: List[str] = None,
        resize: bool = True,
    ):
        super().__init__()

        if layers is None:
            layers = ["relu2_2", "relu3_3"]

        self.resize = resize

        # Load pretrained VGG16 with SSL workaround for HPC
        import os
        import ssl

        old_ssl_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        old_ca_bundle = os.environ.get("CURL_CA_BUNDLE", "")
        os.environ["CURL_CA_BUNDLE"] = ""

        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Warning: Failed to download VGG weights: {e}")
            vgg = models.vgg16(weights=None)
        finally:
            ssl._create_default_https_context = old_ssl_context
            os.environ["CURL_CA_BUNDLE"] = old_ca_bundle

        layer_map = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu4_1": 18,
            "relu4_2": 20,
            "relu4_3": 22,
            "relu5_1": 25,
            "relu5_2": 27,
            "relu5_3": 29,
        }

        max_idx = max(layer_map[layer] for layer in layers) + 1
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:max_idx])

        for param in self.vgg_features.parameters():
            param.requires_grad = False

        self.layer_indices = {layer: layer_map[layer] for layer in layers}
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.normalize(x)
        features = {}
        for name, module in self.vgg_features._modules.items():
            x = module(x)
            idx = int(name)
            for layer_name, layer_idx in self.layer_indices.items():
                if idx == layer_idx:
                    features[layer_name] = x
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        loss = 0.0
        for layer_name in self.layer_indices.keys():
            loss += F.mse_loss(pred_features[layer_name], target_features[layer_name])
        return loss / len(self.layer_indices)


class TemporalCombinedLoss(nn.Module):
    """
    Combined loss for temporal gradient inversion.

    Computes per-frame losses and averages over the sequence.
    Optionally includes temporal smoothness loss.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        action_weight: float = 0.1,
        temporal_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        perceptual_layers: List[str] = None,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.action_weight = action_weight
        self.temporal_weight = temporal_weight
        self.perceptual_weight = perceptual_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        # VGG perceptual loss (only create if weight > 0)
        if perceptual_weight > 0:
            self.perceptual_loss = VGGPerceptualLoss(layers=perceptual_layers)
        else:
            self.perceptual_loss = None

    def forward(
        self,
        pred_images: torch.Tensor,  # (B, T, 3, H, W)
        target_images: torch.Tensor,  # (B, T, 3, H, W)
        pred_actions: torch.Tensor,  # (B, T, num_actions)
        target_actions: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        B, T = pred_images.shape[:2]

        # Flatten for loss computation
        pred_img_flat = pred_images.reshape(-1, *pred_images.shape[2:])
        target_img_flat = target_images.reshape(-1, *target_images.shape[2:])
        pred_act_flat = pred_actions.reshape(-1, pred_actions.shape[-1])
        target_act_flat = target_actions.reshape(-1)

        mse = self.mse_loss(pred_img_flat, target_img_flat)
        l1 = self.l1_loss(pred_img_flat, target_img_flat)
        action_loss = self.ce_loss(pred_act_flat, target_act_flat)

        # Perceptual loss (VGG features)
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred_img_flat, target_img_flat)
        else:
            perceptual = torch.tensor(0.0, device=pred_images.device)

        # Temporal smoothness (penalize large changes between consecutive frames)
        if self.temporal_weight > 0 and T > 1:
            temporal_diff = pred_images[:, 1:] - pred_images[:, :-1]
            temporal_loss = temporal_diff.pow(2).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=pred_images.device)

        # Total loss
        total = (
            self.mse_weight * mse
            + self.l1_weight * l1
            + self.action_weight * action_loss
            + self.perceptual_weight * perceptual
            + self.temporal_weight * temporal_loss
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse.item(),
            "l1": l1.item(),
            "action": action_loss.item(),
            "perceptual": perceptual.item()
            if torch.is_tensor(perceptual)
            else perceptual,
            "temporal": temporal_loss.item()
            if torch.is_tensor(temporal_loss)
            else temporal_loss,
        }

        return total, loss_dict


# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler = None,
    accumulation_steps: int = 1,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_losses = defaultdict(float)
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for gradients, images, actions in pbar:
        gradients = gradients.to(device)
        images = images.to(device)
        actions = actions.to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            pred_images, pred_actions, _ = model(gradients)
            loss, loss_dict = criterion(pred_images, images, pred_actions, actions)
            loss = loss / accumulation_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            if (pbar.n + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (pbar.n + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Accumulate metrics
        for k, v in loss_dict.items():
            total_losses[k] += v

        # Accuracy (flatten over batch and time)
        pred_labels = pred_actions.argmax(dim=-1).flatten()
        correct += (pred_labels == actions.flatten()).sum().item()
        total += actions.numel()

        pbar.set_postfix(
            loss=f"{loss_dict['total']:.4f}", acc=f"{100 * correct / total:.1f}%"
        )

    # Average
    for k in total_losses:
        total_losses[k] /= len(dataloader)
    total_losses["accuracy"] = 100 * correct / total

    return dict(total_losses)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_losses = defaultdict(float)
    correct = 0
    total = 0

    with torch.no_grad():
        for gradients, images, actions in dataloader:
            gradients = gradients.to(device)
            images = images.to(device)
            actions = actions.to(device)

            pred_images, pred_actions, _ = model(gradients)
            loss, loss_dict = criterion(pred_images, images, pred_actions, actions)

            for k, v in loss_dict.items():
                total_losses[k] += v

            pred_labels = pred_actions.argmax(dim=-1).flatten()
            correct += (pred_labels == actions.flatten()).sum().item()
            total += actions.numel()

    for k in total_losses:
        total_losses[k] /= len(dataloader)
    total_losses["accuracy"] = 100 * correct / total

    return dict(total_losses)


def save_temporal_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_sequences: int = 2,
):
    """Save sample reconstructions showing temporal progression."""
    model.eval()

    gradients, images, actions = next(iter(dataloader))
    num_sequences = min(num_sequences, len(gradients))
    gradients = gradients[:num_sequences].to(device)
    images = images[:num_sequences]
    actions = actions[:num_sequences]

    with torch.no_grad():
        pred_images, pred_actions, _ = model(gradients)

    pred_images = pred_images.cpu()
    pred_labels = pred_actions.argmax(dim=-1).cpu()

    T = images.shape[1]

    # Create figure: 2 rows per sequence (GT, Pred), T columns
    fig, axes = plt.subplots(2 * num_sequences, T, figsize=(2 * T, 4 * num_sequences))

    for seq_idx in range(num_sequences):
        for t in range(T):
            row_gt = 2 * seq_idx
            row_pred = 2 * seq_idx + 1

            # Ground truth
            axes[row_gt, t].imshow(images[seq_idx, t].permute(1, 2, 0).numpy())
            axes[row_gt, t].set_title(
                f"t={t} GT (a={actions[seq_idx, t].item()})", fontsize=8
            )
            axes[row_gt, t].axis("off")

            # Prediction
            axes[row_pred, t].imshow(pred_images[seq_idx, t].permute(1, 2, 0).numpy())
            axes[row_pred, t].set_title(
                f"t={t} Pred (a={pred_labels[seq_idx, t].item()})", fontsize=8
            )
            axes[row_pred, t].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved temporal reconstructions to {save_path}")


# =============================================================================
# Main Training Function
# =============================================================================


def train(
    h5_path: str = "trajectory_data/gradients.h5",
    num_epochs: int = 50,
    batch_size: int = 2,
    accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    device: str = "auto",
    save_dir: str = "ckpts/attacker_temporal",
    gradient_dim: Optional[int] = None,
    sequence_length: int = 8,
    stride: int = 4,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    temporal_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    perceptual_layers: List[str] = None,
):
    """Main training function for temporal model."""
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
    )
    actual_gradient_dim = dataset.gradients.shape[1]

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Get number of actions from dataset
    num_actions = len(dataset.actions.unique())
    print(f"  Number of actions: {num_actions}")

    # Create model
    model = TemporalGradientInversion(
        gradient_dim=actual_gradient_dim,
        latent_dim=512,
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # Loss and optimizer
    criterion = TemporalCombinedLoss(
        mse_weight=1.0,
        l1_weight=0.5,
        action_weight=0.1,
        temporal_weight=temporal_weight,
        perceptual_weight=perceptual_weight,
        perceptual_layers=perceptual_layers,
    ).to(device)  # Move to device for VGG buffers

    print(f"\nLoss weights:")
    print(f"  MSE: 1.0, L1: 0.5, Action: 0.1")
    print(f"  Perceptual: {perceptual_weight}, Temporal: {temporal_weight}")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    if scaler:
        print("Using mixed precision training (fp16)")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("\n" + "=" * 60)
    print("Starting temporal training...")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
        )
        train_losses.append(train_loss["total"])

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss["total"])

        scheduler.step()

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss['total']:.4f} | "
            f"Val Loss: {val_loss['total']:.4f} | "
            f"Train Acc: {train_loss['accuracy']:.1f}% | "
            f"Val Acc: {val_loss['accuracy']:.1f}%"
        )

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_dir / "best_model.pt",
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save reconstructions periodically
        if epoch % 5 == 0 or epoch == 1:
            save_temporal_reconstructions(
                model, val_loader, device, save_dir / f"recon_epoch_{epoch:03d}.png"
            )

    # Save final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        save_dir / "final_model.pt",
    )

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Temporal Model Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train temporal gradient inversion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ppo/config_temporal.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--gradient_dim", type=int, default=None, help="Override gradient_dim"
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Override save_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument(
        "--seq_len", type=int, default=None, help="Override sequence_length"
    )

    args = parser.parse_args()

    # Try to load config
    try:
        from omegaconf import OmegaConf

        if Path(args.config).exists():
            cfg = OmegaConf.load(args.config)
            print(f"Loaded config from {args.config}")
        else:
            print(f"Config file not found: {args.config}, using defaults")
            cfg = OmegaConf.create({})
    except ImportError:
        print("OmegaConf not installed, using defaults")
        cfg = {}

    # Extract config with defaults
    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    training_cfg = cfg.get("training", {}) if hasattr(cfg, "get") else {}
    output_cfg = cfg.get("output", {}) if hasattr(cfg, "get") else {}
    loss_cfg = cfg.get("loss", {}) if hasattr(cfg, "get") else {}

    # Apply CLI overrides
    gradient_dim = args.gradient_dim or data_cfg.get("gradient_dim", 131072)
    save_dir = args.save_dir or output_cfg.get("save_dir", "ckpts/attacker_temporal")
    num_epochs = args.epochs or training_cfg.get("num_epochs", 50)
    seq_len = args.seq_len or model_cfg.get("sequence_length", 8)

    train(
        h5_path=data_cfg.get("h5_path", "trajectory_data/gradients.h5"),
        num_epochs=num_epochs,
        batch_size=training_cfg.get("batch_size", 2),
        accumulation_steps=training_cfg.get("accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        device=cfg.get("device", "auto") if hasattr(cfg, "get") else "auto",
        save_dir=save_dir,
        gradient_dim=gradient_dim,
        sequence_length=seq_len,
        stride=model_cfg.get("stride", 4),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        temporal_weight=training_cfg.get("temporal_weight", 0.0),
        perceptual_weight=loss_cfg.get("perceptual_weight", 0.0),
        perceptual_layers=loss_cfg.get("perceptual_layers", None),
    )

# Usage:
# uv run -m ppo.attacker_temporal --gradient_dim 131072
# uv run -m ppo.attacker_temporal --config ppo/config_temporal.yaml

"""
Transformer-based Gradient Inversion Model for PointNav.

This model learns to reconstruct RGB images (3, 84, 84) from flattened
gradients captured during PointNav training.

Data structure (from capture_gradients.py):
- gradients: (N, 3356646) - flattened gradients from all layers
- images: (N, 3, 84, 84) - RGB observations
- actions: (N,) - discrete actions
- episode_ids: (N,) - for sequence grouping
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==============================================================================
# Dataset
# ==============================================================================
class GradientImageDataset(Dataset):
    """
    Dataset for loading gradients and images from HDF5 file.

    Each sample contains:
    - gradient: Tensor of shape (gradient_dim,) - flattened gradients
    - image: Tensor of shape (3, 84, 84) - RGB image normalized to [0, 1]
    - action: Tensor of shape (1,) - action taken
    """

    def __init__(
        self,
        h5_path: str,
        gradient_dim: Optional[int] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            h5_path: Path to HDF5 file with gradients and images
            gradient_dim: Optional dimension to truncate/pad gradients to
            transform: Optional transform for images
        """
        self.h5_path = h5_path
        self.gradient_dim = gradient_dim
        self.transform = transform

        # Load data into memory (1.19GB is manageable)
        print(f"Loading data from {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            self.gradients = torch.from_numpy(f["gradients"][:].astype(np.float32))
            self.images = torch.from_numpy(f["images"][:].astype(np.float32)) / 255.0
            self.actions = torch.from_numpy(f["actions"][:].astype(np.int64))
            self.episode_ids = torch.from_numpy(f["episode_ids"][:].astype(np.int64))

        # Truncate or pad gradients if needed
        if gradient_dim is not None:
            current_dim = self.gradients.shape[1]
            if gradient_dim < current_dim:
                self.gradients = self.gradients[:, :gradient_dim]
            elif gradient_dim > current_dim:
                padding = torch.zeros(len(self.gradients), gradient_dim - current_dim)
                self.gradients = torch.cat([self.gradients, padding], dim=1)

        print(f"  Loaded {len(self)} samples")
        print(f"  Gradient shape: {self.gradients.shape}")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Actions: {self.actions.unique().tolist()}")

    def __len__(self) -> int:
        return len(self.gradients)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            gradient: (gradient_dim,)
            image: (3, 84, 84) normalized to [0, 1]
            action: scalar action
        """
        gradient = self.gradients[idx]
        image = self.images[idx]
        action = self.actions[idx]

        if self.transform:
            image = self.transform(image)

        return gradient, image, action


class GradientSequenceDataset(Dataset):
    """
    Dataset that groups samples by episode for sequence modeling.

    Each sample is a sequence of (gradients, images, actions) from one episode.
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 10,
        gradient_dim: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            h5_path: Path to HDF5 file
            seq_len: Fixed sequence length for each sample
            gradient_dim: Optional dimension to truncate gradients to
        """
        self.seq_len = seq_len
        self.gradient_dim = gradient_dim

        # Load all data
        print(f"Loading sequence data from {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            gradients = f["gradients"][:].astype(np.float32)
            images = f["images"][:].astype(np.float32) / 255.0
            actions = f["actions"][:].astype(np.int64)
            episode_ids = f["episode_ids"][:]

        # Truncate gradients if needed
        if gradient_dim is not None and gradient_dim < gradients.shape[1]:
            gradients = gradients[:, :gradient_dim]

        self.actual_gradient_dim = gradients.shape[1]

        # Group by episode and create fixed-length sequences
        self.sequences = []
        unique_episodes = np.unique(episode_ids)

        for ep_id in tqdm(unique_episodes, desc="Building sequences"):
            mask = episode_ids == ep_id
            ep_gradients = gradients[mask]
            ep_images = images[mask]
            ep_actions = actions[mask]

            # Create overlapping sequences of fixed length
            n_samples = len(ep_gradients)
            for start in range(0, n_samples - seq_len + 1, seq_len // 2):
                end = start + seq_len
                self.sequences.append(
                    {
                        "gradients": torch.from_numpy(ep_gradients[start:end]),
                        "images": torch.from_numpy(ep_images[start:end]),
                        "actions": torch.from_numpy(ep_actions[start:end]),
                    }
                )

        print(f"  Created {len(self)} sequences of length {seq_len}")
        print(f"  Gradient dim: {self.actual_gradient_dim}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence.

        Returns:
            gradients: (seq_len, gradient_dim)
            images: (seq_len, 3, 84, 84)
            actions: (seq_len,)
        """
        seq = self.sequences[idx]
        return seq["gradients"], seq["images"], seq["actions"]


# ==============================================================================
# Model Components
# ==============================================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GradientEncoder(nn.Module):
    """
    Encode high-dimensional gradients to a lower-dimensional representation.

    Uses a series of linear layers with residual connections to compress
    the 3.3M dimensional gradient to a manageable size.
    """

    def __init__(
        self,
        input_dim: int = 3356646,
        hidden_dims: List[int] = [4096, 2048, 1024],
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Progressive compression
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, output_dim)"""
        return self.encoder(x)


class CNNDecoder(nn.Module):
    """
    Decode latent representation to RGB image.

    Upsamples from small spatial feature map to (3, 84, 84) image.
    """

    def __init__(
        self,
        input_dim: int = 512,
        initial_channels: int = 256,
        initial_size: int = 7,
    ):
        super().__init__()

        self.initial_channels = initial_channels
        self.initial_size = initial_size

        # Project to spatial feature map
        self.fc = nn.Linear(input_dim, initial_channels * initial_size * initial_size)

        # Upsample: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 84x84
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 56x56 -> 84x84 (use output_padding to reach exact size)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Adaptive pool to exact size and final conv
            nn.AdaptiveAvgPool2d((84, 84)),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 3, 84, 84)"""
        batch_size = x.size(0)

        # Project to spatial
        x = self.fc(x)
        x = x.view(
            batch_size, self.initial_channels, self.initial_size, self.initial_size
        )

        # Decode to image
        x = self.decoder(x)
        return x


class ActionPredictor(nn.Module):
    """Predict action from latent representation."""

    def __init__(
        self, input_dim: int = 512, num_actions: int = 4, hidden_dim: int = 256
    ):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, num_actions)"""
        return self.predictor(x)


# ==============================================================================
# Main Model
# ==============================================================================
class GradientInversionTransformer(nn.Module):
    """
    Transformer-based model for gradient inversion.

    Takes flattened gradients as input and reconstructs:
    1. RGB observation image (3, 84, 84)
    2. Action that was taken

    Architecture:
    1. Gradient Encoder: Compress 3.3M gradients to 512-dim latent
    2. Optional: Transformer for sequence modeling
    3. CNN Decoder: Upsample latent to RGB image
    4. Action Head: Predict action from latent
    """

    def __init__(
        self,
        gradient_dim: int = 3356646,
        latent_dim: int = 512,
        num_actions: int = 4,
        use_transformer: bool = False,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gradient_dim = gradient_dim
        self.latent_dim = latent_dim
        self.use_transformer = use_transformer

        # Gradient encoder
        self.gradient_encoder = GradientEncoder(
            input_dim=gradient_dim,
            hidden_dims=[4096, 2048, 1024],
            output_dim=latent_dim,
            dropout=dropout,
        )

        # Optional transformer for sequence modeling
        if use_transformer:
            self.pos_encoder = PositionalEncoding(latent_dim, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=nhead,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

        # Decoders
        self.image_decoder = CNNDecoder(input_dim=latent_dim)
        self.action_predictor = ActionPredictor(
            input_dim=latent_dim, num_actions=num_actions
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            gradients: (batch, gradient_dim) or (batch, seq_len, gradient_dim)

        Returns:
            images: (batch, 3, 84, 84) or (batch, seq_len, 3, 84, 84)
            action_logits: (batch, num_actions) or (batch, seq_len, num_actions)
            latent: (batch, latent_dim) or (batch, seq_len, latent_dim)
        """
        is_sequence = gradients.dim() == 3

        if is_sequence and self.use_transformer:
            batch_size, seq_len, _ = gradients.shape

            # Encode each gradient in sequence
            gradients_flat = gradients.view(-1, self.gradient_dim)
            latent = self.gradient_encoder(gradients_flat)
            latent = latent.view(batch_size, seq_len, -1)

            # Apply transformer
            latent = self.pos_encoder(latent)
            latent = self.transformer(latent)

            # Decode each position
            latent_flat = latent.view(-1, self.latent_dim)
            images = self.image_decoder(latent_flat)
            action_logits = self.action_predictor(latent_flat)

            # Reshape back
            images = images.view(batch_size, seq_len, 3, 84, 84)
            action_logits = action_logits.view(batch_size, seq_len, -1)

        else:
            # Single sample mode
            if is_sequence:
                batch_size, seq_len, _ = gradients.shape
                gradients = gradients.view(-1, self.gradient_dim)

            latent = self.gradient_encoder(gradients)
            images = self.image_decoder(latent)
            action_logits = self.action_predictor(latent)

            if is_sequence:
                images = images.view(batch_size, seq_len, 3, 84, 84)
                action_logits = action_logits.view(batch_size, seq_len, -1)
                latent = latent.view(batch_size, seq_len, -1)

        return images, action_logits, latent


# ==============================================================================
# Loss Functions
# ==============================================================================
class CombinedLoss(nn.Module):
    """
    Combined loss for image reconstruction and action prediction.

    Combines:
    1. MSE Loss - Pixel-wise image reconstruction
    2. L1 Loss - Robustness to outliers
    3. Cross-Entropy Loss - Action classification
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        action_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.action_weight = action_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual losses for logging
        """
        # Flatten sequence dimension if present
        if pred_images.dim() == 5:
            pred_images = pred_images.view(-1, 3, 84, 84)
            target_images = target_images.view(-1, 3, 84, 84)
            pred_actions = pred_actions.view(-1, pred_actions.size(-1))
            target_actions = target_actions.view(-1)

        # Image losses
        mse = self.mse_loss(pred_images, target_images)
        l1 = self.l1_loss(pred_images, target_images)

        # Action loss
        action_loss = self.ce_loss(pred_actions, target_actions)

        # Combined
        total = (
            self.mse_weight * mse
            + self.l1_weight * l1
            + self.action_weight * action_loss
        )

        loss_dict = {
            "mse": mse.item(),
            "l1": l1.item(),
            "action": action_loss.item(),
            "total": total.item(),
        }

        return total, loss_dict


# ==============================================================================
# Training Functions
# ==============================================================================
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
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_losses = {"mse": 0, "l1": 0, "action": 0, "total": 0}
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for gradients, images, actions in pbar:
        gradients = gradients.to(device)
        images = images.to(device)
        actions = actions.to(device)

        # Forward with mixed precision
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            pred_images, pred_actions, _ = model(gradients)
            loss, loss_dict = criterion(pred_images, images, pred_actions, actions)
            loss = loss / accumulation_steps  # Scale for accumulation

        # Backward with gradient scaling
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

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] += v

        # Accuracy
        pred_labels = pred_actions.argmax(dim=-1)
        correct += (pred_labels == actions).sum().item()
        total += actions.numel()

        pbar.set_postfix(
            loss=f"{loss_dict['total']:.4f}",
            acc=f"{100 * correct / total:.1f}%",
        )

    # Average
    for k in total_losses:
        total_losses[k] /= len(dataloader)
    total_losses["accuracy"] = 100 * correct / total

    return total_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_losses = {"mse": 0, "l1": 0, "action": 0, "total": 0}
    correct = 0
    total = 0

    with torch.no_grad():
        for gradients, images, actions in dataloader:
            gradients = gradients.to(device)
            images = images.to(device)
            actions = actions.to(device)

            pred_images, pred_actions, _ = model(gradients)
            _, loss_dict = criterion(pred_images, images, pred_actions, actions)

            for k, v in loss_dict.items():
                total_losses[k] += v

            pred_labels = pred_actions.argmax(dim=-1)
            correct += (pred_labels == actions).sum().item()
            total += actions.numel()

    for k in total_losses:
        total_losses[k] /= len(dataloader)
    total_losses["accuracy"] = 100 * correct / total

    return total_losses


def save_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
):
    """Save sample reconstructions for visualization."""
    model.eval()

    gradients, images, actions = next(iter(dataloader))
    gradients = gradients[:num_samples].to(device)
    images = images[:num_samples]
    actions = actions[:num_samples]

    with torch.no_grad():
        pred_images, pred_actions, _ = model(gradients)

    pred_images = pred_images.cpu()
    pred_labels = pred_actions.argmax(dim=-1).cpu()

    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    for i in range(num_samples):
        # Ground truth
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"GT (a={actions[i].item()})")
        axes[0, i].axis("off")

        # Prediction
        axes[1, i].imshow(pred_images[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"Pred (a={pred_labels[i].item()})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reconstructions to {save_path}")


def train(
    h5_path: str = "trajectory_data/gradients.h5",
    num_epochs: int = 50,
    batch_size: int = 4,  # Small batch for large models
    accumulation_steps: int = 8,  # Effective batch = 4 * 8 = 32
    learning_rate: float = 1e-4,
    device: str = "auto",
    save_dir: str = "ckpts/attacker",
    gradient_dim: Optional[int] = None,
    use_transformer: bool = False,
):
    """
    Main training function.

    Args:
        h5_path: Path to HDF5 file with gradients and images
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on ("auto", "cuda", "mps", "cpu")
        save_dir: Directory to save ckpts
        gradient_dim: Optional gradient dimension (None = use full)
        use_transformer: Whether to use transformer for sequence modeling
        accumulation_steps: Number of gradient accumulation steps
    """
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
    dataset = GradientImageDataset(h5_path, gradient_dim=gradient_dim)
    actual_gradient_dim = dataset.gradients.shape[1]

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")

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
    model = GradientInversionTransformer(
        gradient_dim=actual_gradient_dim,
        latent_dim=512,
        num_actions=num_actions,  # PointNav: forward, left, right, look_up, look_down, stop
        use_transformer=use_transformer,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(mse_weight=1.0, l1_weight=0.5, action_weight=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    if scaler:
        print("Using mixed precision training (fp16)")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # Train
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

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss["total"])

        # Update scheduler
        scheduler.step()

        # Log
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss['total']:.4f} | "
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
            save_reconstructions(
                model, val_loader, device, save_dir / f"recon_epoch_{epoch:03d}.png"
            )

    # Save final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"ckpts saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train gradient inversion model")
    parser.add_argument(
        "--h5_path",
        type=str,
        default="trajectory_data/gradients.h5",
        help="Path to HDF5 data file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpts/attacker",
        help="Save directory",
    )
    parser.add_argument(
        "--gradient_dim",
        type=int,
        default=None,
        help="Gradient dimension (None = use full)",
    )
    parser.add_argument(
        "--use_transformer",
        action="store_true",
        default=True,
        help="Use transformer for sequence modeling",
    )

    args = parser.parse_args()

    train(
        h5_path=args.h5_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        gradient_dim=args.gradient_dim,
        use_transformer=args.use_transformer,
    )

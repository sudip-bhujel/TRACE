"""
Training script for the Gradient-to-Image Transformer.

This script trains a transformer to reconstruct images from gradient sequences.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tf.dataset import create_dataloaders
from tf.model import GradientToImageTransformer


class PerceptualLoss(nn.Module):
    """
    Combined loss for better image reconstruction.

    Combines:
    1. MSE Loss - Pixel-wise accuracy
    2. L1 Loss - Robustness to outliers
    3. SSIM Loss - Structural similarity
    4. Gradient Loss - Edge preservation
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        gradient_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        # Simple SSIM approximation
        mu_pred = pred.mean()
        mu_target = target.mean()

        sigma_pred = pred.var()
        sigma_target = target.var()
        sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()

        c1 = 0.01**2
        c2 = 0.03**2

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2)
        )

        return 1 - ssim

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss to preserve edges."""
        # Compute gradients using Sobel-like filters
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = self.l1(pred_dx, target_dx)
        loss_dy = self.l1(pred_dy, target_dy)

        return loss_dx + loss_dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        loss = 0.0

        # MSE loss
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(pred, target)

        # L1 loss
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1(pred, target)

        # SSIM loss
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(pred, target)

        # Gradient loss
        if self.gradient_weight > 0:
            loss += self.gradient_weight * self.gradient_loss(pred, target)

        return loss


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    step: int,
) -> Tuple[float, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for gradients, images in pbar:
        # Move to device
        gradients = gradients.to(device)  # (batch, seq_len, 64, 512)
        images = images.to(device)  # (batch, seq_len, 120, 120)

        # Forward pass
        reconstructed = model(gradients)  # (batch, seq_len, 120, 120)

        # Compute loss
        loss = criterion(reconstructed, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        step += 1

        wandb.log({"train/loss": loss.item()}, step=step)

    avg_loss = total_loss / num_batches
    return avg_loss, step


def validate(
    model: nn.Module, val_loader, criterion: nn.Module, device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for gradients, images in val_loader:
            gradients = gradients.to(device)
            images = images.to(device)

            reconstructed = model(gradients)
            loss = criterion(reconstructed, images)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_sample_reconstruction(
    model: nn.Module, val_loader, device: torch.device, save_dir: Path, epoch: int
):
    """Save a sample reconstruction for visualization."""
    model.eval()

    # Get one batch
    for gradients, images in val_loader:
        gradients = gradients.to(device)
        images = images.to(device)

        with torch.no_grad():
            reconstructed = model(gradients)

        # Take first scene from batch
        orig_images = images[0].cpu()  # (seq_len, 120, 120)
        recon_images = reconstructed[0].cpu()  # (seq_len, 120, 120)

        # Plot first 5 actions
        num_actions = min(5, orig_images.shape[0])
        fig, axes = plt.subplots(2, num_actions, figsize=(15, 6))

        for i in range(num_actions):
            # Original
            axes[0, i].imshow(orig_images[i], cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original", fontsize=10)

            # Reconstructed
            axes[1, i].imshow(recon_images[i], cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstructed", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_dir / f"reconstruction_epoch_{epoch}.png", dpi=150)
        plt.close()

        break  # Only save one example


def plot_training_curves(
    train_losses: List[float], val_losses: List[float], save_dir: Path
):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()


def train(
    features_path: str = "features/all_scenes.pkl",
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "auto",
    save_dir: str = "checkpoints",
    log_interval: int = 5,
):
    """
    Main training function.

    Args:
        features_path: Path to collected gradient features
        num_epochs: Number of training epochs
        batch_size: Batch size (number of scenes)
        learning_rate: Learning rate for optimizer
        device: Device to use ('auto', 'cuda', 'cpu')
        save_dir: Directory to save checkpoints
        log_interval: Save samples every N epochs
    """
    # Setup
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Device
    if device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    print("=" * 80)
    print("Gradient-to-Image Transformer Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Features: {features_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  Save directory: {save_dir}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        features_path=features_path,
        batch_size=batch_size,
        train_split=0.9,
        num_workers=0,  # Use 0 to avoid multiprocessing serialization issues
        use_normalized_gradients=True,  # Use normalized gradients (weight/bias)
        preload_images=True,  # Preload all images into memory
    )

    # Create model
    print("\nCreating model...")
    model = GradientToImageTransformer(
        gradient_shape=(64, 512),
        img_size=(120, 120),  # Updated to 120x120
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.2,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")

    # Loss and optimizer - USE PERCEPTUAL LOSS!
    # print("\nUsing Perceptual Loss:")
    # print("  - MSE Loss (pixel accuracy)")
    # print("  - L1 Loss (robustness)")
    # print("  - SSIM Loss (structural similarity)")
    # print("  - Gradient Loss (edge preservation)")

    # criterion = PerceptualLoss(
    #     mse_weight=1.0, l1_weight=1.0, ssim_weight=0.5, gradient_weight=0.5
    # )
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    step = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, step = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, step
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Learning Rate: {new_lr:.2e}")

        if new_lr < old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_dir / "best_model.pth",
            )
            print(f"  ✅ New best model saved (val_loss: {val_loss:.6f})")

        # Save sample reconstruction
        if epoch % log_interval == 0:
            save_sample_reconstruction(model, val_loader, device, save_dir, epoch)
            plot_training_curves(train_losses, val_losses, save_dir)

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_dir / f"checkpoint_epoch_{epoch}.pth",
            )

        wandb.log(
            {
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Learning Rate": new_lr,
                "Epoch": epoch,
            },
            # step=epoch,
        )

    # Final save
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        save_dir / "final_model.pth",
    )

    plot_training_curves(train_losses, val_losses, save_dir)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Models saved in: {save_dir}")
    print(f"\nTo use the trained model:")
    print(f"  model = GradientToImageTransformer(...)")
    print(f"  checkpoint = torch.load('{save_dir}/best_model.pth')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Gradient-to-Image Transformer")
    parser.add_argument(
        "--features",
        type=str,
        default="features/all_scenes.pkl",
        help="Path to collected features",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/tf_checkpoint",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    wandb.init(
        project="Inversion",
        config={
            "features": args.features,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": args.device,
            "save_dir": args.save_dir,
        },
    )

    train(
        features_path=args.features,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )

# uv run -m tf.train --features features/all_scenes_1000.pkl --epochs 100 --batch-size 64 --lr 1e-4 --device cuda --save-dir ckpts/tf_mse_1000

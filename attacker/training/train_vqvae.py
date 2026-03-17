"""
VQ-VAE Training Script for Indoor Scene Images.

Trains a VQ-VAE on images extracted from the gradient HDF5 dataset.
The trained VQ-VAE provides the visual codebook and decoder for the
gradient inversion token prediction pipeline.

Usage:
    uv run -m attacker.train_vqvae ./attacker/config/train_vqvae.yaml
"""

import os
import sys

import h5py
import lpips
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb
from attacker.models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageOnlyDataset(Dataset):
    """Load only images from gradient HDF5 for VQ-VAE training."""

    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.images = torch.tensor(f["images"][:], dtype=torch.float32) / 255.0
        print(f"Loaded {len(self.images)} images from {h5_path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


def train_vqvae(
    h5_path: str,
    save_dir: str = "ckpts/vqvae",
    codebook_size: int = 512,
    embedding_dim: int = 256,
    hidden_channels: int = 128,
    num_res_blocks: int = 2,
    commitment_cost: float = 0.25,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    lpips_weight: float = 0.5,
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
    num_workers: int = 4,
    wandb_cfg: dict = None,
):
    """Train VQ-VAE on environment images."""
    os.makedirs(save_dir, exist_ok=True)

    # Dataset
    dataset = ImageOnlyDataset(h5_path)
    val_size = min(len(dataset) // 10, 5000)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = VQVAE(
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_res_blocks=num_res_blocks,
        commitment_cost=commitment_cost,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VQ-VAE parameters: {total_params:,}")

    # Losses
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device) if lpips_weight > 0 else None

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # WandB
    if wandb_cfg and wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg.get("project", "gradient-inversion"),
            name=wandb_cfg.get("name", "vqvae-training"),
            entity=wandb_cfg.get("entity", None),
            config={
                "codebook_size": codebook_size,
                "embedding_dim": embedding_dim,
                "hidden_channels": hidden_channels,
                "num_res_blocks": num_res_blocks,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = {"total": 0, "mse": 0, "l1": 0, "lpips": 0, "vq": 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for images in pbar:
            images = images.to(device)

            recon, indices, vq_loss = model(images)

            # Reconstruction losses
            mse = nn.functional.mse_loss(recon, images)
            l1 = nn.functional.l1_loss(recon, images)

            loss = mse_weight * mse + l1_weight * l1 + vq_loss

            if lpips_loss_fn is not None:
                perceptual = lpips_loss_fn(recon, images).mean()
                loss = loss + lpips_weight * perceptual
                train_losses["lpips"] += perceptual.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses["total"] += loss.item()
            train_losses["mse"] += mse.item()
            train_losses["l1"] += l1.item()
            train_losses["vq"] += vq_loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", vq=f"{vq_loss.item():.4f}")

        # Average
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # Validation
        model.eval()
        val_losses = {"total": 0, "mse": 0, "l1": 0, "lpips": 0, "vq": 0}
        codebook_usage = 0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon, indices, vq_loss = model(images)

                mse = nn.functional.mse_loss(recon, images)
                l1 = nn.functional.l1_loss(recon, images)
                loss = mse_weight * mse + l1_weight * l1 + vq_loss

                if lpips_loss_fn is not None:
                    perceptual = lpips_loss_fn(recon, images).mean()
                    loss = loss + lpips_weight * perceptual
                    val_losses["lpips"] += perceptual.item()

                val_losses["total"] += loss.item()
                val_losses["mse"] += mse.item()
                val_losses["l1"] += l1.item()
                val_losses["vq"] += vq_loss.item()

            codebook_usage = model.quantizer.get_codebook_usage()

        for k in val_losses:
            val_losses[k] /= len(val_loader)

        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_losses['total']:.4f} "
            f"(MSE: {train_losses['mse']:.4f}, VQ: {train_losses['vq']:.4f}) | "
            f"Val Loss: {val_losses['total']:.4f} "
            f"(MSE: {val_losses['mse']:.4f}, VQ: {val_losses['vq']:.4f}) | "
            f"Codebook: {codebook_usage:.1%}"
        )

        if wandb.run:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/total": train_losses["total"],
                    "train/mse": train_losses["mse"],
                    "train/l1": train_losses["l1"],
                    "train/lpips": train_losses["lpips"],
                    "train/vq": train_losses["vq"],
                    "val/total": val_losses["total"],
                    "val/mse": val_losses["mse"],
                    "val/l1": val_losses["l1"],
                    "val/lpips": val_losses["lpips"],
                    "val/vq": val_losses["vq"],
                    "codebook_usage": codebook_usage,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config,
                    "epoch": epoch,
                    "val_loss": val_losses["total"],
                    "codebook_usage": codebook_usage,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  Saved best model (val_loss={val_losses['total']:.4f})")

        # Save reconstruction samples every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            save_vqvae_samples(model, val_loader, save_dir, epoch)

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "epoch": num_epochs,
            "val_loss": val_losses["total"],
            "codebook_usage": codebook_usage,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    if wandb.run:
        wandb.finish()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}")


@torch.no_grad()
def save_vqvae_samples(
    model: VQVAE,
    dataloader: DataLoader,
    save_dir: str,
    epoch: int,
    num_samples: int = 8,
):
    """Save VQ-VAE reconstruction samples for visualization."""
    model.eval()
    images = next(iter(dataloader))[:num_samples].to(device)
    recon, indices, _ = model(images)

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        # Reconstruction
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("VQ-VAE Recon", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"vqvae_recon_epoch_{epoch:03d}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved VQ-VAE reconstruction samples to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run -m attacker.train_vqvae ./attacker/config/train_vqvae.yaml"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = OmegaConf.load(config_path)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    output_cfg = cfg.get("output", {})
    wandb_cfg = cfg.get("wandb", {})

    train_vqvae(
        h5_path=data_cfg.get("h5_path", "trajectory_data/gradients_108_augmented.h5"),
        save_dir=output_cfg.get("save_dir", "ckpts/vqvae"),
        codebook_size=model_cfg.get("codebook_size", 512),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        hidden_channels=model_cfg.get("hidden_channels", 128),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        commitment_cost=model_cfg.get("commitment_cost", 0.25),
        num_epochs=training_cfg.get("num_epochs", 50),
        batch_size=training_cfg.get("batch_size", 64),
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        lpips_weight=loss_cfg.get("lpips_weight", 0.5),
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        l1_weight=loss_cfg.get("l1_weight", 0.5),
        num_workers=training_cfg.get("num_workers", 4),
        wandb_cfg=wandb_cfg,
    )

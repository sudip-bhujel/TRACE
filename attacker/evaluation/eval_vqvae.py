"""
VQ-VAE Evaluation Script.

Evaluates a trained VQ-VAE on unseen data to verify reconstruction quality,
codebook utilization, and generalization to new scenes.

Usage:
    uv run -m attacker.eval_vqvae \
        --checkpoint ckpts/vqvae/best_model.pt \
        --h5_path trajectory_data/eval_tf_108.h5 \
        --save_dir ckpts/vqvae/eval_results
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from attacker.models.vqvae import load_vqvae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalImageDataset(Dataset):
    """Load images from HDF5 for VQ-VAE evaluation.

    Handles both flat image arrays and trajectory-structured datasets.
    """

    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            images = f["images"][:]
            # If images are (N, T, C, H, W) flatten to (N*T, C, H, W)
            if images.ndim == 5:
                N, T = images.shape[:2]
                images = images.reshape(N * T, *images.shape[2:])
                print(f"Flattened {N} trajectories x {T} steps = {len(images)} images")
            self.images = torch.tensor(images, dtype=torch.float32) / 255.0

        print(f"Loaded {len(self.images)} images from {h5_path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


@torch.no_grad()
def evaluate_vqvae(
    checkpoint: str,
    h5_path: str,
    save_dir: str = "ckpts/vqvae/eval_results",
    batch_size: int = 64,
    num_workers: int = 4,
    num_vis_samples: int = 16,
):
    """Evaluate VQ-VAE reconstruction quality on unseen data."""
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = load_vqvae(checkpoint, device=str(device))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VQ-VAE parameters: {total_params:,}")
    print(
        f"Codebook: {model.config['codebook_size']} entries x {model.config['embedding_dim']} dim"
    )

    # Load data
    dataset = EvalImageDataset(h5_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Metrics
    total_mse = 0.0
    total_l1 = 0.0
    total_psnr = 0.0
    all_indices = []
    num_batches = 0

    first_originals = None
    first_recons = None

    print("\nEvaluating VQ-VAE...")
    for images in tqdm(loader):
        images = images.to(device)
        recon, indices, vq_loss = model(images)

        # Store first batch for visualization
        if first_originals is None:
            first_originals = images[:num_vis_samples].cpu()
            first_recons = recon[:num_vis_samples].cpu()

        # Metrics
        mse = F.mse_loss(recon, images).item()
        l1 = F.l1_loss(recon, images).item()
        psnr = -10 * np.log10(mse + 1e-10)

        total_mse += mse
        total_l1 += l1
        total_psnr += psnr
        all_indices.append(indices.cpu())
        num_batches += 1

    # Average metrics
    avg_mse = total_mse / num_batches
    avg_l1 = total_l1 / num_batches
    avg_psnr = total_psnr / num_batches

    # Codebook analysis
    all_indices = torch.cat(all_indices, dim=0)  # (N, num_tokens)
    unique_tokens = len(torch.unique(all_indices))
    total_tokens = model.config["codebook_size"]
    usage_pct = unique_tokens / total_tokens * 100

    # Token frequency distribution
    token_counts = torch.bincount(all_indices.flatten(), minlength=total_tokens)
    active_mask = token_counts > 0
    active_counts = token_counts[active_mask]

    print("\n" + "=" * 60)
    print("VQ-VAE Evaluation Results (Unseen Data)")
    print("=" * 60)
    print(f"  Dataset: {h5_path}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Total images: {len(dataset)}")
    print("\nReconstruction Quality:")
    print(f"  MSE:  {avg_mse:.6f}")
    print(f"  L1:   {avg_l1:.4f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print("\nCodebook Utilization:")
    print(f"  Active tokens: {unique_tokens}/{total_tokens} ({usage_pct:.1f}%)")
    print(f"  Most used token:  count={token_counts.max().item()}")
    print(f"  Least used (active): count={active_counts.min().item()}")
    print(f"  Median usage: count={active_counts.median().item()}")
    print("=" * 60)

    # Save metrics to file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"dataset: {h5_path}\n")
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"num_images: {len(dataset)}\n")
        f.write(f"mse: {avg_mse:.6f}\n")
        f.write(f"l1: {avg_l1:.4f}\n")
        f.write(f"psnr: {avg_psnr:.2f}\n")
        f.write(f"codebook_active: {unique_tokens}/{total_tokens}\n")
        f.write(f"codebook_usage_pct: {usage_pct:.1f}\n")

    # === Visualization 1: Reconstruction grid ===
    n = min(num_vis_samples, len(first_originals))
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    fig.suptitle(
        f"VQ-VAE Reconstruction on Unseen Data | MSE={avg_mse:.5f} | PSNR={avg_psnr:.1f}dB | Codebook={usage_pct:.0f}%",
        fontsize=12,
    )
    for i in range(n):
        axes[0, i].imshow(first_originals[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        axes[1, i].imshow(first_recons[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruction", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "reconstruction_grid.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"\nSaved reconstruction grid to {save_dir}/reconstruction_grid.png")

    # === Visualization 2: Codebook usage histogram ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        range(total_tokens),
        token_counts.numpy(),
        width=1.0,
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Usage Count")
    ax.set_title(f"Codebook Usage Distribution ({unique_tokens}/{total_tokens} active)")
    ax.axhline(
        y=token_counts.float().mean().item(),
        color="red",
        linestyle="--",
        label=f"Mean={token_counts.float().mean():.0f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "codebook_usage.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved codebook usage histogram to {save_dir}/codebook_usage.png")

    # === Visualization 3: Per-image error heatmap (first few) ===
    n_error = min(8, len(first_originals))
    fig, axes = plt.subplots(3, n_error, figsize=(2.5 * n_error, 7.5))
    fig.suptitle("Per-Pixel Error Analysis", fontsize=12)
    for i in range(n_error):
        orig = first_originals[i].permute(1, 2, 0).clamp(0, 1)
        rec = first_recons[i].permute(1, 2, 0).clamp(0, 1)
        error = (orig - rec).abs().mean(dim=-1)  # Average over channels

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec)
        axes[1, i].axis("off")
        axes[2, i].imshow(error, cmap="hot", vmin=0, vmax=0.3)
        axes[2, i].axis("off")

        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)
            axes[1, i].set_title("Reconstruction", fontsize=9)
            axes[2, i].set_title("Error", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "error_heatmap.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved error heatmap to {save_dir}/error_heatmap.png")

    return {
        "mse": avg_mse,
        "l1": avg_l1,
        "psnr": avg_psnr,
        "codebook_active": unique_tokens,
        "codebook_total": total_tokens,
        "codebook_usage_pct": usage_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQ-VAE on unseen data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpts/vqvae/best_model.pt",
        help="Path to VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="trajectory_data/eval_tf_108.h5",
        help="Path to evaluation HDF5",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpts/vqvae/eval_results",
        help="Output directory for results",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_vis_samples", type=int, default=16, help="Number of samples to visualize"
    )
    args = parser.parse_args()

    evaluate_vqvae(
        checkpoint=args.checkpoint,
        h5_path=args.h5_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_vis_samples=args.num_vis_samples,
    )

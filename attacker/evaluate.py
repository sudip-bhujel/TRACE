"""
Temporal Gradient Inversion - Evaluation Script

This script loads a trained temporal model and generates reconstruction
visualizations on test data.

Usage:
    uv run -m attacker.evaluate config/eval_temporal.yaml
"""

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacker.dataset import TemporalGradientDataset
from attacker.model import TemporalGradientInversion


def load_model(
    checkpoint_path: str,
    gradient_dim: int,
    device: torch.device,
    num_actions: int = 5,
    latent_dim: int = 512,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
) -> nn.Module:
    """Load trained temporal model from checkpoint."""
    model = TemporalGradientInversion(
        gradient_dim=gradient_dim,
        latent_dim=latent_dim,
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        encoder_hidden_dims=encoder_hidden_dims,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        if "val_loss" in checkpoint:
            print(f"  Val loss: {checkpoint['val_loss'].get('total', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")

    model.eval()
    return model


def evaluate_and_save_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
):
    """
    Run evaluation on test data and save reconstruction figures.

    Saves one figure per sequence showing ground truth vs predictions
    for all timesteps.
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)
    total_correct = 0
    total_samples = 0
    total_mse = 0.0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"Only {seq_idx} sequences available in dataset")
            break

        # Take first sequence from batch
        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        with torch.no_grad():
            pred_images, pred_actions, _ = model(gradients)

        pred_images = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        # Compute metrics
        mse = ((pred_images - images) ** 2).mean().item()
        correct = (pred_labels == actions).sum().item()
        total = actions.numel()

        total_mse += mse
        total_correct += correct
        total_samples += total

        T = images.shape[1]

        # Create figure: 2 rows (GT, Pred), T columns
        fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))

        for t in range(T):
            # Ground truth
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t} GT (a={actions[0, t].item()})", fontsize=8)
            axes[0, t].axis("off")

            # Prediction
            pred_img = pred_images[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(
                f"t={t} Pred (a={pred_labels[0, t].item()})", fontsize=8
            )
            axes[1, t].axis("off")

        plt.suptitle(
            f"Sequence {seq_idx + 1} | MSE: {mse:.4f} | Acc: {100 * correct / total:.1f}%"
        )
        plt.tight_layout()

        save_path = save_dir / f"reconstruction_seq_{seq_idx + 1:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(
            f"Saved {save_path.name} | MSE: {mse:.4f} | Acc: {100 * correct / total:.1f}%"
        )

    # Print summary
    avg_mse = total_mse / num_sequences if num_sequences > 0 else 0
    avg_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"Evaluation Summary ({num_sequences} sequences)")
    print(f"  Average MSE: {avg_mse:.4f}")
    print(f"  Action Accuracy: {avg_acc:.1f}%")
    print(f"  Figures saved to: {save_dir}")
    print(f"{'=' * 50}")


def evaluate(
    checkpoint_path: str,
    h5_path: str,
    save_dir: str = "eval_results",
    num_sequences: int = 5,
    sequence_length: int = 8,
    stride: int = 8,  # Non-overlapping for test
    gradient_dim: Optional[int] = None,
    device: str = "auto",
    batch_size: int = 1,
    # Model architecture params
    latent_dim: int = 512,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
):
    """Main evaluation function."""
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

    # Load test dataset
    print(f"\nLoading test data from: {h5_path}")
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
    )

    # Use limited dimension if specified, otherwise full gradient shape
    if gradient_dim is not None and gradient_dim < dataset.gradient_shape[1]:
        actual_gradient_dim = gradient_dim
    else:
        actual_gradient_dim = dataset.gradient_shape[1]
    num_actions = len(dataset.actions.unique())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Sequences: {len(dataset)}")
    print(f"  Gradient dim: {actual_gradient_dim}")
    print(f"  Num actions: {num_actions}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(
        checkpoint_path,
        gradient_dim=actual_gradient_dim,
        device=device,
        num_actions=num_actions,
        latent_dim=latent_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        encoder_hidden_dims=encoder_hidden_dims,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    )

    # Run evaluation
    print(f"\nGenerating reconstructions for {num_sequences} sequences...")
    evaluate_and_save_reconstructions(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=Path(save_dir),
        num_sequences=num_sequences,
    )


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python evaluate.py <config_path>"

    cfg = OmegaConf.load(sys.argv[1])
    print(f"Loaded config from {sys.argv[1]}")

    # Extract config values
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    evaluate(
        checkpoint_path=eval_cfg.get("checkpoint"),
        h5_path=data_cfg.get("h5_path"),
        save_dir=output_cfg.get("save_dir", "eval_results"),
        num_sequences=eval_cfg.get("num_sequences", 5),
        sequence_length=model_cfg.get("sequence_length", 8),
        stride=model_cfg.get("stride", 8),
        gradient_dim=data_cfg.get("gradient_dim", None),
        device=cfg.get("device", "auto"),
        latent_dim=model_cfg.get("latent_dim", 512),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", None),
        encoder_type=model_cfg.get("encoder_type", "basic"),
        decoder_type=model_cfg.get("decoder_type", "basic"),
    )

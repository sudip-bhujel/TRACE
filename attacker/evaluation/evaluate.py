"""Evaluate a trained gradient inversion model and produce metrics + reconstruction figures."""

import math
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.metrics import (
    MetricsComputer,
    print_per_timestep_results,
    print_results,
)
from attacker.models.autoregressive_model import AutoregressiveGradientInversion
from attacker.models.model import TemporalGradientInversion

ACTION_NAMES = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookDown",
    "LookUp",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
]


def load_model(
    checkpoint_path: str,
    gradient_dim: int,
    device: torch.device,
    num_actions: int = 5,
    latent_dim: int = 512,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_hidden_dim: Optional[int] = None,
    encoder_num_blocks: Optional[int] = None,
    encoder_expansion: Optional[int] = None,
    encoder_projection_rank: Optional[int] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
    skip_transformer: bool = False,
    temporal_model_type: str = "transformer",
    ff_multiplier: int = 4,
    use_rope: bool = False,
    model_type: str = "temporal",
    **kwargs,
) -> nn.Module:
    """Load a trained attacker model from checkpoint."""
    if model_type == "autoregressive":
        model = AutoregressiveGradientInversion(
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            num_actions=num_actions,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_blocks=encoder_num_blocks,
            encoder_expansion=encoder_expansion,
            encoder_projection_rank=encoder_projection_rank,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            temporal_model_type=temporal_model_type,
            ff_multiplier=ff_multiplier,
            use_rope=use_rope,
        ).to(device)
    else:
        model = TemporalGradientInversion(
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            num_actions=num_actions,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_blocks=encoder_num_blocks,
            encoder_expansion=encoder_expansion,
            encoder_projection_rank=encoder_projection_rank,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            skip_transformer=skip_transformer,
            temporal_model_type=temporal_model_type,
            ff_multiplier=ff_multiplier,
            use_rope=use_rope,
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    is_pretrained = (
        checkpoint.get("decoder_type") == "pretrained" or decoder_type == "pretrained"
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=not is_pretrained)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        if is_pretrained:
            print("  (strict=False; frozen VAE weights from HuggingFace)")
        if "val_loss" in checkpoint:
            print(f"  Val loss: {checkpoint['val_loss'].get('total', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint, strict=not is_pretrained)
        print("Loaded model weights (old format)")

    model.eval()
    return model


def _save_confusion_matrix(cm: np.ndarray, save_path: Path, action_names: List[str]):
    """Save action prediction confusion matrix as a heatmap (PNG + PDF)."""
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=action_names,
        yticklabels=action_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Action Prediction Confusion Matrix")
    plt.tight_layout()
    stem = save_path.with_suffix("")
    plt.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def evaluate_and_save_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
    enable_fid: bool = True,
    num_actions: int = 5,
    model_type: str = "temporal",
    teacher_forcing: bool = False,
    action_names: Optional[List[str]] = None,
):
    """Compute aggregate metrics and save reconstruction figures + confusion matrix."""
    plt.rcParams["font.family"] = "serif"
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)

    action_labels = action_names
    if action_labels is None or len(action_labels) != num_actions:
        action_labels = (
            ACTION_NAMES[:num_actions]
            if num_actions <= len(ACTION_NAMES)
            else [f"Action {i}" for i in range(num_actions)]
        )

    data_iter = iter(dataloader)
    evaluated = 0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"Only {seq_idx} sequences available in dataset")
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        with torch.no_grad():
            if model_type == "autoregressive":
                pred_images, pred_actions, _, _ = model(
                    gradients,
                    images=images.to(device) if teacher_forcing else None,
                    teacher_forcing=teacher_forcing,
                )
            else:
                pred_images, pred_actions, _, _ = model(gradients)

        metrics.update(pred_images, images, pred_actions, actions)

        pred_images_cpu = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        mse = ((pred_images_cpu - images) ** 2).mean().item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        correct = (pred_labels == actions).sum().item()
        total = actions.numel()
        evaluated += 1

        T = images.shape[1]

        fig, axes = plt.subplots(2, T, figsize=(1.8 * T, 4), squeeze=False)
        fig.subplots_adjust(wspace=0.07, hspace=0.12)
        for t in range(T):
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t}, a={actions[0, t].item()}", fontsize=10)
            axes[0, t].axis("off")

            pred_img = pred_images_cpu[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(f"t={t}, a={pred_labels[0, t].item()}", fontsize=10)
            axes[1, t].axis("off")

        axes[0, 0].set_ylabel("Ground Truth", fontsize=10, rotation=90, labelpad=4)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=10, rotation=90, labelpad=4)
        for row in range(2):
            axes[row, 0].axis("on")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            for spine in axes[row, 0].spines.values():
                spine.set_visible(False)

        stem = save_dir / f"reconstruction_seq_{seq_idx + 1:03d}"
        plt.savefig(
            stem.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.0
        )
        plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0)
        plt.close()

        print(
            f"Saved {stem.name} (.png/.pdf) | MSE: {mse:.4f} | PSNR: {psnr:.1f} dB"
            f" | Action Accuracy: {100 * correct / total:.1f}%"
        )

    results = metrics.compute()

    print_results(
        results,
        header=(
            f"\nEvaluation Summary ({results.get('num_images', 0)} images"
            f" from {evaluated} sequences)"
        ),
    )

    per_timestep = metrics.compute_per_timestep()
    if per_timestep:
        print("\n  Per-timestep metrics:")
        print_per_timestep_results(per_timestep)
        print()

    metrics.save(save_dir / "metrics.json")
    print(f"  Metrics saved to: {save_dir / 'metrics.json'}")
    print(f"  Metrics saved to: {save_dir / 'metrics.csv'}")

    if metrics.all_pred_actions:
        cm = metrics.confusion_matrix(num_actions=num_actions)
        cm_path = save_dir / "confusion_matrix.png"
        _save_confusion_matrix(cm, cm_path, action_labels)
        print(
            f"  Confusion matrix saved to: {cm_path} / {cm_path.with_suffix('.pdf').name}"
        )

    print(f"  Figures saved to: {save_dir}")
    return results


def evaluate(
    checkpoint_path: str,
    h5_path: str,
    save_dir: str = "eval_results",
    num_sequences: int = 5,
    sequence_length: int = 8,
    stride: int = 8,  # Non-overlapping for test
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    num_actions: Optional[int] = None,
    device: str = "auto",
    batch_size: int = 1,
    # Model architecture params
    latent_dim: int = 512,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_hidden_dim: Optional[int] = None,
    encoder_num_blocks: Optional[int] = None,
    encoder_expansion: Optional[int] = None,
    encoder_projection_rank: Optional[int] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
    enable_fid: bool = True,
    skip_transformer: bool = False,
    temporal_model_type: str = "transformer",
    ff_multiplier: int = 4,
    use_rope: bool = False,
    model_type: str = "temporal",
    teacher_forcing: bool = False,
    **kwargs,
):
    """Run full evaluation: load checkpoint, compute metrics, save reconstructions."""
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

    print(f"\nLoading test data from: {h5_path}")
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
        num_actions=num_actions,
    )

    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = dataset.num_actions

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Sequences: {len(dataset)}")
    print(f"  Gradient dim: {actual_gradient_dim}")
    print(f"  Num actions: {num_actions}")

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
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_blocks=encoder_num_blocks,
        encoder_expansion=encoder_expansion,
        encoder_projection_rank=encoder_projection_rank,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        skip_transformer=skip_transformer,
        temporal_model_type=temporal_model_type,
        ff_multiplier=ff_multiplier,
        use_rope=use_rope,
        model_type=model_type,
        **kwargs,
    )

    print(f"\nGenerating reconstructions for {num_sequences} sequences...")
    print(
        f"  Inference mode: {'teacher-forced' if teacher_forcing else 'autoregressive'}"
    )
    evaluate_and_save_reconstructions(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=Path(save_dir),
        num_sequences=num_sequences,
        enable_fid=enable_fid,
        num_actions=num_actions,
        model_type=model_type,
        teacher_forcing=teacher_forcing,
        action_names=dataset.action_names,
    )


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python evaluate.py <config_path>"

    cfg = OmegaConf.load(sys.argv[1])
    print(f"Loaded config from {sys.argv[1]}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    gradient_dim = data_cfg.get("gradient_dim", None)
    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    evaluate(
        checkpoint_path=eval_cfg.get("checkpoint"),
        h5_path=data_cfg.get("h5_path"),
        save_dir=output_cfg.get("save_dir", "eval_results"),
        num_sequences=eval_cfg.get("num_sequences", 5),
        sequence_length=model_cfg.get("sequence_length", 8),
        stride=model_cfg.get("stride", 8),
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
        num_actions=model_cfg.get("num_actions"),
        device=cfg.get("device", "auto"),
        latent_dim=model_cfg.get("latent_dim", 512),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", None),
        encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", None),
        encoder_num_blocks=model_cfg.get("encoder_num_blocks", None),
        encoder_expansion=model_cfg.get("encoder_expansion", None),
        encoder_projection_rank=model_cfg.get("encoder_projection_rank", None),
        encoder_type=model_cfg.get("encoder_type", "basic"),
        decoder_type=model_cfg.get("decoder_type", "basic"),
        enable_fid=eval_cfg.get("enable_fid", True),
        skip_transformer=model_cfg.get("skip_transformer", False),
        temporal_model_type=model_cfg.get("temporal_model_type", "transformer"),
        ff_multiplier=model_cfg.get("ff_multiplier", 4),
        use_rope=model_cfg.get("use_rope", False),
        model_type=model_cfg.get("model_type", "autoregressive"),
        teacher_forcing=eval_cfg.get("teacher_forcing", False),
    )

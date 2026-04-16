"""
Baseline Evaluation Script for Gradient Inversion

Evaluates baseline methods and the full autoregressive model:
  1. DLG (Zhu et al., 2019) — optimization-based, L2 gradient matching
  2. IG  (Geiping et al., 2020) — cosine similarity + TV regularization
  3. LtI (Wu et al., UAI 2023) — learning-based MLP gradient-to-image
  4. SingleFrame — learned encoder+decoder without temporal transformer
  5. Base (Ours) — full autoregressive gradient inversion model

All methods are evaluated on the same test data with the same metrics
(MSE, PSNR, SSIM, LPIPS, FID, action accuracy).

Usage:
    uv run -m attacker.evaluate_baselines attacker/config/eval_baselines.yaml
"""

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacker.baselines.dlg import DLGBaseline
from attacker.baselines.ig import IGBaseline
from attacker.baselines.learning_to_invert import LearningToInvertModel
from attacker.baselines.single_frame import SingleFrameInversion
from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.metrics import MetricsComputer
from attacker.models.autoregressive_model import AutoregressiveGradientInversion

ACTION_NAMES = ["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp"]


def load_victim_model(
    checkpoint_path: str,
    device: torch.device,
    num_actions: int = 5,
) -> nn.Module:
    """Load the victim ActorCritic model for optimization-based baselines."""
    from victim.model import ActorCritic

    model = ActorCritic(in_channels=3, num_actions=num_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_single_frame_model(
    checkpoint_path: str,
    gradient_dim: int,
    device: torch.device,
    num_actions: int = 5,
    latent_dim: int = 512,
    encoder_type: str = "residual",
    decoder_type: str = "residual",
) -> nn.Module:
    """Load a trained SingleFrameInversion model."""
    model = SingleFrameInversion(
        gradient_dim=gradient_dim,
        latent_dim=latent_dim,
        num_actions=num_actions,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    # The checkpoint may come from the full autoregressive model, which has
    # extra modules (start_token, image_encoder, type_embedding, temporal_model).
    # Filter to only keys present in the SingleFrameInversion model.
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = set(state_dict.keys()) - model_keys
    if skipped:
        print(
            f"  Skipping {len(skipped)} keys not in SingleFrameInversion "
            f"(e.g. {sorted(skipped)[:3]})"
        )

    model.load_state_dict(filtered, strict=True)
    print(
        f"  Loaded SingleFrame from epoch {checkpoint.get('epoch', '?')} "
        f"({len(filtered)}/{len(state_dict)} keys)"
    )
    model.eval()
    return model


def load_lti_model(
    checkpoint_path: str,
    gradient_dim: int,
    device: torch.device,
    num_actions: int = 5,
    hidden_size: int = 3000,
    image_size: int = 84,
    compress_rate: float = 0.1,
    seed: int = 0,
) -> nn.Module:
    """Load a trained LearningToInvertModel (Wu et al., UAI 2023)."""
    model = LearningToInvertModel(
        gradient_dim=gradient_dim,
        hidden_size=hidden_size,
        num_actions=num_actions,
        image_size=image_size,
        compress_rate=compress_rate,
        seed=seed,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    print(
        f"  Loaded LtI model from epoch {checkpoint.get('epoch', '?')} "
        f"({len(state_dict)} keys, compress_rate={compress_rate})"
    )
    model.eval()
    return model


def load_base_model(
    checkpoint_path: str,
    gradient_dim: int,
    device: torch.device,
    num_actions: int = 5,
    latent_dim: int = 1024,
    encoder_type: str = "residual",
    decoder_type: str = "residual",
    num_transformer_layers: int = 6,
    num_heads: int = 8,
) -> nn.Module:
    """Load a trained AutoregressiveGradientInversion model."""
    model = AutoregressiveGradientInversion(
        gradient_dim=gradient_dim,
        latent_dim=latent_dim,
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    print(
        f"  Loaded base model from epoch {checkpoint.get('epoch', '?')} "
        f"({len(state_dict)} keys)"
    )
    model.eval()
    return model


def evaluate_optimization_baseline(
    baseline,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
    num_actions: int = 5,
    enable_fid: bool = False,
):
    """
    Evaluate an optimization-based baseline (DLG or IG).

    These methods are slow (hundreds of optimizer steps per image), so we
    evaluate a limited number of sequences.
    """
    name = baseline.__class__.__name__
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)
    # action_labels = ACTION_NAMES[:num_actions]

    data_iter = iter(dataloader)
    total_time = 0.0
    num_images = 0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"  Only {seq_idx} sequences available")
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        print(f"  [{name}] Sequence {seq_idx + 1}/{num_sequences} ...")
        seq_start = time.perf_counter()
        pred_images, pred_actions = baseline.reconstruct(gradients)
        seq_elapsed = time.perf_counter() - seq_start
        total_time += seq_elapsed

        B, T = images.shape[:2]
        num_images += B * T

        pred_images = pred_images.detach()
        pred_actions = pred_actions.detach()

        metrics.update(pred_images, images, pred_actions, actions)

        # Visualization
        pred_images_cpu = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        # mse = ((pred_images_cpu - images) ** 2).mean().item()
        # psnr = -10.0 * math.log10(max(mse, 1e-10))
        # correct = (pred_labels == actions).sum().item()
        # total = actions.numel()

        T = images.shape[1]
        fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
        if T == 1:
            axes = axes.reshape(2, 1)
        for t in range(T):
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t} GT (a={actions[0, t].item()})", fontsize=8)
            axes[0, t].axis("off")
            pred_img = pred_images_cpu[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(
                f"t={t} Pred (a={pred_labels[0, t].item()})", fontsize=8
            )
            axes[1, t].axis("off")

        plt.tight_layout()
        stem = save_dir / f"{name.lower()}_seq_{seq_idx + 1:03d}"
        plt.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close()

    results = metrics.compute()
    # Add timing metrics
    results["inference_time_per_image_sec"] = (
        total_time / num_images if num_images > 0 else 0.0
    )
    results["inference_time_total_sec"] = total_time
    results["inference_time_num_images"] = num_images
    metrics.save(save_dir / f"{name.lower()}_metrics.json")
    return results


def evaluate_learned_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
    num_actions: int = 5,
    enable_fid: bool = True,
):
    """Evaluate the SingleFrameInversion learned baseline."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)

    data_iter = iter(dataloader)
    evaluated = 0

    # Timing accumulation
    total_time = 0.0
    num_images = 0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"  Only {seq_idx} sequences available")
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        seq_start = time.perf_counter()
        with torch.no_grad():
            pred_images, pred_actions, _, _ = model(gradients)
        seq_elapsed = time.perf_counter() - seq_start
        total_time += seq_elapsed

        B, T = images.shape[:2]
        num_images += B * T

        metrics.update(pred_images, images, pred_actions, actions)

        pred_images_cpu = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        mse = ((pred_images_cpu - images) ** 2).mean().item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        correct = (pred_labels == actions).sum().item()
        total = actions.numel()
        evaluated += 1

        T = images.shape[1]
        fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
        if T == 1:
            axes = axes.reshape(2, 1)
        for t in range(T):
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t} GT (a={actions[0, t].item()})", fontsize=8)
            axes[0, t].axis("off")
            pred_img = pred_images_cpu[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(
                f"t={t} Pred (a={pred_labels[0, t].item()})", fontsize=8
            )
            axes[1, t].axis("off")

        plt.tight_layout()
        stem = save_dir / f"singleframe_seq_{seq_idx + 1:03d}"
        plt.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close()

        print(
            f"  [SingleFrame] Seq {seq_idx + 1} | MSE: {mse:.4f}"
            f" | PSNR: {psnr:.1f} dB | Acc: {100 * correct / total:.1f}%"
        )

    results = metrics.compute()
    # Add timing metrics
    results["inference_time_per_image_sec"] = (
        total_time / num_images if num_images > 0 else 0.0
    )
    results["inference_time_total_sec"] = total_time
    results["inference_time_num_images"] = num_images
    metrics.save(save_dir / "singleframe_metrics.json")
    return results


def evaluate_lti_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
    num_actions: int = 5,
    enable_fid: bool = True,
):
    """Evaluate the Learning-to-Invert (Wu et al., UAI 2023) baseline."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)

    data_iter = iter(dataloader)
    total_time = 0.0
    num_images = 0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"  Only {seq_idx} sequences available")
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        seq_start = time.perf_counter()
        with torch.no_grad():
            pred_images, pred_actions = model.reconstruct(gradients)
        seq_elapsed = time.perf_counter() - seq_start
        total_time += seq_elapsed

        B, T = images.shape[:2]
        num_images += B * T

        pred_images = pred_images.detach()
        pred_actions = pred_actions.detach()

        metrics.update(pred_images, images, pred_actions, actions)

        pred_images_cpu = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        mse = ((pred_images_cpu - images) ** 2).mean().item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        correct = (pred_labels == actions).sum().item()
        total = actions.numel()

        T = images.shape[1]
        fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
        if T == 1:
            axes = axes.reshape(2, 1)
        for t in range(T):
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t} GT (a={actions[0, t].item()})", fontsize=8)
            axes[0, t].axis("off")
            pred_img = pred_images_cpu[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(
                f"t={t} Pred (a={pred_labels[0, t].item()})", fontsize=8
            )
            axes[1, t].axis("off")

        plt.tight_layout()
        stem = save_dir / f"lti_seq_{seq_idx + 1:03d}"
        plt.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close()

        print(
            f"  [LtI] Seq {seq_idx + 1} | MSE: {mse:.4f}"
            f" | PSNR: {psnr:.1f} dB | Acc: {100 * correct / total:.1f}%"
        )

    results = metrics.compute()
    results["inference_time_per_image_sec"] = (
        total_time / num_images if num_images > 0 else 0.0
    )
    results["inference_time_total_sec"] = total_time
    results["inference_time_num_images"] = num_images
    metrics.save(save_dir / "lti_metrics.json")
    return results


def evaluate_base_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 5,
    num_actions: int = 5,
    enable_fid: bool = True,
    use_flash_attention: bool = True,
):
    """Evaluate the full AutoregressiveGradientInversion model."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)

    data_iter = iter(dataloader)
    evaluated = 0

    total_time = 0.0
    num_images = 0

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            print(f"  Only {seq_idx} sequences available")
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        seq_start = time.perf_counter()
        with torch.no_grad():
            pred_images, pred_actions, _, _ = model(
                gradients,
                teacher_forcing=False,
                use_flash_attention=use_flash_attention,
            )
        seq_elapsed = time.perf_counter() - seq_start
        total_time += seq_elapsed

        B, T = images.shape[:2]
        num_images += B * T

        metrics.update(pred_images, images, pred_actions, actions)

        pred_images_cpu = pred_images.cpu()
        pred_labels = pred_actions.argmax(dim=-1).cpu()

        mse = ((pred_images_cpu - images) ** 2).mean().item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        correct = (pred_labels == actions).sum().item()
        total = actions.numel()
        evaluated += 1

        T = images.shape[1]
        fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
        if T == 1:
            axes = axes.reshape(2, 1)
        for t in range(T):
            axes[0, t].imshow(images[0, t].permute(1, 2, 0).numpy())
            axes[0, t].set_title(f"t={t} GT (a={actions[0, t].item()})", fontsize=8)
            axes[0, t].axis("off")
            pred_img = pred_images_cpu[0, t].permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, t].imshow(pred_img)
            axes[1, t].set_title(
                f"t={t} Pred (a={pred_labels[0, t].item()})", fontsize=8
            )
            axes[1, t].axis("off")

        plt.tight_layout()
        stem = save_dir / f"base_seq_{seq_idx + 1:03d}"
        plt.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close()

        print(
            f"  [Base] Seq {seq_idx + 1} | MSE: {mse:.4f}"
            f" | PSNR: {psnr:.1f} dB | Acc: {100 * correct / total:.1f}%"
        )

    results = metrics.compute()
    results["inference_time_per_image_sec"] = (
        total_time / num_images if num_images > 0 else 0.0
    )
    results["inference_time_total_sec"] = total_time
    results["inference_time_num_images"] = num_images
    metrics.save(save_dir / "base_metrics.json")
    return results


METRIC_COLUMNS = [
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "fid",
    "action_accuracy",
    "inference_time_per_image_sec",
]


def save_method_csv(name: str, results: dict, path: Path):
    """Write a single method's metrics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "std"])
        for key in METRIC_COLUMNS:
            val = results.get(key, "")
            std = results.get(f"{key}_std", "")
            writer.writerow([key, val, std])
        writer.writerow(["num_images", results.get("num_images", 0), ""])
        writer.writerow(
            [
                "inference_time_per_image_sec",
                results.get("inference_time_per_image_sec", ""),
                "",
            ]
        )
        writer.writerow(
            [
                "inference_time_total_sec",
                results.get("inference_time_total_sec", ""),
                "",
            ]
        )
        writer.writerow(
            [
                "inference_time_num_images",
                results.get("inference_time_num_images", ""),
                "",
            ]
        )
    print(f"  Metrics CSV saved to: {path}")


def save_comparison_csv(all_results: Dict[str, dict], path: Path):
    """Write the combined comparison table to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(all_results.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + METRIC_COLUMNS)
        for name in methods:
            r = all_results[name]
            row = [name] + [r.get(k, "") for k in METRIC_COLUMNS]
            writer.writerow(row)
    print(f"Comparison CSV saved to: {path}")


def evaluate_baselines(
    h5_path: str,
    victim_checkpoint: str,
    save_dir: str = "eval_results/baselines",
    singleframe_checkpoint: str = "",
    lti_checkpoint: str = "",
    base_checkpoint: str = "",
    num_sequences: int = 5,
    sequence_length: int = 8,
    stride: int = 8,
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    device: str = "auto",
    # DLG hyperparameters (defaults match official mit-han-lab/dlg)
    dlg_iterations: int = 300,
    dlg_lr: float = 1.0,
    dlg_restarts: int = 1,
    # IG hyperparameters (defaults match official JonasGeiping/invertinggradients)
    ig_iterations: int = 4800,
    ig_lr: float = 0.1,
    ig_tv_weight: float = 1e-1,
    ig_restarts: int = 1,
    ig_signed: bool = False,
    # LtI hyperparameters (Wu et al., UAI 2023)
    lti_hidden_size: int = 3000,
    lti_compress_rate: float = 0.1,
    lti_seed: int = 0,
    # SingleFrame architecture (must match its training config)
    sf_latent_dim: int = 1024,
    sf_encoder_type: str = "residual",
    sf_decoder_type: str = "residual",
    # Base model architecture
    base_latent_dim: int = 1024,
    base_encoder_type: str = "residual",
    base_decoder_type: str = "residual",
    base_num_transformer_layers: int = 6,
    base_num_heads: int = 8,
    use_flash_attention: bool = True,
    # Which methods to run
    run_dlg: bool = True,
    run_ig: bool = True,
    run_lti: bool = False,
    run_singleframe: bool = True,
    run_base: bool = False,
    enable_fid: bool = False,
):
    """Run evaluation for all enabled baselines."""
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

    # Load dataset
    print(f"\nLoading test data from: {h5_path}")
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
    )
    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = len(dataset.actions.unique())

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Sequences: {len(dataset)}")
    print(f"  Gradient dim: {actual_gradient_dim}")
    print(f"  Num actions: {num_actions}")

    save_dir = Path(save_dir)
    all_results = {}

    # ------------------------------------------------------------------
    # 1. DLG
    # ------------------------------------------------------------------
    if run_dlg:
        print(f"\n{'=' * 60}")
        print("Evaluating DLG baseline ...")
        print(f"{'=' * 60}")

        victim = load_victim_model(victim_checkpoint, device, num_actions)
        dlg = DLGBaseline(
            victim_model=victim,
            gradient_dim=actual_gradient_dim,
            num_actions=num_actions,
            num_iterations=dlg_iterations,
            lr=dlg_lr,
            num_restarts=dlg_restarts,
        )

        dlg_results = evaluate_optimization_baseline(
            dlg,
            dataloader,
            device,
            save_dir / "dlg",
            num_sequences=num_sequences,
            num_actions=num_actions,
            enable_fid=enable_fid,
        )
        all_results["DLG"] = dlg_results
        save_method_csv("DLG", dlg_results, save_dir / "dlg" / "results.csv")
        del dlg, victim
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 2. Inverting Gradients (IG)
    # ------------------------------------------------------------------
    if run_ig:
        print(f"\n{'=' * 60}")
        print("Evaluating IG baseline ...")
        print(f"{'=' * 60}")

        victim = load_victim_model(victim_checkpoint, device, num_actions)
        ig = IGBaseline(
            victim_model=victim,
            gradient_dim=actual_gradient_dim,
            num_actions=num_actions,
            num_iterations=ig_iterations,
            lr=ig_lr,
            tv_weight=ig_tv_weight,
            num_restarts=ig_restarts,
            signed=ig_signed,
        )

        ig_results = evaluate_optimization_baseline(
            ig,
            dataloader,
            device,
            save_dir / "ig",
            num_sequences=num_sequences,
            num_actions=num_actions,
            enable_fid=enable_fid,
        )
        all_results["IG"] = ig_results
        save_method_csv("IG", ig_results, save_dir / "ig" / "results.csv")
        del ig, victim
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 3. Learning to Invert (LtI)
    # ------------------------------------------------------------------
    if run_lti and lti_checkpoint:
        print(f"\n{'=' * 60}")
        print("Evaluating Learning-to-Invert (LtI) baseline ...")
        print(f"{'=' * 60}")

        lti_model = load_lti_model(
            lti_checkpoint,
            gradient_dim=actual_gradient_dim,
            device=device,
            num_actions=num_actions,
            hidden_size=lti_hidden_size,
            compress_rate=lti_compress_rate,
            seed=lti_seed,
        )

        lti_results = evaluate_lti_baseline(
            lti_model,
            dataloader,
            device,
            save_dir / "lti",
            num_sequences=num_sequences,
            num_actions=num_actions,
            enable_fid=enable_fid,
        )
        all_results["LtI"] = lti_results
        save_method_csv("LtI", lti_results, save_dir / "lti" / "results.csv")
        del lti_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    elif run_lti:
        print("\nSkipping LtI: no checkpoint provided")

    # ------------------------------------------------------------------
    # 4. Single-Frame Learned
    # ------------------------------------------------------------------
    if run_singleframe and singleframe_checkpoint:
        print(f"\n{'=' * 60}")
        print("Evaluating SingleFrame learned baseline ...")
        print(f"{'=' * 60}")

        sf_model = load_single_frame_model(
            singleframe_checkpoint,
            gradient_dim=actual_gradient_dim,
            device=device,
            num_actions=num_actions,
            latent_dim=sf_latent_dim,
            encoder_type=sf_encoder_type,
            decoder_type=sf_decoder_type,
        )

        sf_results = evaluate_learned_baseline(
            sf_model,
            dataloader,
            device,
            save_dir / "singleframe",
            num_sequences=num_sequences,
            num_actions=num_actions,
            enable_fid=enable_fid,
        )
        all_results["SingleFrame"] = sf_results
        save_method_csv(
            "SingleFrame", sf_results, save_dir / "singleframe" / "results.csv"
        )
        del sf_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    elif run_singleframe:
        print("\nSkipping SingleFrame: no checkpoint provided")

    # ------------------------------------------------------------------
    # 5. Base (Autoregressive) Model — Ours
    # ------------------------------------------------------------------
    if run_base and base_checkpoint:
        print(f"\n{'=' * 60}")
        print("Evaluating Base autoregressive model (Ours) ...")
        print(f"{'=' * 60}")

        base_model = load_base_model(
            base_checkpoint,
            gradient_dim=actual_gradient_dim,
            device=device,
            num_actions=num_actions,
            latent_dim=base_latent_dim,
            encoder_type=base_encoder_type,
            decoder_type=base_decoder_type,
            num_transformer_layers=base_num_transformer_layers,
            num_heads=base_num_heads,
        )

        base_results = evaluate_base_model(
            base_model,
            dataloader,
            device,
            save_dir / "base",
            num_sequences=num_sequences,
            num_actions=num_actions,
            enable_fid=enable_fid,
            use_flash_attention=use_flash_attention,
        )
        all_results["Base (Ours)"] = base_results
        save_method_csv(
            "Base", base_results, save_dir / "base" / "results.csv"
        )
        del base_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    elif run_base:
        print("\nSkipping Base model: no checkpoint provided")

    # ------------------------------------------------------------------
    # Combined comparison CSV
    # ------------------------------------------------------------------
    if all_results:
        save_comparison_csv(all_results, save_dir / "comparison.csv")

    return all_results


# ============================================================================
# CLI entry point
# ============================================================================


if __name__ == "__main__":
    assert len(sys.argv) > 1, (
        "Usage: uv run -m attacker.evaluate_baselines <config_path>"
    )

    cfg = OmegaConf.load(sys.argv[1])
    print(f"Loaded config from {sys.argv[1]}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})
    dlg_cfg = cfg.get("dlg", {})
    ig_cfg = cfg.get("ig", {})
    lti_cfg = cfg.get("lti", {})
    sf_cfg = cfg.get("singleframe", {})
    base_cfg = cfg.get("base", {})

    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    evaluate_baselines(
        h5_path=data_cfg.get("h5_path"),
        victim_checkpoint=eval_cfg.get("victim_checkpoint"),
        save_dir=output_cfg.get("save_dir", "eval_results/baselines"),
        singleframe_checkpoint=sf_cfg.get("checkpoint", ""),
        lti_checkpoint=lti_cfg.get("checkpoint", ""),
        base_checkpoint=base_cfg.get("checkpoint", ""),
        num_sequences=eval_cfg.get("num_sequences", 5),
        sequence_length=model_cfg.get("sequence_length", 8),
        stride=model_cfg.get("stride", 8),
        gradient_dim=data_cfg.get("gradient_dim", None),
        gradient_layers=gradient_layers,
        device=cfg.get("device", "auto"),
        # DLG
        dlg_iterations=dlg_cfg.get("num_iterations", 300),
        dlg_lr=dlg_cfg.get("lr", 1.0),
        dlg_restarts=dlg_cfg.get("num_restarts", 1),
        # IG
        ig_iterations=ig_cfg.get("num_iterations", 4800),
        ig_lr=ig_cfg.get("lr", 0.1),
        ig_tv_weight=ig_cfg.get("tv_weight", 1e-1),
        ig_restarts=ig_cfg.get("num_restarts", 1),
        ig_signed=ig_cfg.get("signed", False),
        # LtI
        lti_hidden_size=lti_cfg.get("hidden_size", 3000),
        lti_compress_rate=lti_cfg.get("compress_rate", 0.1),
        lti_seed=lti_cfg.get("seed", 0),
        # SingleFrame architecture
        sf_latent_dim=sf_cfg.get("latent_dim", 1024),
        sf_encoder_type=sf_cfg.get("encoder_type", "residual"),
        sf_decoder_type=sf_cfg.get("decoder_type", "residual"),
        # Base model architecture
        base_latent_dim=base_cfg.get("latent_dim", 1024),
        base_encoder_type=base_cfg.get("encoder_type", "residual"),
        base_decoder_type=base_cfg.get("decoder_type", "residual"),
        base_num_transformer_layers=base_cfg.get("num_transformer_layers", 6),
        base_num_heads=base_cfg.get("num_heads", 8),
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        # Flags
        run_dlg=eval_cfg.get("run_dlg", True),
        run_ig=eval_cfg.get("run_ig", True),
        run_lti=eval_cfg.get("run_lti", False),
        run_singleframe=eval_cfg.get("run_singleframe", True),
        run_base=eval_cfg.get("run_base", False),
        enable_fid=eval_cfg.get("enable_fid", False),
    )

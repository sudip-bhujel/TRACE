"""SingleFrameInversion baseline training (encoder + decoder; no temporal model)."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from attacker.baselines.single_frame import SingleFrameInversion
from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.loss import TemporalCombinedLoss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 1,
) -> dict:
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

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            pred_images, pred_actions, _, _ = model(gradients)
            loss, loss_dict = criterion(pred_images, images, pred_actions, actions)
            loss = loss / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue

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

        for k, v in loss_dict.items():
            total_losses[k] += v

        pred_labels = pred_actions.argmax(dim=-1).flatten()
        correct += (pred_labels == actions.flatten()).sum().item()
        total += actions.numel()

        pbar.set_postfix(
            loss=f"{loss_dict['total']:.4f}", acc=f"{100 * correct / total:.1f}%"
        )

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
    model.eval()
    total_losses = defaultdict(float)
    correct = 0
    total = 0

    with torch.no_grad():
        for gradients, images, actions in dataloader:
            gradients = gradients.to(device)
            images = images.to(device)
            actions = actions.to(device)

            pred_images, pred_actions, _, _ = model(gradients)
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


def train(
    h5_path: str,
    num_epochs: int = 50,
    batch_size: int = 2,
    accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    device: Union[str, torch.device] = "auto",
    save_dir: Union[str, Path] = "ckpts/singleframe",
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    sequence_length: int = 8,
    latent_dim: int = 512,
    stride: int = 4,
    encoder_type: str = "residual",
    decoder_type: str = "residual",
    dropout: float = 0.1,
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
    action_weight: float = 0.1,
    temporal_weight: float = 0.0,
    lpips_weight: float = 0.0,
    lr_schedule: str = "cosine",
    min_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 0,
    num_workers: int = 4,
    use_wandb: bool = False,
):
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

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
    )
    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = len(dataset.actions.unique())

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Gradient dim: {actual_gradient_dim}, Actions: {num_actions}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = SingleFrameInversion(
        gradient_dim=actual_gradient_dim,
        latent_dim=latent_dim,
        num_actions=num_actions,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = TemporalCombinedLoss(
        mse_weight=mse_weight,
        l1_weight=l1_weight,
        action_weight=action_weight,
        temporal_weight=temporal_weight,
        lpips_weight=lpips_weight,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    else:
        scheduler = None

    if warmup_epochs > 0 and scheduler is not None:
        base_scheduler = scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epochs
                ),
                base_scheduler,
            ],
            milestones=[warmup_epochs],
        )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("\nStarting SingleFrame baseline training...")

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

        print(
            f"Epoch {epoch}: Train {train_loss['total']:.4f}"
            f" (MSE: {train_loss['mse']:.4f}) | Val {val_loss['total']:.4f}"
            f" (MSE: {val_loss['mse']:.4f}) | Acc: {val_loss['accuracy']:.1f}%"
        )

        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss["total"],
                    "train/mse": train_loss["mse"],
                    "train/accuracy": train_loss["accuracy"],
                    "val/loss": val_loss["total"],
                    "val/mse": val_loss["mse"],
                    "val/accuracy": val_loss["accuracy"],
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if scheduler is not None:
            scheduler.step()

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

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        save_dir / "final_model.pt",
    )

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    assert len(sys.argv) > 1, (
        "Usage: uv run -m attacker.train_singleframe <config_path>"
    )

    cfg_loaded = OmegaConf.load(sys.argv[1])
    assert isinstance(cfg_loaded, DictConfig), "Config root must be a mapping"
    cfg: DictConfig = cfg_loaded
    print(f"Loaded config from {sys.argv[1]}")

    data_cfg = cast(Dict[str, Any], cfg.get("data", {}))
    model_cfg = cast(Dict[str, Any], cfg.get("model", {}))
    training_cfg = cast(Dict[str, Any], cfg.get("training", {}))
    output_cfg = cast(Dict[str, Any], cfg.get("output", {}))
    loss_cfg = cast(Dict[str, Any], cfg.get("loss", {}))

    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    wandb_cfg = cast(Dict[str, Any], cfg.get("wandb", {}))
    if wandb_cfg.get("enabled", False):
        wandb.login()
        wandb_container = OmegaConf.to_container(cfg, resolve=True)
        wandb_config: Dict[str, Any] = {}
        if isinstance(wandb_container, dict):
            wandb_config = {str(k): v for k, v in wandb_container.items()}
        wandb.init(
            project=wandb_cfg.get("project", "gradient-inversion"),
            name=wandb_cfg.get("name", "singleframe-baseline"),
            entity=wandb_cfg.get("entity", "gradinversion"),
            config=wandb_config,
            tags=wandb_cfg.get("tags", []),
        )

    train(
        h5_path=data_cfg.get("h5_path", "trajectory_data/gradients.h5"),
        num_epochs=training_cfg.get("num_epochs", 50),
        batch_size=training_cfg.get("batch_size", 2),
        accumulation_steps=training_cfg.get("accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        device=cfg.get("device", "auto"),
        save_dir=output_cfg.get("save_dir", "ckpts/singleframe"),
        gradient_dim=data_cfg.get("gradient_dim", None),
        gradient_layers=gradient_layers,
        sequence_length=model_cfg.get("sequence_length", 8),
        latent_dim=model_cfg.get("latent_dim", 512),
        stride=model_cfg.get("stride", 4),
        encoder_type=model_cfg.get("encoder_type", "residual"),
        decoder_type=model_cfg.get("decoder_type", "residual"),
        dropout=model_cfg.get("dropout", 0.1),
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        l1_weight=loss_cfg.get("l1_weight", 0.5),
        action_weight=loss_cfg.get("action_weight", 0.1),
        temporal_weight=0.0,
        lpips_weight=loss_cfg.get("lpips_weight", 0.0),
        lr_schedule=training_cfg.get("lr_schedule", "cosine"),
        min_lr=training_cfg.get("min_lr", 1e-6),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        num_workers=training_cfg.get("num_workers", 4),
        use_wandb=wandb_cfg.get("enabled", False),
    )

"""
Training script for the temporal gradient inversion model.
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.evaluate import evaluate
from attacker.evaluation.loss import TemporalCombinedLoss
from attacker.models.autoregressive_model import AutoregressiveGradientInversion
from attacker.models.model import TemporalGradientInversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 1,
    use_flash_attention: bool = True,
    gradient_noise_scale: float = 0.0,
    gradient_mask_ratio: float = 0.0,
    model_type: str = "temporal",
    sampling_prob: float = 0.0,
    rollout_steps: int = 0,
    rollout_weight: float = 0.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_losses = defaultdict(float)
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for gradients, images, actions in pbar:
        gradients = gradients.to(device)
        images = images.to(device)
        actions = actions.to(device)

        # Augmentation: gradient noise
        if gradient_noise_scale > 0:
            grad_std = gradients.std(dim=-1, keepdim=True).clamp(min=1e-6)
            gradients = gradients + gradient_noise_scale * grad_std * torch.randn_like(
                gradients
            )

        # Augmentation: gradient masking
        if gradient_mask_ratio > 0:
            mask = (
                torch.rand(gradients.shape[:-1], device=device).unsqueeze(-1)
                > gradient_mask_ratio
            ).float()
            gradients = gradients * mask

        # Forward pass with mixed precision (Flash Attention auto-enabled via SDPA)
        # For autoregressive models, a single model() call computes both the
        # primary output AND rollout (if rollout_steps > 0). This is required
        # for DDP: calling the model twice causes DDP's per-parameter backward
        # hooks to fire twice for shared params (e.g. start_token).
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            if model_type == "autoregressive":
                pred_images, pred_actions, latents, ro_result = model(
                    gradients,
                    images=images,
                    teacher_forcing=True,
                    sampling_prob=sampling_prob,
                    rollout_steps=rollout_steps if rollout_weight > 0 else 0,
                    use_flash_attention=use_flash_attention,
                )
            else:
                pred_images, pred_actions, latents, ro_result = model(
                    gradients, use_flash_attention=use_flash_attention
                )
            loss, loss_dict = criterion(
                pred_images, images, pred_actions, actions, latents=latents
            )

            # Add rollout loss if present
            if ro_result is not None and rollout_weight > 0:
                ro_imgs, ro_acts, ro_lats, t_start = ro_result
                t_end = t_start + ro_imgs.shape[1]
                ro_gt_imgs = images[:, t_start:t_end]
                ro_gt_acts = actions[:, t_start:t_end]
                ro_loss, _ = criterion(
                    ro_imgs,
                    ro_gt_imgs,
                    ro_acts,
                    ro_gt_acts,
                    latents=ro_lats,
                )
                loss = loss + rollout_weight * ro_loss
                loss_dict["rollout"] = ro_loss.item()

            loss = loss / accumulation_steps

        # NaN detection - skip bad batches to prevent training corruption
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (pbar.n + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                # Check for NaN gradients before stepping
                valid_grads = True
                for param in model.parameters():
                    if param.grad is not None and (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    ):
                        valid_grads = False
                        break
                if valid_grads:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=0.5
                    )  # Tighter clipping
                    scaler.step(optimizer)
                else:
                    print(
                        "Warning: NaN/Inf gradients detected, skipping optimizer step"
                    )
                scaler.update()
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.5
                )  # Tighter clipping
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
    use_flash_attention: bool = True,
    model_type: str = "temporal",
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

            if model_type == "autoregressive":
                pred_images, pred_actions, latents, _ = model(
                    gradients,
                    images=images,
                    teacher_forcing=True,
                    use_flash_attention=use_flash_attention,
                )
            else:
                pred_images, pred_actions, latents, _ = model(
                    gradients, use_flash_attention=use_flash_attention
                )
            loss, loss_dict = criterion(
                pred_images, images, pred_actions, actions, latents=latents
            )

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
    use_flash_attention: bool = True,
    model_type: str = "temporal",
):
    """Save sample reconstructions showing temporal progression."""
    plt.rcParams["font.family"] = "serif"
    model.eval()

    gradients, images, actions = next(iter(dataloader))
    num_sequences = min(num_sequences, len(gradients))
    gradients = gradients[:num_sequences].to(device)
    images = images[:num_sequences]
    actions = actions[:num_sequences]

    with torch.no_grad():
        if model_type == "autoregressive":
            pred_images, pred_actions, _, _ = model(
                gradients,
                images=images.to(device),
                teacher_forcing=True,
                use_flash_attention=use_flash_attention,
            )
        else:
            pred_images, pred_actions, _, _ = model(
                gradients, use_flash_attention=use_flash_attention
            )

    pred_images = pred_images.cpu()
    pred_labels = pred_actions.argmax(dim=-1).cpu()

    T = images.shape[1]

    # Create figure: 2 rows per sequence (GT, Pred), T columns
    fig, axes = plt.subplots(
        2 * num_sequences, T, figsize=(2 * T, 4 * num_sequences), squeeze=False
    )

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


def train(
    h5_path: str = "trajectory_data/gradients.h5",
    num_epochs: int = 50,
    batch_size: int = 2,
    accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    device: Union[str, torch.device] = "auto",
    save_dir: Union[str, Path] = "ckpts/attacker_temporal",
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    sequence_length: int = 8,
    latent_dim: int = 512,
    stride: int = 4,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_hidden_dim: Optional[int] = None,
    encoder_num_blocks: Optional[int] = None,
    encoder_expansion: Optional[int] = None,
    encoder_projection_rank: Optional[int] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
    dropout: float = 0.1,
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
    action_weight: float = 0.1,
    temporal_weight: float = 0.0,
    lpips_weight: float = 0.0,
    lpips_net: str = "vgg",
    use_flash_attention: bool = False,
    use_wandb: bool = False,
    # New quick-win parameters
    gradient_noise_scale: float = 0.0,
    warmup_epochs: int = 0,
    lr_schedule: str = "none",
    min_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    num_workers: int = 4,
    pretrained_checkpoint: str = "",
    finetune_fraction: float = 1.0,
    # Ablation: skip transformer
    skip_transformer: bool = False,
    is_causal: bool = True,
    # Temporal model selection
    temporal_model_type: str = "transformer",
    ff_multiplier: int = 4,
    use_rope: bool = False,
    latent_temporal_weight: float = 0.0,
    # Ablation: cap dataset size
    max_sequences: Optional[int] = None,
    # Temporal dropout: fraction of timesteps to mask (0.0 = disabled)
    gradient_mask_ratio: float = 0.0,
    # Model type: "temporal" (existing) or "autoregressive" (new)
    model_type: str = "temporal",
    # Scheduled sampling (autoregressive only)
    scheduled_sampling_start: float = 0.0,
    scheduled_sampling_end: float = 0.0,
    scheduled_sampling_warmup: int = 5,
    # Short rollout loss (autoregressive only)
    rollout_steps: int = 0,
    rollout_weight: float = 0.0,
):
    """
    Main training function for temporal model.

    Args:
        use_flash_attention: If True, explicitly enable Flash Attention via
            torch.backends.cuda.sdp_kernel on CUDA (PyTorch 2.0+ required).
        use_wandb: If True, enable Weights & Biases logging.

    Supports DDP (Distributed Data Parallel) training when launched with torchrun:
        torchrun --standalone --nproc_per_node=2 -m ppo.attacker_temporal --config ...
    """

    # =========================================================================
    # DDP Setup
    # =========================================================================
    ddp = int(os.environ.get("RANK", -1)) != -1  # Is this a DDP run?
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # Only rank 0 logs/saves
        seed_offset = ddp_rank  # Each process gets a different seed
    else:
        # Single GPU / CPU training
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
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

    # Set seed for reproducibility
    torch.manual_seed(42 + seed_offset)

    if master_process:
        print(f"Using device: {device}")
        if ddp:
            print(
                f"DDP enabled: world_size={ddp_world_size}, accumulation_steps={accumulation_steps}"
            )

    # Check Flash Attention availability
    if device.type == "cuda" and hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        if master_process:
            print(f"Flash Attention available: {flash_enabled}")

    # Create save directory (only master)
    save_dir = Path(save_dir)
    if master_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
        max_sequences=max_sequences,
    )
    # Use effective gradient dim (handles both layer selection and dim truncation)
    actual_gradient_dim = dataset.effective_gradient_dim

    # Stratified subsampling for few-shot fine-tuning
    if finetune_fraction < 1.0:
        rng = random.Random(42)
        # Group sequence indices by episode
        ep_to_seqs = {}
        for seq_idx, (start_idx, _end_idx) in enumerate(dataset.sequence_indices):
            ep_id = dataset.episode_ids[start_idx].item()
            ep_to_seqs.setdefault(ep_id, []).append(seq_idx)

        # Keep finetune_fraction of sequences from EACH episode
        keep_indices = []
        for ep_id in sorted(ep_to_seqs.keys()):
            seqs = ep_to_seqs[ep_id]
            n_keep = max(1, int(len(seqs) * finetune_fraction))
            keep_indices.extend(rng.sample(seqs, n_keep))

        if master_process:
            print(f"\n[FEW-SHOT] Keeping {finetune_fraction * 100:.0f}% per episode:")
            print(f"  {len(keep_indices)} / {len(dataset)} sequences")
            print(f"  Episodes: {len(ep_to_seqs)}")

        # Train on ALL selected data; use 10% overlap as val for monitoring
        train_dataset = torch.utils.data.Subset(dataset, keep_indices)
        val_size = max(1, int(len(keep_indices) * 0.1))
        val_indices = rng.sample(keep_indices, val_size)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
    else:
        # Normal 95/5 split
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    if master_process:
        print("\nDataset split:")
        print(f"  Train: {len(train_dataset)} sequences")
        print(f"  Val: {len(val_dataset)} sequences")

    # Create dataloaders with DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get number of actions from dataset
    num_actions = len(dataset.actions.unique())
    if master_process:
        print(f"  Number of actions: {num_actions}")

    # Create model (Flash Attention auto-enabled by PyTorch 2.0+ with mixed precision)
    if model_type == "autoregressive":
        model: nn.Module = AutoregressiveGradientInversion(
            gradient_dim=actual_gradient_dim,
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
            dropout=dropout,
            temporal_model_type=temporal_model_type,
            ff_multiplier=ff_multiplier,
            use_rope=use_rope,
        ).to(device)
    else:
        model = TemporalGradientInversion(
            gradient_dim=actual_gradient_dim,
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
            dropout=dropout,
            skip_transformer=skip_transformer,
            is_causal=is_causal,
            temporal_model_type=temporal_model_type,
            ff_multiplier=ff_multiplier,
            use_rope=use_rope,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f"\nModel parameters: {num_params:,}")

    # Load pretrained checkpoint for fine-tuning
    if pretrained_checkpoint:
        ckpt = torch.load(
            pretrained_checkpoint, map_location=device, weights_only=False
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        if master_process:
            pretrained_epoch = ckpt.get("epoch", "?")
            print(
                f"\n[FINE-TUNING] Loaded pretrained weights from {pretrained_checkpoint}"
            )
            print(f"  Pretrained epoch: {pretrained_epoch}")

    # Wrap model with DDP
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            find_unused_parameters=(model_type == "autoregressive"),
        )
    raw_model = cast(nn.Module, model.module if ddp else model)  # Unwrap for saving

    # Loss and optimizer
    criterion = TemporalCombinedLoss(
        mse_weight=mse_weight,
        l1_weight=l1_weight,
        action_weight=action_weight,
        temporal_weight=temporal_weight,
        lpips_weight=lpips_weight,
        lpips_net=lpips_net,
        latent_temporal_weight=latent_temporal_weight,
    ).to(device)  # Move to device for VGG buffers

    if master_process:
        print("\nLoss weights:")
        print(f"  MSE: {mse_weight}, L1: {l1_weight}, Action: {action_weight}")
        print(f"  LPIPS: {lpips_weight}, Temporal: {temporal_weight}")
        if gradient_mask_ratio > 0:
            print(
                f"\nTemporal dropout: masking {gradient_mask_ratio * 100:.0f}% of timesteps"
            )

    optimizer = optim.AdamW(
        raw_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    if lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
        if master_process:
            print(f"\nUsing cosine annealing LR schedule (min_lr={min_lr})")
    else:
        scheduler = None
        if master_process:
            print("\nNo LR scheduling")

    # LR warmup wrapper
    if warmup_epochs > 0 and scheduler is not None:
        from_scheduler = scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epochs
                ),
                from_scheduler,
            ],
            milestones=[warmup_epochs],
        )
        if master_process:
            print(f"Added {warmup_epochs} epoch LR warmup")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler and master_process:
        print("Using mixed precision training (fp16)")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if master_process:
        print("\n" + "=" * 60)
        print("Starting temporal training...")
        print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Compute scheduled sampling probability for this epoch
        if epoch <= scheduled_sampling_warmup:
            sampling_prob = 0.0
        else:
            progress = (epoch - scheduled_sampling_warmup) / max(
                1, num_epochs - scheduled_sampling_warmup
            )
            sampling_prob = (
                scheduled_sampling_start
                + (scheduled_sampling_end - scheduled_sampling_start) * progress
            )

        if master_process and model_type == "autoregressive" and sampling_prob > 0:
            print(f"  Scheduled sampling prob: {sampling_prob:.3f}")

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            use_flash_attention=use_flash_attention,
            gradient_noise_scale=gradient_noise_scale,
            gradient_mask_ratio=gradient_mask_ratio,
            model_type=model_type,
            sampling_prob=sampling_prob,
            rollout_steps=rollout_steps,
            rollout_weight=rollout_weight,
        )
        train_losses.append(train_loss["total"])

        # Log train metrics (master only)
        if master_process:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss['total']:.4f} (MSE: {train_loss['mse']:.4f}, L1: {train_loss['l1']:.4f}, Act: {train_loss['action']:.4f}, Temp: {train_loss['temporal']:.4f}, LTemp: {train_loss.get('latent_temporal', 0):.4f}, LPIPS: {train_loss['lpips']:.4f}) | Acc: {train_loss['accuracy']:.2f}%"
            )
            if use_wandb:
                log_dict = {
                    "train/loss": train_loss["total"],
                    "train/mse_loss": train_loss["mse"],
                    "train/l1_loss": train_loss["l1"],
                    "train/action_loss": train_loss["action"],
                    "train/temporal_loss": train_loss["temporal"],
                    "train/lpips_loss": train_loss["lpips"],
                    "train/accuracy": train_loss["accuracy"],
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict)
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            use_flash_attention=use_flash_attention,
            model_type=model_type,
        )
        val_losses.append(val_loss["total"])

        # Log val metrics (master only)
        if master_process:
            print(
                f"Epoch {epoch}: Val Loss: {val_loss['total']:.4f} (MSE: {val_loss['mse']:.4f}, L1: {val_loss['l1']:.4f}, Act: {val_loss['action']:.4f}, Temp: {val_loss['temporal']:.4f}, LTemp: {val_loss.get('latent_temporal', 0):.4f}, LPIPS: {val_loss['lpips']:.4f}) | Acc: {val_loss['accuracy']:.2f}%"
            )
            if use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss["total"],
                        "val/mse_loss": val_loss["mse"],
                        "val/l1_loss": val_loss["l1"],
                        "val/action_loss": val_loss["action"],
                        "val/temporal_loss": val_loss["temporal"],
                        "val/lpips_loss": val_loss["lpips"],
                        "val/accuracy": val_loss["accuracy"],
                    }
                )

        if scheduler is not None:
            scheduler.step()

        # Save best model (master only)
        if val_loss["total"] < best_val_loss and master_process:
            best_val_loss = val_loss["total"]
            state_dict = raw_model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "decoder_type": decoder_type,
                },
                save_dir / "best_model.pt",
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save reconstructions periodically (master only)
        if master_process and (epoch % 5 == 0 or epoch == 1):
            save_temporal_reconstructions(
                raw_model,
                val_loader,
                device,
                save_dir / f"recon_epoch_{epoch:03d}.png",
                use_flash_attention=use_flash_attention,
                model_type=model_type,
            )

    # Save final model (master only)
    if master_process:
        state_dict = raw_model.state_dict()
        torch.save(
            {
                "epoch": num_epochs,
                "model_state_dict": state_dict,
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
        # plt.title("Temporal Model Training Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "training_curves.png", dpi=150)
        plt.close()

        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print("=" * 60)

    # DDP cleanup
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python train.py <config_path>"

    cfg_loaded = OmegaConf.load(sys.argv[1])
    assert isinstance(cfg_loaded, DictConfig), "Config root must be a mapping"
    cfg: DictConfig = cfg_loaded

    # Extract config with defaults
    data_cfg = cast(Dict[str, Any], cfg.get("data", {}))
    model_cfg = cast(Dict[str, Any], cfg.get("model", {}))
    training_cfg = cast(Dict[str, Any], cfg.get("training", {}))
    output_cfg = cast(Dict[str, Any], cfg.get("output", {}))
    loss_cfg = cast(Dict[str, Any], cfg.get("loss", {}))
    eval_cfg = cast(Dict[str, Any], cfg.get("eval", {}))

    # Apply CLI overrides
    gradient_dim = data_cfg.get("gradient_dim", None)
    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)  # Convert from OmegaConf ListConfig
    save_dir = output_cfg.get("save_dir", "ckpts/attacker_temporal")
    num_epochs = training_cfg.get("num_epochs", 50)
    seq_len = model_cfg.get("sequence_length", 8)

    # Save config to output directory for reproducibility
    is_master = int(os.environ.get("RANK", 0)) == 0
    if is_master:
        os.makedirs(save_dir, exist_ok=True)
        config_save_path = Path(save_dir) / "config.yaml"
        OmegaConf.save(cfg, str(config_save_path))
        print(f"Saved config to {config_save_path}")

    wandb_cfg = cast(Dict[str, Any], cfg.get("wandb", {}))

    # Only init WandB on master process (rank 0) to avoid duplicate runs
    is_master = int(os.environ.get("RANK", 0)) == 0
    if wandb_cfg.get("enabled", False) and is_master:
        wandb.login()
        wandb_container = OmegaConf.to_container(cfg, resolve=True)
        wandb_config: Dict[str, Any] = {}
        if isinstance(wandb_container, dict):
            wandb_config = {str(k): v for k, v in wandb_container.items()}
        wandb.init(
            project=wandb_cfg.get("project", "gradient-inversion"),
            name=wandb_cfg.get("name", None),
            group=wandb_cfg.get("group", None),
            entity=wandb_cfg.get("entity", None),
            config=wandb_config,
        )

    train(
        h5_path=data_cfg.get("h5_path", "trajectory_data/gradients.h5"),
        num_epochs=num_epochs,
        batch_size=training_cfg.get("batch_size", 2),
        accumulation_steps=training_cfg.get("accumulation_steps", 1),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        device=cfg.get("device", "auto"),
        save_dir=save_dir,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
        sequence_length=seq_len,
        latent_dim=model_cfg.get("latent_dim", 512),
        stride=model_cfg.get("stride", 4),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", None),
        encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", None),
        encoder_num_blocks=model_cfg.get("encoder_num_blocks", None),
        encoder_expansion=model_cfg.get("encoder_expansion", None),
        encoder_projection_rank=model_cfg.get("encoder_projection_rank", None),
        encoder_type=model_cfg.get("encoder_type", "basic"),
        decoder_type=model_cfg.get("decoder_type", "basic"),
        dropout=model_cfg.get("dropout", 0.1),
        # Loss weights
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        l1_weight=loss_cfg.get("l1_weight", 0.5),
        action_weight=loss_cfg.get("action_weight", 0.1),
        temporal_weight=loss_cfg.get("temporal_weight", 0.0),
        lpips_weight=loss_cfg.get("lpips_weight", 0.0),
        lpips_net=loss_cfg.get("lpips_net", "vgg"),
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        use_wandb=wandb_cfg.get("enabled", False),
        # Training improvements
        gradient_noise_scale=training_cfg.get("gradient_noise_scale", 0.0),
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        lr_schedule=training_cfg.get("lr_schedule", "none"),
        min_lr=training_cfg.get("min_lr", 1e-6),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
        num_workers=training_cfg.get("num_workers", 4),
        # Fine-tuning
        pretrained_checkpoint=model_cfg.get("pretrained_checkpoint", ""),
        finetune_fraction=data_cfg.get("finetune_fraction", 1.0),
        # Ablation
        skip_transformer=model_cfg.get("skip_transformer", False),
        is_causal=model_cfg.get("is_causal", True),
        temporal_model_type=model_cfg.get("temporal_model_type", "transformer"),
        ff_multiplier=model_cfg.get("ff_multiplier", 4),
        use_rope=model_cfg.get("use_rope", False),
        latent_temporal_weight=loss_cfg.get("latent_temporal_weight", 0.0),
        max_sequences=data_cfg.get("max_sequences", None),
        gradient_mask_ratio=training_cfg.get("gradient_mask_ratio", 0.0),
        model_type=model_cfg.get("model_type", "temporal"),
        # Scheduled sampling & rollout
        scheduled_sampling_start=training_cfg.get("scheduled_sampling_start", 0.0),
        scheduled_sampling_end=training_cfg.get("scheduled_sampling_end", 0.0),
        scheduled_sampling_warmup=training_cfg.get("scheduled_sampling_warmup", 5),
        rollout_steps=training_cfg.get("rollout_steps", 0),
        rollout_weight=training_cfg.get("rollout_weight", 0.0),
    )

    # Run evaluation on best model after training (master process only)
    is_master = int(os.environ.get("RANK", 0)) == 0
    if is_master and eval_cfg.get("enabled", True):
        print("Running post-training evaluation...")

        evaluate(
            checkpoint_path=str(Path(save_dir) / "best_model.pt"),
            h5_path=eval_cfg.get("h5_path", "trajectory_data/eval_tf_108.h5"),
            save_dir=str(Path(save_dir) / "eval_results"),
            num_sequences=eval_cfg.get("num_sequences", 5),
            sequence_length=seq_len,
            stride=model_cfg.get("stride", 8),
            gradient_dim=gradient_dim,
            gradient_layers=gradient_layers,
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
            skip_transformer=model_cfg.get("skip_transformer", False),
            temporal_model_type=model_cfg.get("temporal_model_type", "transformer"),
            ff_multiplier=model_cfg.get("ff_multiplier", 4),
            use_rope=model_cfg.get("use_rope", False),
            model_type=model_cfg.get("model_type", "temporal"),
        )

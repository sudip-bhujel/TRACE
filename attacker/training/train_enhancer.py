"""
Super-Resolution Enhancer Training

Two-phase approach:
  Phase 1: Single-GPU pair generation → streams to HDF5 (constant memory)
  Phase 2: Multi-GPU DDP enhancer training → reads from HDF5

Usage:
    Generate pairs:  python -m attacker.train_enhancer <config> --generate-only
    Train (DDP):     torchrun --standalone --nproc_per_node=4 -m attacker.train_enhancer <config>
    Both (1 GPU):    python -m attacker.train_enhancer <config>
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import h5py
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from attacker.data.dataset import TemporalGradientDataset
from attacker.models.enhancer import EnhancerUNet
from attacker.evaluation.evaluate import load_model


class HDF5PairDataset(Dataset):
    """Dataset that reads (blurry, GT) pairs from an HDF5 file."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length = f["predictions"].shape[0]

        # Lazy open per-worker (HDF5 isn't fork-safe)
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open()
        pred = torch.from_numpy(self._h5["predictions"][idx].astype(np.float32))
        gt = torch.from_numpy(self._h5["ground_truths"][idx].astype(np.float32))
        return pred, gt

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def generate_pairs(
    baseline_checkpoint: str,
    h5_path: str,
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    sequence_length: int = 8,
    stride: int = 2,
    latent_dim: int = 1024,
    num_transformer_layers: int = 6,
    num_heads: int = 8,
    encoder_hidden_dims: Optional[List[int]] = None,
    encoder_type: str = "residual",
    decoder_type: str = "residual",
    save_path: str = "ckpts/enhancer/pairs.h5",
    device: str = "auto",
    batch_size: int = 4,
):
    """
    Generate (blurry_prediction, ground_truth) pairs and stream to HDF5.
    Memory stays constant — no accumulation.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"Device: {device}")

    # Load dataset
    gradient_dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
    )
    actual_gradient_dim = gradient_dataset.effective_gradient_dim
    num_actions = len(gradient_dataset.actions.unique())

    # Load baseline model
    print(f"Loading baseline model from: {baseline_checkpoint}")
    base_model = load_model(
        baseline_checkpoint,
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
    base_model.eval()

    loader = DataLoader(
        gradient_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Calculate total images
    total_images = len(gradient_dataset) * sequence_length
    print(f"Generating {total_images} pairs from {len(gradient_dataset)} sequences...")

    # Create HDF5 file with preallocated datasets
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as float16 to save disk space (84x84x3 images don't need float32 precision)
    with h5py.File(save_path, "w") as f:
        pred_ds = f.create_dataset(
            "predictions",
            shape=(total_images, 3, 84, 84),
            dtype=np.float16,
            chunks=(64, 3, 84, 84),
        )
        gt_ds = f.create_dataset(
            "ground_truths",
            shape=(total_images, 3, 84, 84),
            dtype=np.float16,
            chunks=(64, 3, 84, 84),
        )

        idx = 0
        with torch.no_grad():
            for gradients, images, _ in tqdm(loader, desc="Generating pairs"):
                gradients = gradients.to(device)
                pred_images, _, _, _ = base_model(gradients)
                pred_images = pred_images.cpu().clamp(0, 1)

                B, T = images.shape[:2]
                for b in range(B):
                    for t in range(T):
                        if idx < total_images:
                            pred_ds[idx] = pred_images[b, t].numpy().astype(np.float16)
                            gt_ds[idx] = images[b, t].numpy().astype(np.float16)
                            idx += 1

        # Trim if we got fewer than expected
        if idx < total_images:
            pred_ds.resize(idx, axis=0)
            gt_ds.resize(idx, axis=0)

    size_gb = save_path.stat().st_size / 1e9
    print(f"Saved {idx} pairs to {save_path} ({size_gb:.1f} GB)")


def train_enhancer(
    # Pair data
    pairs_path: str = "ckpts/enhancer/pairs.h5",
    # Enhancer config
    base_channels: int = 64,
    # Training
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
    # Loss
    mse_weight: float = 1.0,
    lpips_weight: float = 0.5,
    lpips_net: str = "vgg",
    l1_weight: float = 0.5,
    # Output
    save_dir: str = "ckpts/enhancer",
    device: str = "auto",
    num_workers: int = 4,
    use_wandb: bool = False,
    **kwargs,
):
    """Train the super-resolution enhancer with optional DDP support."""
    # =========================================================================
    # DDP Setup
    # =========================================================================
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        master_process = True
        ddp_world_size = 1
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

    if master_process:
        print(f"Device: {device} (DDP: {ddp}, world_size: {ddp_world_size})")

    save_dir = Path(save_dir)
    if master_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs from HDF5
    if master_process:
        print(f"\nLoading pairs from: {pairs_path}")
    pair_dataset = HDF5PairDataset(pairs_path)
    n = len(pair_dataset)

    if master_process:
        print(f"  Total pairs: {n}")

    # Split into train/val (90/10) using index ranges
    n_val = max(n // 10, 1)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        pair_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=num_workers > 0,
    )

    if master_process:
        print(f"  Train: {n_train} images, Val: {n_val} images")

    # Create enhancer model
    enhancer = EnhancerUNet(in_channels=3, base_channels=base_channels).to(device)

    if master_process:
        num_params = sum(p.numel() for p in enhancer.parameters())
        print(f"\nEnhancer params: {num_params:,}")

    # Wrap in DDP
    if ddp:
        enhancer = DDP(enhancer, device_ids=[int(os.environ["LOCAL_RANK"])])
    raw_enhancer = enhancer.module if ddp else enhancer

    # Optimizer
    optimizer = optim.AdamW(
        enhancer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )

    # Warmup
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )

    # LPIPS loss
    lpips_fn = None
    if lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
        for param in lpips_fn.parameters():
            param.requires_grad = False

    # Mixed precision
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler()

    # Training loop
    if master_process:
        print(f"\n{'=' * 60}")
        print("Starting enhancer training...")
        print(f"  Epochs: {num_epochs}")
        print(
            f"  Batch size: {batch_size} x {ddp_world_size} GPUs = {batch_size * ddp_world_size}"
        )
        print(f"  Loss: MSE({mse_weight}) + L1({l1_weight}) + LPIPS({lpips_weight})")
        print(f"{'=' * 60}")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        enhancer.train()
        train_losses = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not master_process)
        for blurry, sharp in pbar:
            blurry = blurry.to(device)
            sharp = sharp.to(device)

            with torch.amp.autocast(
                device_type=device.type, enabled=scaler is not None
            ):
                enhanced = enhancer(blurry)

                mse_loss = nn.functional.mse_loss(enhanced, sharp)
                l1_loss = nn.functional.l1_loss(enhanced, sharp)

                lp_loss = torch.tensor(0.0, device=device)
                if lpips_fn is not None:
                    lp_loss = lpips_fn(enhanced, sharp).mean()

                loss = (
                    mse_weight * mse_loss + l1_weight * l1_loss + lpips_weight * lp_loss
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(enhancer.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enhancer.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses["total"] += loss.item()
            train_losses["mse"] += mse_loss.item()
            train_losses["l1"] += l1_loss.item()
            train_losses["lpips"] += lp_loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse_loss.item():.4f}",
                lpips=f"{lp_loss.item():.4f}",
            )

        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # Validate
        enhancer.eval()
        val_losses = defaultdict(float)

        with torch.no_grad():
            for blurry, sharp in val_loader:
                blurry = blurry.to(device)
                sharp = sharp.to(device)

                enhanced = enhancer(blurry)
                mse_loss = nn.functional.mse_loss(enhanced, sharp)
                l1_loss = nn.functional.l1_loss(enhanced, sharp)

                lp_loss = torch.tensor(0.0, device=device)
                if lpips_fn is not None:
                    lp_loss = lpips_fn(enhanced, sharp).mean()

                loss = (
                    mse_weight * mse_loss + l1_weight * l1_loss + lpips_weight * lp_loss
                )

                val_losses["total"] += loss.item()
                val_losses["mse"] += mse_loss.item()
                val_losses["l1"] += l1_loss.item()
                val_losses["lpips"] += lp_loss.item()

        for k in val_losses:
            val_losses[k] /= len(val_loader)

        # LR scheduling
        if epoch <= warmup_epochs and warmup_scheduler is not None:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # Print and save (master only)
        if master_process:
            print(
                f"Epoch {epoch}: "
                f"Train {train_losses['total']:.4f} (MSE:{train_losses['mse']:.4f}, L1:{train_losses['l1']:.4f}, LPIPS:{train_losses['lpips']:.4f}) | "
                f"Val {val_losses['total']:.4f} (MSE:{val_losses['mse']:.4f}, L1:{val_losses['l1']:.4f}, LPIPS:{val_losses['lpips']:.4f})"
            )

            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "train/loss": train_losses["total"],
                        "train/mse": train_losses["mse"],
                        "train/l1": train_losses["l1"],
                        "train/lpips": train_losses["lpips"],
                        "val/loss": val_losses["total"],
                        "val/mse": val_losses["mse"],
                        "val/l1": val_losses["l1"],
                        "val/lpips": val_losses["lpips"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            # Save best
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_enhancer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_losses["total"],
                        "base_channels": base_channels,
                    },
                    save_dir / "best_enhancer.pt",
                )
                print(f"  -> Saved best enhancer (val_loss={best_val_loss:.4f})")

            # Save sample reconstructions every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                save_enhancer_samples(
                    raw_enhancer,
                    val_loader,
                    device,
                    save_dir / f"enhancer_epoch_{epoch:03d}.png",
                )

    if master_process:
        print(f"\n{'=' * 60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoint: {save_dir / 'best_enhancer.pt'}")
        print(f"{'=' * 60}")

    # Cleanup DDP
    if ddp:
        destroy_process_group()


def save_enhancer_samples(
    enhancer: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
):
    """Save before/after comparison images."""
    enhancer.eval()

    blurry_batch, sharp_batch = next(iter(dataloader))
    num_samples = min(num_samples, len(blurry_batch))
    blurry = blurry_batch[:num_samples].to(device)
    sharp = sharp_batch[:num_samples]

    with torch.no_grad():
        enhanced = enhancer(blurry).cpu()

    blurry = blurry.cpu()

    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

    for i in range(num_samples):
        axes[0, i].imshow(blurry[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].set_title("Blurry", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(enhanced[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].set_title("Enhanced", fontsize=7)
        axes[1, i].axis("off")

        axes[2, i].imshow(sharp[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[2, i].set_title("Ground Truth", fontsize=7)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved enhancer samples to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Generate pairs:  python -m attacker.train_enhancer <config> --generate-only"
        )
        print(
            "  Train (DDP):     torchrun --nproc_per_node=4 -m attacker.train_enhancer <config>"
        )
        print("  Both (1 GPU):    python -m attacker.train_enhancer <config>")
        sys.exit(1)

    # Check for --generate-only flag
    generate_only = "--generate-only" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--generate-only"]

    cfg = OmegaConf.load(argv[0])
    if len(argv) > 1:
        cli_cfg = OmegaConf.from_dotlist(argv[1:])
        cfg = OmegaConf.merge(cfg, cli_cfg)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    baseline_cfg = cfg.get("baseline", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    output_cfg = cfg.get("output", {})
    eval_cfg = cfg.get("eval", {})

    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    save_dir = output_cfg.get("save_dir", "ckpts/enhancer")
    pairs_path = os.path.join(save_dir, "pairs.h5")

    # Phase 1: Generate pairs (single GPU)
    is_master = int(os.environ.get("RANK", 0)) == 0
    need_generate = not os.path.exists(pairs_path)

    if need_generate:
        if is_master:
            print(f"Pairs file not found at {pairs_path}, generating...")
            os.makedirs(save_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

            generate_pairs(
                baseline_checkpoint=baseline_cfg.get("checkpoint", ""),
                h5_path=data_cfg.get("h5_path", ""),
                gradient_dim=data_cfg.get("gradient_dim", None),
                gradient_layers=gradient_layers,
                sequence_length=model_cfg.get("sequence_length", 8),
                stride=model_cfg.get("stride", 2),
                latent_dim=baseline_cfg.get("latent_dim", 1024),
                num_transformer_layers=baseline_cfg.get("num_transformer_layers", 6),
                num_heads=baseline_cfg.get("num_heads", 8),
                encoder_hidden_dims=baseline_cfg.get("encoder_hidden_dims", None),
                encoder_type=baseline_cfg.get("encoder_type", "residual"),
                decoder_type=baseline_cfg.get("decoder_type", "residual"),
                save_path=pairs_path,
                device=cfg.get("device", "auto"),
                batch_size=training_cfg.get("gen_batch_size", 4),
            )
    else:
        if is_master and generate_only:
            print(f"Pairs file already exists at {pairs_path}. Skipping generation.")

    if generate_only:
        if is_master:
            print(
                "Pair generation phase complete (or skipped). Run again without --generate-only to train."
            )
        sys.exit(0)

    # Phase 2: Train enhancer (supports DDP)
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and is_master:
        import wandb

        wandb.login()
        wandb.init(
            project=wandb_cfg.get("project", "gradient-inversion"),
            name=wandb_cfg.get("name", None),
            entity=wandb_cfg.get("entity", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_enhancer(
        pairs_path=pairs_path,
        base_channels=model_cfg.get("base_channels", 64),
        num_epochs=training_cfg.get("num_epochs", 50),
        batch_size=training_cfg.get("batch_size", 32),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
        lr_schedule=training_cfg.get("lr_schedule", "cosine"),
        warmup_epochs=training_cfg.get("warmup_epochs", 3),
        min_lr=training_cfg.get("min_lr", 1e-6),
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        lpips_weight=loss_cfg.get("lpips_weight", 0.5),
        lpips_net=loss_cfg.get("lpips_net", "vgg"),
        l1_weight=loss_cfg.get("l1_weight", 0.5),
        save_dir=save_dir,
        device=cfg.get("device", "auto"),
        num_workers=training_cfg.get("num_workers", 4),
        use_wandb=wandb_cfg.get("enabled", False),
    )

    # Post-training evaluation with enhancer (master only)
    if eval_cfg.get("enabled", True) and eval_cfg.get("h5_path", "") and is_master:
        from attacker.evaluation.evaluate import evaluate

        print("\nRunning post-training evaluation with enhancer...")
        enhancer_ckpt = os.path.join(save_dir, "best_enhancer.pt")
        eval_save_dir = os.path.join(save_dir, "eval_results")

        evaluate(
            checkpoint_path=baseline_cfg.get("checkpoint", ""),
            h5_path=eval_cfg.get("h5_path"),
            save_dir=eval_save_dir,
            num_sequences=eval_cfg.get("num_sequences", 20),
            sequence_length=model_cfg.get("sequence_length", 8),
            stride=model_cfg.get("stride", 8),
            gradient_dim=data_cfg.get("gradient_dim", None),
            gradient_layers=gradient_layers,
            device=cfg.get("device", "auto"),
            latent_dim=baseline_cfg.get("latent_dim", 1024),
            num_transformer_layers=baseline_cfg.get("num_transformer_layers", 6),
            num_heads=baseline_cfg.get("num_heads", 8),
            encoder_hidden_dims=baseline_cfg.get("encoder_hidden_dims", None),
            encoder_type=baseline_cfg.get("encoder_type", "residual"),
            decoder_type=baseline_cfg.get("decoder_type", "residual"),
            enhancer_checkpoint=enhancer_ckpt,
        )

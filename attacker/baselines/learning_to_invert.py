"""
Learning to Invert (Wu et al., UAI 2023):
    Feed-forward MLP that maps flattened gradients directly to images.
    Adapted from the original CIFAR10/LeNet setting to our embodied RL
    setting (PPO gradients, 84x84 images, 5-action discrete space).

Training:
    uv run -m attacker.baselines.learning_to_invert attacker/config/lti_train.yaml

Based on: https://github.com/wrh14/Learning_to_Invert
"""

import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb


class FlatGradientImageDataset(Dataset):
    """
    Per-frame (gradient, image, action) dataset from HDF5.

    Unlike ``TemporalGradientDataset``, this returns individual frames rather
    than sequences — matching the per-sample i.i.d. assumption of Wu et al.
    """

    def __init__(
        self,
        h5_path: str,
        selected_para: Optional[torch.Tensor] = None,
        gradient_dim: Optional[int] = None,
    ):
        self.h5_path = h5_path
        self.selected_para = selected_para  # index tensor for compression

        with h5py.File(h5_path, "r") as f:
            self.images = torch.tensor(f["images"][:], dtype=torch.float32) / 255.0
            self.actions = torch.tensor(f["actions"][:], dtype=torch.long)
            self.gradient_shape = f["gradients"].shape

        self.gradient_dim = gradient_dim
        self._h5_file = None
        self._gradients_ds = None

        print(
            f"FlatGradientImageDataset: {len(self)} samples, "
            f"grad_dim={self.gradient_shape[1]}"
        )

    def _ensure_h5_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._gradients_ds = self._h5_file["gradients"]

    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            self._h5_file.close()

    def __len__(self) -> int:
        return self.gradient_shape[0]

    def __getitem__(self, idx: int):
        self._ensure_h5_open()
        grad_np = self._gradients_ds[idx]

        if self.gradient_dim is not None and self.gradient_dim < len(grad_np):
            grad_np = grad_np[: self.gradient_dim]

        gradient = torch.tensor(grad_np, dtype=torch.float32)

        # Apply random parameter selection (compression) if provided
        if self.selected_para is not None:
            gradient = gradient[self.selected_para]

        return gradient, self.images[idx], self.actions[idx]


class LearningToInvertModel(nn.Module):
    """
    Learning-based gradient-to-image MLP (Wu et al., UAI 2023).

    Architecture: gradient -> Linear -> ReLU -> Linear -> ReLU ->
                  |-> Linear -> Sigmoid  (image head)
                  |-> Linear             (action head)

    The model uses random parameter selection to compress the input gradient.
    The ``selected_para`` index tensor is stored as a buffer so it persists
    in checkpoints.
    """

    def __init__(
        self,
        gradient_dim: int,
        hidden_size: int = 3000,
        num_actions: int = 5,
        image_channels: int = 3,
        image_size: int = 84,
        compress_rate: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.gradient_dim = gradient_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.image_flat_dim = image_channels * image_size * image_size
        self.num_actions = num_actions
        self.compress_rate = compress_rate

        # Deterministic random parameter selection (matching Wu et al.)
        compressed_dim = int(gradient_dim * compress_rate)
        gen = torch.Generator()
        gen.manual_seed(seed)
        selected_para = torch.randperm(gradient_dim, generator=gen)[:compressed_dim]
        self.register_buffer("selected_para", selected_para)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(compressed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Image head
        self.image_head = nn.Sequential(
            nn.Linear(hidden_size, self.image_flat_dim),
            nn.Sigmoid(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_size, num_actions)

    def compress_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Select a random subset of gradient dimensions."""
        return gradient[..., self.selected_para]

    def forward(self, gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gradient: (..., compressed_dim) — already compressed gradients.

        Returns:
            images:  (..., C, H, W) in [0, 1]
            actions: (..., num_actions) logits
        """
        leading = gradient.shape[:-1]
        flat = gradient.reshape(-1, gradient.shape[-1])

        h = self.trunk(flat)
        img = self.image_head(h)
        act = self.action_head(h)

        img = img.reshape(
            *leading, self.image_channels, self.image_size, self.image_size
        )
        act = act.reshape(*leading, self.num_actions)
        return img, act

    def reconstruct(
        self,
        gradients: torch.Tensor,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct images from a batch of gradient sequences.

        This matches the DLG/IG ``reconstruct`` API so evaluate_baselines
        can call it uniformly.

        Args:
            gradients: (B, T, gradient_dim) — raw (uncompressed) gradients.

        Returns:
            images:  (B, T, 3, H, W)
            actions: (B, T, num_actions)
        """
        compressed = self.compress_gradient(gradients)
        with torch.no_grad():
            images, actions = self.forward(compressed)
        return images, actions


def train_lti(cfg_path: str):
    """Train the Learning-to-Invert baseline from a YAML config."""

    cfg = OmegaConf.load(cfg_path)
    print(f"Loaded config from {cfg_path}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    gradient_dim = data_cfg.get("gradient_dim", 936102)

    # Build model
    model = LearningToInvertModel(
        gradient_dim=gradient_dim,
        hidden_size=model_cfg.get("hidden_size", 3000),
        num_actions=model_cfg.get("num_actions", 5),
        image_size=model_cfg.get("image_size", 84),
        compress_rate=model_cfg.get("compress_rate", 0.1),
        seed=model_cfg.get("seed", 0),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build datasets — pass selected_para for on-the-fly compression
    selected_para = model.selected_para.cpu()
    train_dataset = FlatGradientImageDataset(
        data_cfg["train_h5_path"],
        selected_para=selected_para,
        gradient_dim=gradient_dim,
    )
    test_dataset = FlatGradientImageDataset(
        data_cfg["test_h5_path"],
        selected_para=selected_para,
        gradient_dim=gradient_dim,
    )

    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer
    lr = train_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_weight = train_cfg.get("action_weight", 0.01)

    num_epochs = train_cfg.get("num_epochs", 200)
    lr_decay_epoch = int(num_epochs * train_cfg.get("lr_decay_epoch_frac", 0.75))
    lr_decay_factor = train_cfg.get("lr_decay_factor", 0.1)

    save_dir = Path(output_cfg.get("save_dir", "ckpts/baselines/lti"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Wandb (optional)
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "gradient-inversion"),
            name=wandb_cfg.get("name", "lti-baseline"),
            entity=wandb_cfg.get("entity", None),
            group=wandb_cfg.get("group", None),
            tags=wandb_cfg.get("tags", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_test_loss = float("inf")
    best_state_dict = None

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for grads, images, actions in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            grads = grads.to(device)
            images = images.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            pred_images, pred_actions = model(grads)
            img_loss = F.mse_loss(pred_images, images)
            act_loss = F.cross_entropy(pred_actions, actions)
            loss = img_loss + action_weight * act_loss

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * grads.size(0)
            train_count += grads.size(0)

        train_loss = train_loss_sum / train_count

        # --- LR decay ---
        if epoch + 1 == lr_decay_epoch:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay_factor
            print(f"  LR decayed by {lr_decay_factor} at epoch {epoch + 1}")

        # --- Eval ---
        model.eval()
        test_loss_sum = 0.0
        test_count = 0
        correct = 0

        with torch.no_grad():
            for grads, images, actions in tqdm(
                test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [test]", leave=False
            ):
                grads = grads.to(device)
                images = images.to(device)
                actions = actions.to(device)

                pred_images, pred_actions = model(grads)
                img_loss = F.mse_loss(pred_images, images)
                act_loss = F.cross_entropy(pred_actions, actions)
                loss = img_loss + action_weight * act_loss

                test_loss_sum += loss.item() * grads.size(0)
                test_count += grads.size(0)
                correct += (pred_actions.argmax(-1) == actions).sum().item()

        test_loss = test_loss_sum / test_count
        test_acc = 100.0 * correct / test_count

        # Compute test PSNR from MSE
        test_psnr = -10.0 * math.log10(max(test_loss, 1e-10))

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train loss: {train_loss:.6f} | "
            f"Test loss: {test_loss:.6f} | "
            f"Test PSNR: {test_psnr:.2f} dB | "
            f"Test Acc: {test_acc:.1f}%"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_psnr": test_psnr,
                    "test_action_accuracy": test_acc,
                },
                step=epoch + 1,
            )

        # Track best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = deepcopy(model.cpu().state_dict())
            model.to(device)

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": {"total": test_loss},
                    "test_action_accuracy": test_acc,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                save_dir / "best_model.pt",
            )

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0 or epoch + 1 == num_epochs:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": {"total": test_loss},
                },
                save_dir / f"epoch_{epoch + 1:04d}.pt",
            )

    print(f"\nTraining complete. Best test loss: {best_test_loss:.6f}")
    print(f"Checkpoints saved to: {save_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    assert len(sys.argv) > 1, (
        "Usage: uv run -m attacker.baselines.learning_to_invert <config_path>"
    )
    train_lti(sys.argv[1])

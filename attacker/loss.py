from typing import Dict, Tuple

import lpips
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalCombinedLoss(nn.Module):
    """
    Combined loss for temporal gradient inversion.

    Computes per-frame losses and averages over the sequence.
    Optionally includes temporal smoothness loss.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        action_weight: float = 0.1,
        temporal_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.action_weight = action_weight
        self.temporal_weight = temporal_weight
        self.lpips_weight = lpips_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        # LPIPS loss (only create if weight > 0)
        if lpips_weight > 0:
            self.lpips_loss = lpips.LPIPS(net=lpips_net).to(device)
        else:
            self.lpips_loss = None

    def forward(
        self,
        pred_images: torch.Tensor,  # (B, T, 3, H, W)
        target_images: torch.Tensor,  # (B, T, 3, H, W)
        pred_actions: torch.Tensor,  # (B, T, num_actions)
        target_actions: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        B, T = pred_images.shape[:2]

        # Flatten for loss computation
        pred_img_flat = pred_images.reshape(-1, *pred_images.shape[2:])
        target_img_flat = target_images.reshape(-1, *target_images.shape[2:])
        pred_act_flat = pred_actions.reshape(-1, pred_actions.shape[-1])
        target_act_flat = target_actions.reshape(-1)

        mse = self.mse_loss(pred_img_flat, target_img_flat)
        l1 = self.l1_loss(pred_img_flat, target_img_flat)
        action_loss = self.ce_loss(pred_act_flat, target_act_flat)

        # LPIPS loss (VGG features)
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(pred_img_flat, target_img_flat).mean()
        else:
            lpips = torch.tensor(0.0, device=pred_images.device)

        # Temporal smoothness (penalize large changes between consecutive frames)
        if self.temporal_weight > 0 and T > 1:
            temporal_diff = pred_images[:, 1:] - pred_images[:, :-1]
            temporal_loss = temporal_diff.pow(2).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=pred_images.device)

        # Total loss
        total = (
            self.mse_weight * mse
            + self.l1_weight * l1
            + self.action_weight * action_loss
            + self.lpips_weight * lpips
            + self.temporal_weight * temporal_loss
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse.item(),
            "l1": l1.item(),
            "action": action_loss.item(),
            "lpips": lpips.item(),
            "temporal": temporal_loss.item()
            if torch.is_tensor(temporal_loss)
            else temporal_loss,
        }

        return total, loss_dict

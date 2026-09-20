from typing import Dict, Optional, Tuple

import lpips
import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalCombinedLoss(nn.Module):
    """Combined per-frame loss: MSE + L1 + action + optional LPIPS / temporal terms."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        action_weight: float = 0.1,
        temporal_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg",
        latent_temporal_weight: float = 0.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.action_weight = action_weight
        self.temporal_weight = temporal_weight
        self.lpips_weight = lpips_weight
        self.latent_temporal_weight = latent_temporal_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        if lpips_weight > 0:
            self.lpips_loss = lpips.LPIPS(net=lpips_net).to(device)
            # LPIPS VGG uses ReLU(inplace=True), which corrupts the autograd graph of
            # upstream BatchNorm layers. Disable inplace before any forward pass.
            for module in self.lpips_loss.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
        else:
            self.lpips_loss = None

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        token_logits: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T = pred_images.shape[:2]

        pred_img_flat = pred_images.reshape(-1, *pred_images.shape[2:])
        target_img_flat = target_images.reshape(-1, *target_images.shape[2:])
        pred_act_flat = pred_actions.reshape(-1, pred_actions.shape[-1])
        target_act_flat = (
            target_actions.reshape_as(pred_act_flat)
            if target_actions.ndim == pred_actions.ndim
            else target_actions.reshape(-1)
        )

        mse = self.mse_loss(pred_img_flat, target_img_flat)
        l1 = self.l1_loss(pred_img_flat, target_img_flat)
        action_loss = self.ce_loss(pred_act_flat, target_act_flat)
        # Do not zero histogram gradients when the pretrained classifier is confident.
        if target_actions.ndim != pred_actions.ndim:
            action_loss = torch.clamp(action_loss, max=2.0)

        if self.lpips_loss is not None:
            lpips_val = self.lpips_loss(pred_img_flat, target_img_flat).mean()
        else:
            lpips_val = torch.tensor(0.0, device=pred_images.device)

        if self.temporal_weight > 0 and T > 1:
            pred_diff = pred_images[:, 1:] - pred_images[:, :-1]
            gt_diff = target_images[:, 1:] - target_images[:, :-1]
            temporal_loss = F.mse_loss(pred_diff, gt_diff)
        else:
            temporal_loss = torch.tensor(0.0, device=pred_images.device)

        if self.latent_temporal_weight > 0 and latents is not None and T > 1:
            latent_diff = latents[:, 1:] - latents[:, :-1]
            latent_temporal_loss = latent_diff.pow(2).mean()
        else:
            latent_temporal_loss = torch.tensor(0.0, device=pred_images.device)

        total = (
            self.mse_weight * mse
            + self.l1_weight * l1
            + self.action_weight * action_loss
            + self.lpips_weight * lpips_val
            + self.temporal_weight * temporal_loss
            + self.latent_temporal_weight * latent_temporal_loss
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse.item(),
            "l1": l1.item(),
            "action": action_loss.item(),
            "lpips": lpips_val.item(),
            "temporal": temporal_loss.item()
            if torch.is_tensor(temporal_loss)
            else temporal_loss,
            "latent_temporal": latent_temporal_loss.item()
            if torch.is_tensor(latent_temporal_loss)
            else latent_temporal_loss,
        }

        return total, loss_dict

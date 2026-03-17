from typing import Dict, Optional, Tuple

import lpips
import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalCombinedLoss(nn.Module):
    """
    Combined loss for temporal gradient inversion.

    Computes per-frame losses and averages over the sequence.
    Optionally includes temporal smoothness loss and VQ token prediction loss.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        action_weight: float = 0.1,
        temporal_weight: float = 0.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg",
        token_weight: float = 0.0,
        dino_weight: float = 0.0,
        dino_model: str = "dinov2_vits14",
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.action_weight = action_weight
        self.temporal_weight = temporal_weight
        self.lpips_weight = lpips_weight
        self.token_weight = token_weight
        self.dino_weight = dino_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        # LPIPS loss (only create if weight > 0)
        if lpips_weight > 0:
            self.lpips_loss = lpips.LPIPS(net=lpips_net).to(device)
        else:
            self.lpips_loss = None

        # DINOv2 feature loss (only create if weight > 0)
        if dino_weight > 0:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", dino_model, verbose=False
            )
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino = self.dino.to(device)
        else:
            self.dino = None

    def forward(
        self,
        pred_images: torch.Tensor,  # (B, T, 3, H, W)
        target_images: torch.Tensor,  # (B, T, 3, H, W)
        pred_actions: torch.Tensor,  # (B, T, num_actions)
        target_actions: torch.Tensor,  # (B, T)
        token_logits: Optional[torch.Tensor] = None,  # (B*T, num_tokens, codebook_size)
        target_tokens: Optional[torch.Tensor] = None,  # (B*T, num_tokens)
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
        action_loss = torch.clamp(action_loss, max=2.0)  # prevent explosion

        # LPIPS loss (VGG features)
        if self.lpips_loss is not None:
            lpips_val = self.lpips_loss(pred_img_flat, target_img_flat).mean()
        else:
            lpips_val = torch.tensor(0.0, device=pred_images.device)

        # DINOv2 feature loss (semantic similarity)
        if self.dino is not None:
            # Run DINOv2 in float32 to avoid fp16 overflow
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # Upscale to 98x98 for clean 7x7 patch grid (98/14=7)
                pred_up = F.interpolate(
                    pred_img_flat.float(), size=98, mode="bilinear", align_corners=False
                )
                tgt_up = F.interpolate(
                    target_img_flat.float(),
                    size=98,
                    mode="bilinear",
                    align_corners=False,
                )
                with torch.no_grad():
                    tgt_feats = self.dino.forward_features(tgt_up)["x_norm_patchtokens"]
                pred_feats = self.dino.forward_features(pred_up)["x_norm_patchtokens"]
                dino_val = F.mse_loss(pred_feats, tgt_feats)
        else:
            dino_val = torch.tensor(0.0, device=pred_images.device)

        # Temporal smoothness (penalize large changes between consecutive frames)
        if self.temporal_weight > 0 and T > 1:
            pred_diff = pred_images[:, 1:] - pred_images[:, :-1]
            gt_diff = target_images[:, 1:] - target_images[:, :-1]
            temporal_loss = F.mse_loss(pred_diff, gt_diff)
        else:
            temporal_loss = torch.tensor(0.0, device=pred_images.device)

        # Token prediction loss (VQ-GAN mode)
        if (
            self.token_weight > 0
            and token_logits is not None
            and target_tokens is not None
        ):
            # token_logits: (B*T, num_tokens, codebook_size)
            # target_tokens: (B*T, num_tokens)
            token_loss = F.cross_entropy(
                token_logits.reshape(-1, token_logits.shape[-1]),
                target_tokens.reshape(-1),
            )
        else:
            token_loss = torch.tensor(0.0, device=pred_images.device)

        # Total loss
        total = (
            self.mse_weight * mse
            + self.l1_weight * l1
            + self.action_weight * action_loss
            + self.lpips_weight * lpips_val
            + self.dino_weight * dino_val
            + self.temporal_weight * temporal_loss
            + self.token_weight * token_loss
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse.item(),
            "l1": l1.item(),
            "action": action_loss.item(),
            "lpips": lpips_val.item(),
            "dino": dino_val.item() if torch.is_tensor(dino_val) else dino_val,
            "temporal": temporal_loss.item()
            if torch.is_tensor(temporal_loss)
            else temporal_loss,
            "token": token_loss.item() if torch.is_tensor(token_loss) else token_loss,
        }

        return total, loss_dict

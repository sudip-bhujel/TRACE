"""
Inverting Gradients (Geiping et al., NeurIPS 2020):
   Optimization-based gradient matching using cosine similarity plus total
   variation regularization. Labels are recovered analytically first, then
   only the image is optimized via Adam.

Based on: https://github.com/JonasGeiping/invertinggradients
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attacker.baselines.common import (
    _sorted_param_info,
    _unflatten_gradient,
    get_param_layer_map,
    recover_action_from_gradient,
    total_variation_loss,
)


class IGBaseline:
    """
    Inverting Gradients — Geiping et al., NeurIPS 2020.
    """

    def __init__(
        self,
        victim_model: nn.Module,
        gradient_dim: int,
        image_size: int = 84,
        num_actions: int = 5,
        num_iterations: int = 4800,
        lr: float = 0.1,
        tv_weight: float = 1e-1,
        num_restarts: int = 1,
        signed: bool = False,
    ):
        self.victim_model = victim_model
        self.gradient_dim = gradient_dim
        self.image_size = image_size
        self.num_actions = num_actions
        self.num_iterations = num_iterations
        self.lr = lr
        self.tv_weight = tv_weight
        self.num_restarts = num_restarts
        self.signed = signed

        self.layer_map, self.total_param_dim = get_param_layer_map(
            num_actions=num_actions
        )

        for p in self.victim_model.parameters():
            p.requires_grad_(True)
        self.victim_model.eval()

    def _run_once(
        self,
        observed_gradient: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Single optimisation run matching the official IG code."""
        device = observed_gradient.device
        observed_gradient = observed_gradient.detach()

        x_hat = torch.randn(
            1, 3, self.image_size, self.image_size, device=device
        ).requires_grad_(True)

        sorted_names, sorted_params, param_dict = _sorted_param_info(self.victim_model)
        original_grads = _unflatten_gradient(
            observed_gradient, sorted_names, param_dict, self.gradient_dim
        )

        opt = torch.optim.Adam([x_hat], lr=self.lr)

        # MultiStepLR matching the official IG code:
        # milestones at 3/8, 5/8, 7/8 of max_iterations, gamma=0.1
        n = self.num_iterations
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[n * 3 // 8, n * 5 // 8, n * 7 // 8],
            gamma=0.1,
        )

        best_loss = float("inf")
        best_x = x_hat.data.clone()

        for _ in range(self.num_iterations):

            def closure():
                opt.zero_grad()
                self.victim_model.zero_grad()

                x_input = x_hat * 255.0
                logits, value = self.victim_model(x_input)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action_tensor)
                policy_loss = -log_prob.mean()
                value_loss = value.mean()
                loss = policy_loss + 0.5 * value_loss

                trial_grads = torch.autograd.grad(
                    loss, sorted_params, create_graph=True
                )

                # Cosine similarity (cost_fn='sim') matching official code:
                # 1 + sum(-g1*g2) / sqrt(sum(g1^2)) / sqrt(sum(g2^2))
                dot = torch.zeros((), device=device)
                for tg, og in zip(trial_grads, original_grads):
                    dot = dot + (tg * og).sum()

                pnorm_trial = torch.zeros((), device=device)
                for tg in trial_grads:
                    pnorm_trial = pnorm_trial + tg.pow(2).sum()

                pnorm_orig = torch.zeros((), device=device)
                for og in original_grads:
                    pnorm_orig = pnorm_orig + og.pow(2).sum()
                rec_loss = 1.0 - dot / (pnorm_trial.sqrt() * pnorm_orig.sqrt() + 1e-12)

                if self.tv_weight > 0:
                    rec_loss = rec_loss + self.tv_weight * total_variation_loss(x_hat)

                rec_loss.backward()

                if self.signed and x_hat.grad is not None:
                    x_hat.grad.sign_()

                return rec_loss

            loss = closure()
            opt.step()
            scheduler.step()

            # boxed=True: project into valid image range [0, 1]
            with torch.no_grad():
                x_hat.data.clamp_(0, 1)

            cur = loss.item()
            if cur < best_loss:
                best_loss = cur
                best_x = x_hat.data.clone()

        return best_x, best_loss

    def reconstruct_single(
        self, observed_gradient: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Reconstruct one image from a single observed gradient.
        """
        device = observed_gradient.device

        action = recover_action_from_gradient(
            observed_gradient, self.layer_map, self.num_actions
        )
        if action < 0:
            action = 0
        action_tensor = torch.tensor([action], device=device)

        best_loss = float("inf")
        best_x: Optional[torch.Tensor] = None

        for _ in range(self.num_restarts):
            x, loss = self._run_once(observed_gradient, action_tensor)
            if loss < best_loss:
                best_loss = loss
                best_x = x

        assert best_x is not None
        return best_x.squeeze(0), action

    def reconstruct(
        self,
        gradients: torch.Tensor,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct images from a batch of gradient sequences.

        Args:
            gradients: (B, T, gradient_dim)
        Returns:
            images:  (B, T, 3, H, W)
            actions: (B, T, num_actions) — logits (10.0 at recovered action)
        """
        B, T, _ = gradients.shape
        all_images = []
        all_actions = []

        pbar = tqdm(
            total=B * T,
            desc="IG",
            disable=not show_progress,
        )

        for b in range(B):
            seq_images = []
            seq_actions = []
            for t in range(T):
                img, act = self.reconstruct_single(gradients[b, t])
                seq_images.append(img)
                act_logits = torch.zeros(self.num_actions, device=gradients.device)
                act_logits[act] = 10.0
                seq_actions.append(act_logits)
                pbar.update(1)
            all_images.append(torch.stack(seq_images))
            all_actions.append(torch.stack(seq_actions))

        pbar.close()
        return torch.stack(all_images), torch.stack(all_actions)

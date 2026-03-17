"""
DLG (Zhu et al., NeurIPS 2019):
   Optimization-based gradient matching using L2 distance. Jointly optimizes
   a dummy image AND dummy label via L-BFGS (Algorithm 1 from the paper).

Based on: https://github.com/mit-han-lab/dlg
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
)


class DLGBaseline:
    """
    DLG — Deep Leakage from Gradients (Zhu et al., NeurIPS 2019).
    """

    def __init__(
        self,
        victim_model: nn.Module,
        gradient_dim: int,
        image_size: int = 84,
        num_actions: int = 5,
        num_iterations: int = 300,
        lr: float = 1.0,
        num_restarts: int = 1,
    ):
        self.victim_model = victim_model
        self.gradient_dim = gradient_dim
        self.image_size = image_size
        self.num_actions = num_actions
        self.num_iterations = num_iterations
        self.lr = lr
        self.num_restarts = num_restarts

        self.layer_map, self.total_param_dim = get_param_layer_map(
            num_actions=num_actions
        )

        for p in self.victim_model.parameters():
            p.requires_grad_(True)
        self.victim_model.eval()

    def _run_once(
        self, observed_gradient: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Single optimisation run matching the official DLG code."""
        device = observed_gradient.device
        observed_gradient = observed_gradient.detach()

        x_hat = torch.randn(
            1, 3, self.image_size, self.image_size, device=device
        ).requires_grad_(True)
        y_hat = torch.randn(1, self.num_actions, device=device).requires_grad_(True)

        # Match official: torch.optim.LBFGS([dummy_data, dummy_label])
        opt = torch.optim.LBFGS([x_hat, y_hat], lr=self.lr)

        # Unflatten the observed gradient into per-parameter tensors
        sorted_names, sorted_params, param_dict = _sorted_param_info(self.victim_model)
        original_grads = _unflatten_gradient(
            observed_gradient, sorted_names, param_dict, self.gradient_dim
        )

        best_loss = float("inf")
        best_x = x_hat.data.clone()
        best_y = y_hat.data.clone()

        for _ in range(self.num_iterations):

            def closure():
                opt.zero_grad()
                self.victim_model.zero_grad()

                # Forward pass with soft label (cross_entropy_for_onehot)
                x_input = x_hat * 255.0
                logits, value = self.victim_model(x_input)
                dummy_onehot = F.softmax(y_hat, dim=-1)
                dummy_loss = (
                    torch.mean(
                        torch.sum(-dummy_onehot * F.log_softmax(logits, dim=-1), 1)
                    )
                    + 0.5 * value.mean()
                )

                dummy_grads = torch.autograd.grad(
                    dummy_loss, sorted_params, create_graph=True
                )

                # Per-parameter L2 matching (matches official code exactly)
                grad_diff = torch.zeros((), device=device)
                for dg, og in zip(dummy_grads, original_grads):
                    grad_diff = grad_diff + ((dg - og) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            loss = opt.step(closure)

            # NO clamping — matching the official DLG implementation

            cur = loss.item() if isinstance(loss, torch.Tensor) else loss
            if cur < best_loss:
                best_loss = cur
                best_x = x_hat.data.clone()
                best_y = y_hat.data.clone()

        return best_x, best_y, best_loss

    def reconstruct_single(
        self, observed_gradient: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Reconstruct one image and recover the action.

        Runs ``num_restarts`` independent optimisations from different random
        initialisations and keeps the result with the lowest matching loss.
        """
        best_loss = float("inf")
        best_x: Optional[torch.Tensor] = None
        best_y: Optional[torch.Tensor] = None

        for _ in range(self.num_restarts):
            x, y, loss = self._run_once(observed_gradient)
            if loss < best_loss:
                best_loss = loss
                best_x = x
                best_y = y

        assert best_x is not None and best_y is not None
        action = int(best_y.squeeze(0).argmax().item())
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
            desc="DLG",
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

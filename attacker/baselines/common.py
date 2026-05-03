"""Shared utilities for gradient inversion baselines."""

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


def get_param_layer_map(num_actions: int = 5, hidden_size: int = 512):
    """
    Layer map built from ``named_parameters()`` to match the capture script's
    alphabetical flattening order. ``analyze_gradients.get_layer_map()`` uses
    ``state_dict()`` and includes non-parameter buffers, so it does not match.
    """
    from victim.model import ActorCritic

    model = ActorCritic(in_channels=3, num_actions=num_actions, hidden_size=hidden_size)

    layer_map = OrderedDict()
    offset = 0
    for name, param in sorted(model.named_parameters()):
        n = param.numel()
        layer_map[name] = {
            "start": offset,
            "end": offset + n,
            "shape": tuple(param.shape),
            "num_params": n,
        }
        offset += n

    return layer_map, offset


def recover_action_from_gradient(
    gradient_vector: torch.Tensor,
    layer_map: Dict,
    num_actions: int = 5,
) -> int:
    """
    Recover the true action from the policy-head gradient.

    For policy head W of shape (num_actions, hidden_size) the gradient is
    dL/dW[k, j] = (pi(k|s) - delta_{k,a}) * h_j, so the row sum for the true
    action is the unique negative entry.
    """
    info = layer_map.get("policy.weight")
    if info is None or info["end"] > len(gradient_vector):
        return -1

    hidden_size = info["num_params"] // num_actions
    policy_grad = gradient_vector[info["start"] : info["end"]]
    policy_grad = policy_grad.reshape(num_actions, hidden_size)
    row_sums = policy_grad.sum(dim=1)
    return int(row_sums.argmin().item())


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
    dy = torch.mean(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
    return dx + dy


def _sorted_param_info(model: nn.Module):
    param_dict = dict(model.named_parameters())
    sorted_names = sorted(param_dict.keys())
    sorted_params = [param_dict[n] for n in sorted_names]
    return sorted_names, sorted_params, param_dict


def _unflatten_gradient(
    flat_grad: torch.Tensor,
    sorted_names: list,
    param_dict: dict,
    gradient_dim: int,
) -> list:
    """Split a flat gradient vector back into per-parameter tensors."""
    tensors = []
    offset = 0
    for name in sorted_names:
        n = param_dict[name].numel()
        if offset + n <= gradient_dim:
            tensors.append(
                flat_grad[offset : offset + n].reshape(param_dict[name].shape).detach()
            )
        else:
            remaining = max(0, gradient_dim - offset)
            chunk = flat_grad[offset : offset + remaining]
            padded = torch.zeros(n, device=flat_grad.device)
            padded[:remaining] = chunk
            tensors.append(padded.reshape(param_dict[name].shape).detach())
        offset += n
    return tensors

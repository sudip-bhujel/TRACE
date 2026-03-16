"""
Common utilities for baseline methods.
"""

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


def get_param_layer_map(num_actions: int = 5, hidden_size: int = 512):
    """
    Build layer map from named_parameters only (matching gradient capture order).

    The capture script (ppo/capture_gradients.py) flattens gradients sorted
    alphabetically by parameter name. This function reproduces those exact
    offsets, unlike analyze_gradients.get_layer_map() which uses state_dict()
    and includes non-parameter buffers.
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
    Recover the true action from the policy-head gradient (Theorem 1).

    For policy head W (num_actions x hidden_size) the gradient is
        dL/dW[k, j] = (pi(k|s) - delta_{k,a}) * h_j
    so the row sum for the true action is uniquely negative.
    """
    info = layer_map.get("policy.weight")
    if info is None or info["end"] > len(gradient_vector):
        return -1

    hidden_size = info["num_params"] // num_actions
    policy_grad = gradient_vector[info["start"] : info["end"]]
    policy_grad = policy_grad.reshape(num_actions, hidden_size)
    row_sums = policy_grad.sum(dim=1)
    return row_sums.argmin().item()


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    """Total variation regularization for spatial smoothness."""
    diff_h = images[:, :, 1:, :] - images[:, :, :-1, :]
    diff_w = images[:, :, :, 1:] - images[:, :, :, :-1]
    return (diff_h.pow(2).mean() + diff_w.pow(2).mean()) / 2


def _sorted_param_info(model: nn.Module):
    """Return (sorted_names, sorted_params) matching capture-script order."""
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
    """Split a flattened gradient vector back into per-parameter tensors."""
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

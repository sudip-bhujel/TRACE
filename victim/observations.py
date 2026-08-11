from typing import Dict, List, Optional, Union

import numpy as np
import torch

NumpyObservation = Union[np.ndarray, Dict[str, np.ndarray]]
TorchObservation = Union[torch.Tensor, Dict[str, torch.Tensor]]


def observation_to_torch(
    observation: NumpyObservation,
    device: torch.device,
    add_batch_dim: bool = True,
) -> TorchObservation:
    """Convert RGB or structured observations without changing their modalities."""
    if isinstance(observation, dict):
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in observation.items()
        }
        if add_batch_dim:
            tensors = {key: value.unsqueeze(0) for key, value in tensors.items()}
        return tensors

    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0) if add_batch_dim else tensor


def stack_observations(
    observations: List[NumpyObservation],
    device: torch.device,
) -> TorchObservation:
    """Stack a PPO rollout while retaining structured observation fields."""
    first = observations[0]
    if isinstance(first, dict):
        return {
            key: torch.as_tensor(
                np.stack([observation[key] for observation in observations]),
                dtype=torch.float32,
                device=device,
            )
            for key in first
        }
    return torch.as_tensor(
        np.stack(observations), dtype=torch.float32, device=device
    )


def index_observations(
    observations: TorchObservation,
    indices: np.ndarray,
) -> TorchObservation:
    """Select PPO minibatch rows from tensor or structured observations."""
    if isinstance(observations, dict):
        return {key: value[indices] for key, value in observations.items()}
    return observations[indices]


def observation_image(observation: NumpyObservation) -> np.ndarray:
    """Return the private RGB reconstruction target from either observation form."""
    return observation["rgb"] if isinstance(observation, dict) else observation


def observation_goal(observation: NumpyObservation) -> Optional[np.ndarray]:
    """Return the point-goal modality when present."""
    return observation.get("goal") if isinstance(observation, dict) else None

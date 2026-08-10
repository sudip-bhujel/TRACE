from collections import namedtuple
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

Transition = namedtuple(
    "Transition", ["obs", "action", "logp", "reward", "done", "value"]
)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages and returns. ``values`` must include a bootstrap entry."""
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def conv_block(in_c: int, out_c: int, k: int = 3, s: int = 2, p: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.BatchNorm2d(out_c),
    )


class _MpsSafeFlattenFunction(torch.autograd.Function):
    """Flatten while making both sides of the autograd boundary contiguous."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = x.shape
        return x.contiguous().reshape(x.size(0), -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.contiguous().reshape(ctx.input_shape)


class _ReshapeFlatten(nn.Module):
    # MPS may supply a strided Linear gradient that ViewBackward cannot reshape.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _MpsSafeFlattenFunction.apply(x)


class ActorCritic(nn.Module):
    """Shared CNN encoder with policy and value heads."""

    def __init__(
        self, in_channels: int = 3, num_actions: int = 5, hidden_size: int = 512
    ):
        super().__init__()
        self.architecture = "cnn"
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            conv_block(in_channels, 32, k=8, s=4, p=2),
            conv_block(32, 64, k=4, s=2, p=1),
            conv_block(64, 64, k=3, s=1, p=1),
            conv_block(64, 64, k=3, s=2, p=1),
            _ReshapeFlatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_size, num_actions)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float() / 255.0
        z = self.encoder(x)
        h = self.fc(z)
        return self.policy(h), self.value(h).squeeze(-1)


class RecurrentActorCritic(nn.Module):
    """CNN actor-critic with a GRU state carried across environment steps."""

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 5,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.architecture = "cnn_gru"
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.is_recurrent = True

        self.encoder = nn.Sequential(
            conv_block(in_channels, 32, k=8, s=4, p=2),
            conv_block(32, 64, k=4, s=2, p=1),
            conv_block(64, 64, k=3, s=1, p=1),
            conv_block(64, 64, k=3, s=2, p=1),
            _ReshapeFlatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, num_actions)
        self.value = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.float() / 255.0
        features = self.fc(self.encoder(x))
        if recurrent_state is None:
            recurrent_state = self.initial_state(features.shape[0], features.device)
        hidden = self.gru(features, recurrent_state)
        return self.policy(hidden), self.value(hidden).squeeze(-1), hidden

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward_recurrent(x)
        return logits, value


def forward_actor_critic(
    model: nn.Module,
    observations: torch.Tensor,
    recurrent_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run feed-forward and recurrent victims through one explicit interface."""
    if getattr(model, "is_recurrent", False):
        logits, value, next_state = model.forward_recurrent(
            observations, recurrent_state
        )
        return logits, value, next_state
    logits, value = model(observations)
    return logits, value, None


def initial_recurrent_state(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if getattr(model, "is_recurrent", False):
        return model.initial_state(batch_size, device)
    return None


class _ImpalaResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + residual


class _ImpalaConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual1 = _ImpalaResidualBlock(out_channels)
        self.residual2 = _ImpalaResidualBlock(out_channels)
        # Batch-independent normalization keeps the residual stream stable on
        # both single-observation capture and small PPO minibatches.
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x))
        x = self.residual1(x)
        return self.norm(self.residual2(x))


class IMPALAActorCritic(nn.Module):
    """Lightweight IMPALA-style residual CNN with shared policy/value features."""

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 5,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.architecture = "impala_cnn"
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            _ImpalaConvSequence(in_channels, 16),
            _ImpalaConvSequence(16, 32),
            _ImpalaConvSequence(32, 32),
            nn.ReLU(),
            _ReshapeFlatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            feature_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, num_actions)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float() / 255.0
        features = self.encoder(x)
        hidden = self.fc(features)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class TinyViTActorCritic(nn.Module):
    """Small ViT victim with the same policy/value interface as the CNN."""

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 5,
        image_size: int = 84,
        patch_size: int = 14,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

        self.architecture = "tiny_vit"
        self.num_actions = num_actions
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.policy = nn.Linear(embed_dim, num_actions)
        self.value = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"TinyViT expects {self.image_size}x{self.image_size} observations, "
                f"got {tuple(x.shape[-2:])}"
            )

        x = x.float() / 255.0
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.encoder(tokens + self.pos_embed)
        features = self.norm(tokens[:, 0])
        return self.policy(features), self.value(features).squeeze(-1)


def build_actor_critic(
    architecture: str = "cnn",
    in_channels: int = 3,
    num_actions: int = 5,
) -> nn.Module:
    """Build a victim while preserving a common PPO/capture interface."""
    architecture = architecture.lower()
    if architecture == "cnn":
        return ActorCritic(in_channels=in_channels, num_actions=num_actions)
    if architecture in {"cnn_gru", "recurrent_cnn", "gru"}:
        return RecurrentActorCritic(
            in_channels=in_channels,
            num_actions=num_actions,
        )
    if architecture in {"impala_cnn", "impala"}:
        return IMPALAActorCritic(
            in_channels=in_channels,
            num_actions=num_actions,
        )
    if architecture in {"tiny_vit", "tinyvit"}:
        return TinyViTActorCritic(
            in_channels=in_channels,
            num_actions=num_actions,
        )
    raise ValueError(
        f"Unknown victim architecture '{architecture}'. "
        "Choose: cnn, cnn_gru, impala_cnn, tiny_vit"
    )


# A2C shares architecture with ActorCritic; only the training objective differs.
A2C = ActorCritic

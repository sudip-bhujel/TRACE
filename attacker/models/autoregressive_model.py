"""Autoregressive gradient inversion model with interleaved gradient/image tokens."""

import random
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder
from attacker.models.image_encoder import ImageEncoder
from attacker.models.temporal_baselines import get_temporal_model


class AutoregressiveGradientInversion(nn.Module):
    """
    Builds an interleaved sequence ``[z_1, e_0, z_2, e_1, ..., z_T, e_{T-1}]`` where
    ``z_t`` is the encoded gradient and ``e_t`` is the encoded image (``e_0`` is a
    learned start token). A causal transformer reads gradient and previous-image
    context; predictions are extracted at gradient positions.
    """

    def __init__(
        self,
        gradient_dim: int,
        latent_dim: int = 512,
        num_actions: int = 5,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        encoder_hidden_dims: Optional[List[int]] = None,
        encoder_hidden_dim: Optional[int] = None,
        encoder_num_blocks: Optional[int] = None,
        encoder_expansion: Optional[int] = None,
        encoder_projection_rank: Optional[int] = None,
        encoder_type: str = "basic",
        decoder_type: str = "basic",
        dropout: float = 0.1,
        image_size: int = 84,
        temporal_model_type: str = "transformer",
        ff_multiplier: int = 4,
        use_rope: bool = False,
        **decoder_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_kwargs = {}
        if encoder_type == "basic" and encoder_hidden_dims is not None:
            encoder_kwargs["hidden_dims"] = encoder_hidden_dims
        if encoder_type == "residual":
            if encoder_hidden_dim is not None:
                encoder_kwargs["hidden_dim"] = encoder_hidden_dim
            if encoder_num_blocks is not None:
                encoder_kwargs["num_blocks"] = encoder_num_blocks
            if encoder_expansion is not None:
                encoder_kwargs["expansion"] = encoder_expansion
            encoder_kwargs["projection_rank"] = encoder_projection_rank

        self.gradient_encoder = get_encoder(
            encoder_type=encoder_type,
            gradient_dim=gradient_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            **encoder_kwargs,
        )

        self.image_encoder = ImageEncoder(
            latent_dim=latent_dim,
            image_size=image_size,
        )

        self.start_token = nn.Parameter(torch.randn(latent_dim) * 0.02)

        self.type_embedding = nn.Embedding(2, latent_dim)
        nn.init.normal_(self.type_embedding.weight, std=0.02)

        self.temporal_model = get_temporal_model(
            temporal_model_type=temporal_model_type,
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            max_seq_len=64,
            is_causal=True,
            use_rope=use_rope,
        )

        self.image_decoder = get_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            image_size=image_size,
            num_actions=num_actions,
            **decoder_kwargs,
        )

    def _build_interleaved_sequence(
        self,
        gradient_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = gradient_embeds.shape
        device = gradient_embeds.device

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        gradient_embeds = gradient_embeds + grad_type
        image_embeds = image_embeds + img_type

        interleaved = torch.stack([gradient_embeds, image_embeds], dim=2)
        return interleaved.reshape(B, 2 * T, D)

    def forward_teacher_forced(
        self,
        gradients: torch.Tensor,
        images: torch.Tensor,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Teacher-forced training pass: GT images shifted right are used as context."""
        B, T, _ = gradients.shape

        gradient_embeds = self.gradient_encoder(gradients)

        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        if T > 1:
            prev_image_embeds = self.image_encoder(images[:, :-1])
            image_embeds = torch.cat([start, prev_image_embeds], dim=1)
        else:
            image_embeds = start

        interleaved = self._build_interleaved_sequence(gradient_embeds, image_embeds)

        transformer_out = self.temporal_model(
            interleaved, use_flash_attention=use_flash_attention
        )

        grad_positions = torch.arange(0, 2 * T, 2, device=gradients.device)
        latents = transformer_out[:, grad_positions, :]

        output = self.image_decoder(latents)
        if len(output) == 3:
            pred_images, pred_actions, _ = output
        else:
            pred_images, pred_actions = output

        return pred_images, pred_actions, latents, None

    def forward_scheduled_sampling(
        self,
        gradients: torch.Tensor,
        images: torch.Tensor,
        sampling_prob: float = 0.5,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """At each step, with probability ``sampling_prob`` use the model's predicted image as context."""
        _, T, _ = gradients.shape
        device = gradients.device

        gradient_embeds = self.gradient_encoder(gradients)

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        # Freeze BN running stats so they don't conflict with the concurrent teacher-forced graph.
        decoder_was_training = self.image_decoder.training
        encoder_was_training = self.image_encoder.training
        self.image_decoder.eval()
        self.image_encoder.eval()

        all_images = []
        all_actions = []
        all_latents = []
        context_tokens = []

        for t in range(T):
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)

            context = torch.cat(context_tokens, dim=1)
            transformer_out = self.temporal_model(
                context, use_flash_attention=use_flash_attention
            )

            h_t = transformer_out[:, -1:, :]
            all_latents.append(h_t)

            output = self.image_decoder(h_t)
            if len(output) == 3:
                img_t, act_t, _ = output
            else:
                img_t, act_t = output

            all_images.append(img_t)
            all_actions.append(act_t)

            if t < T - 1:
                use_predicted = random.random() < sampling_prob
                if use_predicted:
                    img_embed = self.image_encoder(img_t.detach())
                else:
                    img_embed = self.image_encoder(images[:, t : t + 1])
                e_t = img_embed + img_type
                context_tokens.append(e_t)

        if decoder_was_training:
            self.image_decoder.train()
        if encoder_was_training:
            self.image_encoder.train()

        pred_images = torch.cat(all_images, dim=1)
        pred_actions = torch.cat(all_actions, dim=1)
        latents = torch.cat(all_latents, dim=1)

        return pred_images, pred_actions, latents, None

    def forward_rollout(
        self,
        gradients: torch.Tensor,
        images: torch.Tensor,
        rollout_steps: int = 2,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Short autoregressive rollout from a random starting timestep, with gradients flowing through it."""
        B, T, _ = gradients.shape
        device = gradients.device

        max_start = max(0, T - rollout_steps)
        t_start = random.randint(0, max_start)

        gradient_embeds = self.gradient_encoder(gradients)

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        # Freeze BN running stats so the teacher-forced graph's autograd state is not corrupted.
        decoder_was_training = self.image_decoder.training
        encoder_was_training = self.image_encoder.training
        self.image_decoder.eval()
        self.image_encoder.eval()

        start_embed = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        context_tokens = []
        for t in range(t_start):
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)
            if t == 0:
                e_t = start_embed + img_type
            else:
                e_t = self.image_encoder(images[:, t - 1 : t]) + img_type
            context_tokens.append(e_t)

        ro_images = []
        ro_actions = []
        ro_latents = []

        for k in range(rollout_steps):
            t = t_start + k
            if t >= T:
                break

            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)

            context = torch.cat(context_tokens, dim=1)
            transformer_out = self.temporal_model(
                context, use_flash_attention=use_flash_attention
            )

            h_t = transformer_out[:, -1:, :]
            ro_latents.append(h_t)

            output = self.image_decoder(h_t)
            if len(output) == 3:
                img_t, act_t, _ = output
            else:
                img_t, act_t = output

            ro_images.append(img_t)
            ro_actions.append(act_t)

            # Detach predicted image: prevents inplace BN version conflicts via the decoder→encoder chain.
            if k < rollout_steps - 1 and t < T - 1:
                e_t = self.image_encoder(img_t.detach()) + img_type
                context_tokens.append(e_t)

        if decoder_was_training:
            self.image_decoder.train()
        if encoder_was_training:
            self.image_encoder.train()

        pred_images = torch.cat(ro_images, dim=1)
        pred_actions = torch.cat(ro_actions, dim=1)
        latents = torch.cat(ro_latents, dim=1)

        return pred_images, pred_actions, latents, t_start

    def forward_autoregressive(
        self,
        gradients: torch.Tensor,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Inference-mode autoregressive forward: predicted images are fed back as context."""
        _, T, _ = gradients.shape
        device = gradients.device

        gradient_embeds = self.gradient_encoder(gradients)

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        all_images = []
        all_actions = []
        all_latents = []
        context_tokens = []

        for t in range(T):
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)

            context = torch.cat(context_tokens, dim=1)
            transformer_out = self.temporal_model(
                context, use_flash_attention=use_flash_attention
            )

            h_t = transformer_out[:, -1:, :]
            all_latents.append(h_t)

            output = self.image_decoder(h_t)
            if len(output) == 3:
                img_t, act_t, _ = output
            else:
                img_t, act_t = output

            all_images.append(img_t)
            all_actions.append(act_t)

            if t < T - 1:
                with torch.no_grad():
                    img_embed = self.image_encoder(img_t)
                e_t = img_embed + img_type
                context_tokens.append(e_t)

        pred_images = torch.cat(all_images, dim=1)
        pred_actions = torch.cat(all_actions, dim=1)
        latents = torch.cat(all_latents, dim=1)

        return pred_images, pred_actions, latents, None

    def forward(
        self,
        gradients: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        sampling_prob: float = 0.0,
        rollout_steps: int = 0,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Dispatch to teacher-forced, scheduled-sampling, or autoregressive forward.
        Optional rollout output is returned alongside the primary output in a single
        call so DDP backward hooks fire only once per shared parameter per step.
        """
        if not teacher_forcing:
            return self.forward_autoregressive(gradients, use_flash_attention)

        assert images is not None, "images must be provided when teacher_forcing=True"

        if sampling_prob > 0.0:
            pred_images, pred_actions, latents, _ = self.forward_scheduled_sampling(
                gradients, images, sampling_prob, use_flash_attention
            )
        else:
            pred_images, pred_actions, latents, _ = self.forward_teacher_forced(
                gradients, images, use_flash_attention
            )

        if rollout_steps > 0:
            ro_result = self.forward_rollout(
                gradients, images, rollout_steps, use_flash_attention
            )
            return pred_images, pred_actions, latents, ro_result

        return pred_images, pred_actions, latents, None

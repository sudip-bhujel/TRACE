"""
Autoregressive Gradient Inversion Model.

Predicts images one timestep at a time, conditioning each prediction
on all previous gradient embeddings and (ground-truth or predicted) images
via a causal transformer.

Training uses teacher forcing (ground-truth images as context).
Inference feeds back predicted images autoregressively.
"""

import random
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from attacker.models.decoder import get_decoder
from attacker.models.encoder import get_encoder
from attacker.models.image_encoder import ImageEncoder
from attacker.models.temporal_baselines import get_temporal_model


class AutoregressiveGradientInversion(nn.Module):
    """Autoregressive gradient inversion model.

    Constructs an interleaved sequence of gradient and image tokens:
        [z_1, e_0, z_2, e_1, z_3, e_2, ...]

    where z_t = GradientEncoder(g_t) and e_t = ImageEncoder(i_t).
    The first image token e_0 is a learned start token.

    The causal transformer processes this sequence and predictions are
    extracted at the gradient token positions (indices 0, 2, 4, ...).
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

        # Gradient encoder (reused from existing codebase)
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

        # Image encoder (new)
        self.image_encoder = ImageEncoder(
            latent_dim=latent_dim,
            image_size=image_size,
        )

        # Learned start token (used as e_0, the image context before z_1)
        self.start_token = nn.Parameter(torch.randn(latent_dim) * 0.02)

        # Type embeddings to distinguish gradient vs image tokens
        self.type_embedding = nn.Embedding(2, latent_dim)  # 0=gradient, 1=image
        nn.init.normal_(self.type_embedding.weight, std=0.02)

        # Causal temporal model (transformer)
        # max_seq_len is 2*T for the interleaved sequence
        self.temporal_model = get_temporal_model(
            temporal_model_type=temporal_model_type,
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            max_seq_len=64,  # 2*T, handles up to T=32
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
        """Build interleaved sequence [z_1, e_0, z_2, e_1, ..., z_T, e_{T-1}].

        Args:
            gradient_embeds: (B, T, D) — encoded gradients
            image_embeds: (B, T, D) — image embeddings where index 0 is the
                start token and indices 1..T-1 are encoded images i_1..i_{T-1}

        Returns:
            (B, 2*T, D) — interleaved sequence
        """
        B, T, D = gradient_embeds.shape
        device = gradient_embeds.device

        # Create type embeddings
        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )  # (1, 1, D)
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )  # (1, 1, D)

        # Add type embeddings
        gradient_embeds = gradient_embeds + grad_type
        image_embeds = image_embeds + img_type

        # Interleave: [z_1, e_0, z_2, e_1, ..., z_T, e_{T-1}]
        interleaved = torch.stack(
            [gradient_embeds, image_embeds], dim=2
        )  # (B, T, 2, D)
        interleaved = interleaved.reshape(B, 2 * T, D)

        return interleaved

    def forward_teacher_forced(
        self,
        gradients: torch.Tensor,
        images: torch.Tensor,
        use_flash_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Forward pass with teacher forcing (training mode).

        Uses ground-truth images as context for all timesteps.
        All timesteps processed in parallel via causal masking.

        Args:
            gradients: (B, T, gradient_dim)
            images: (B, T, 3, H, W) — ground-truth images
            use_flash_attention: Flash Attention toggle.

        Returns:
            pred_images: (B, T, 3, H, W)
            pred_actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
            None (for API compat)
        """
        B, T, _ = gradients.shape

        # Encode gradients
        gradient_embeds = self.gradient_encoder(gradients)  # (B, T, D)

        # Encode images: shift right by 1 (use start_token for t=0)
        # images[:, :-1] gives i_1 ... i_{T-1}, used as context for z_2 ... z_T
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B,1,D)
        if T > 1:
            prev_image_embeds = self.image_encoder(images[:, :-1])  # (B, T-1, D)
            image_embeds = torch.cat([start, prev_image_embeds], dim=1)  # (B, T, D)
        else:
            image_embeds = start  # (B, 1, D)

        # Build interleaved sequence and run through transformer
        interleaved = self._build_interleaved_sequence(
            gradient_embeds, image_embeds
        )  # (B, 2T, D)

        transformer_out = self.temporal_model(
            interleaved, use_flash_attention=use_flash_attention
        )  # (B, 2T, D)

        # Extract outputs at gradient positions (0, 2, 4, ...)
        grad_positions = torch.arange(0, 2 * T, 2, device=gradients.device)
        latents = transformer_out[:, grad_positions, :]  # (B, T, D)

        # Decode
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
        """Forward pass with scheduled sampling.

        Processes timesteps sequentially.  At each step after the first,
        a coin flip decides whether to use the ground-truth previous image
        or the model's own predicted image as context.  This reduces the
        train-infer mismatch (exposure bias).

        Args:
            gradients: (B, T, gradient_dim)
            images: (B, T, 3, H, W) — ground-truth images
            sampling_prob: probability of using predicted (not GT) image
                as context for the next step.  0.0 = pure teacher forcing,
                1.0 = fully autoregressive training.
            use_flash_attention: Flash Attention toggle.

        Returns:
            pred_images: (B, T, 3, H, W)
            pred_actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
            None (for API compat)
        """
        _, T, _ = gradients.shape
        device = gradients.device

        gradient_embeds = self.gradient_encoder(gradients)  # (B, T, D)

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        # Freeze BN running stats to avoid inplace version conflicts
        # when this sequential path coexists with teacher-forced forward.
        decoder_was_training = self.image_decoder.training
        encoder_was_training = self.image_encoder.training
        self.image_decoder.eval()
        self.image_encoder.eval()

        all_images = []
        all_actions = []
        all_latents = []
        context_tokens = []

        for t in range(T):
            # Gradient token
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)

            # Run transformer on current context
            context = torch.cat(context_tokens, dim=1)
            transformer_out = self.temporal_model(
                context, use_flash_attention=use_flash_attention
            )

            h_t = transformer_out[:, -1:, :]
            all_latents.append(h_t)

            # Decode
            output = self.image_decoder(h_t)
            if len(output) == 3:
                img_t, act_t, _ = output
            else:
                img_t, act_t = output

            all_images.append(img_t)
            all_actions.append(act_t)

            # Image context for next step
            if t < T - 1:
                use_predicted = random.random() < sampling_prob
                if use_predicted:
                    # Use model's own prediction (detached to avoid
                    # backprop through the full autoregressive chain)
                    img_embed = self.image_encoder(img_t.detach())
                else:
                    # Use ground-truth image
                    img_embed = self.image_encoder(images[:, t : t + 1])
                e_t = img_embed + img_type
                context_tokens.append(e_t)

        # Restore BN training state
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
        """Forward pass with short autoregressive rollout loss.

        Uses ground-truth images as context up to a random starting
        timestep, then unrolls ``rollout_steps`` autoregressive
        predictions (feeding predictions back as context).  Gradients
        flow through the entire rollout so the model learns to recover
        from its own errors.

        NOTE: The image_decoder and image_encoder are set to eval mode
        during rollout to prevent BatchNorm from updating running_mean /
        running_var inplace, which would corrupt the autograd graph of
        the concurrent teacher-forced forward pass.

        Args:
            gradients: (B, T, gradient_dim)
            images: (B, T, 3, H, W) — ground-truth images
            rollout_steps: number of autoregressive steps to unroll.
            use_flash_attention: Flash Attention toggle.

        Returns:
            pred_images: (B, rollout_steps, 3, H, W) — predictions for
                the rollout window only.
            pred_actions: (B, rollout_steps, num_actions)
            latents: (B, rollout_steps, latent_dim)
            t_start: int — starting timestep of the rollout window
                (caller uses this to slice ground-truth targets).
        """
        B, T, _ = gradients.shape
        device = gradients.device

        # Ensure rollout fits within the sequence
        max_start = max(0, T - rollout_steps)
        t_start = random.randint(0, max_start)

        gradient_embeds = self.gradient_encoder(gradients)  # (B, T, D)

        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        # --- Freeze BN running stats for the rollout ---
        # This prevents inplace updates to running_mean/running_var that
        # would conflict with the teacher-forced graph's autograd state.
        decoder_was_training = self.image_decoder.training
        encoder_was_training = self.image_encoder.training
        self.image_decoder.eval()
        self.image_encoder.eval()

        # --- Build GT context up to t_start (parallel, no grad needed) ---
        start_embed = (
            self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        )  # (B, 1, D)

        context_tokens = []
        for t in range(t_start):
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type
            context_tokens.append(z_t)
            if t == 0:
                e_t = start_embed + img_type
            else:
                e_t = self.image_encoder(images[:, t - 1 : t]) + img_type
            context_tokens.append(e_t)

        # --- Autoregressive rollout from t_start ---
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

            # Feed predicted image back as context for next rollout step.
            # Detach to prevent gradient flow backward through the
            # decoder→encoder chain, which causes inplace BatchNorm
            # version conflicts with the concurrent teacher-forced graph.
            if k < rollout_steps - 1 and t < T - 1:
                e_t = self.image_encoder(img_t.detach()) + img_type
                context_tokens.append(e_t)

        # --- Restore BN training state ---
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
        """Forward pass with autoregressive inference.

        Predicts images one at a time, feeding each predicted image
        back as context for the next timestep.

        Args:
            gradients: (B, T, gradient_dim)
            use_flash_attention: Flash Attention toggle.

        Returns:
            pred_images: (B, T, 3, H, W)
            pred_actions: (B, T, num_actions)
            latents: (B, T, latent_dim)
            None (for API compat)
        """
        _, T, _ = gradients.shape
        device = gradients.device

        # Encode all gradients upfront
        gradient_embeds = self.gradient_encoder(gradients)  # (B, T, D)

        # Type embeddings
        grad_type = self.type_embedding(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        img_type = self.type_embedding(
            torch.ones(1, 1, dtype=torch.long, device=device)
        )

        all_images = []
        all_actions = []
        all_latents = []

        # Build context incrementally
        context_tokens = []

        for t in range(T):
            # Add gradient token for timestep t
            z_t = gradient_embeds[:, t : t + 1, :] + grad_type  # (B, 1, D)
            context_tokens.append(z_t)

            # Run transformer on current context
            context = torch.cat(context_tokens, dim=1)  # (B, 2t+1, D)
            transformer_out = self.temporal_model(
                context, use_flash_attention=use_flash_attention
            )

            # Last token output corresponds to current gradient
            h_t = transformer_out[:, -1:, :]  # (B, 1, D)
            all_latents.append(h_t)

            # Decode to image
            output = self.image_decoder(h_t)
            if len(output) == 3:
                img_t, act_t, _ = output
            else:
                img_t, act_t = output

            all_images.append(img_t)
            all_actions.append(act_t)

            # Encode predicted image and add as context for next step
            if t < T - 1:
                with torch.no_grad():
                    img_embed = self.image_encoder(img_t)  # (B, 1, D)
                e_t = img_embed + img_type  # (B, 1, D)
                context_tokens.append(e_t)

        # Stack along time dimension
        pred_images = torch.cat(all_images, dim=1)  # (B, T, 3, H, W)
        pred_actions = torch.cat(all_actions, dim=1)  # (B, T, num_actions)
        latents = torch.cat(all_latents, dim=1)  # (B, T, D)

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
        """Forward pass dispatching to the appropriate mode.

        When rollout_steps > 0 and teacher_forcing is True, this method
        computes BOTH the primary output (teacher-forced or scheduled
        sampling) AND the rollout output in a single call.  This is
        required for DDP compatibility — calling the model twice per
        iteration causes DDP's per-parameter backward hooks to fire
        twice for shared parameters (e.g. start_token).

        Args:
            gradients: (B, T, gradient_dim)
            images: (B, T, 3, H, W) — required when teacher_forcing=True
            teacher_forcing: If True, use ground-truth images as context.
            sampling_prob: Scheduled sampling probability. Only used when
                teacher_forcing=True. 0.0 = pure teacher forcing.
            rollout_steps: If > 0 and teacher_forcing=True, also compute
                rollout predictions, returned in the 4th element.
            use_flash_attention: Flash Attention toggle.

        Returns:
            pred_images, pred_actions, latents, aux
            When rollout_steps > 0: aux is a tuple
                (ro_images, ro_actions, ro_latents, t_start)
            Otherwise: aux is None
        """
        if not teacher_forcing:
            return self.forward_autoregressive(gradients, use_flash_attention)

        assert images is not None, "images must be provided when teacher_forcing=True"

        # Primary forward pass
        if sampling_prob > 0.0:
            pred_images, pred_actions, latents, _ = self.forward_scheduled_sampling(
                gradients, images, sampling_prob, use_flash_attention
            )
        else:
            pred_images, pred_actions, latents, _ = self.forward_teacher_forced(
                gradients, images, use_flash_attention
            )

        # Optional rollout (computed in the same DDP forward call)
        if rollout_steps > 0:
            ro_result = self.forward_rollout(
                gradients, images, rollout_steps, use_flash_attention
            )
            return pred_images, pred_actions, latents, ro_result

        return pred_images, pred_actions, latents, None

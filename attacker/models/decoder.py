"""
Image Decoder Architectures for Gradient Inversion

This module provides multiple decoder architectures for decoding latent
representations into images and action predictions.

Available decoders:
- ImageDecoder: Basic transposed convolution decoder
- ResidualImageDecoder: Decoder with residual blocks for better detail
- UNetImageDecoder: U-Net style decoder with skip connections
- StyleImageDecoder: StyleGAN-inspired decoder with adaptive instance norm
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoder(nn.Module):
    """Basic transposed convolution decoder."""

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4  # 21

        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # Transposed convolutions for image generation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 21 -> 42
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 42 -> 84
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),  # 84 -> 84
            nn.Sigmoid(),
        )

        # Action prediction head
        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)
        images = self.decoder(h)

        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class ResidualBlock2d(nn.Module):
    """2D Residual block for image decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class ResidualImageDecoder(nn.Module):
    """
    Decoder with residual blocks for better reconstruction quality.

    Adds residual blocks after each upsampling layer to preserve details
    and improve gradient flow during training.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4

        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # First upsample block with residuals
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.res1 = nn.Sequential(
            *[ResidualBlock2d(128) for _ in range(num_res_blocks)]
        )

        # Second upsample block with residuals
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = nn.Sequential(*[ResidualBlock2d(64) for _ in range(num_res_blocks)])

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        h = F.relu(self.bn1(self.up1(h)))
        h = self.res1(h)

        h = F.relu(self.bn2(self.up2(h)))
        h = self.res2(h)

        images = self.final(h)
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class UNetImageDecoder(nn.Module):
    """
    Improved U-Net style decoder with multi-level skip connections.

    Features:
    - Larger initial spatial size (image_size // 4) for better detail
    - Residual blocks at each decoder level
    - Multi-level skip connections for feature reuse
    - Optional attention at bottleneck for global context
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        use_attention: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4  # 21

        # Project to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        # Encoder path (creates features for skip connections)
        self.enc1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock2d(256),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1),  # Downsample: 21 -> 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock2d(128),
        )

        # Bottleneck with optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )
        self.bottleneck = nn.Sequential(
            ResidualBlock2d(128),
            ResidualBlock2d(128),
        )

        # Decoder path with skip connections
        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)  # 10 -> 20
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1),  # Concat with enc1 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock2d(128),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 21 -> 42
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock2d(64),
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Initial projection: (B*T, latent_dim) -> (B*T, 256, 21, 21)
        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        # Encoder with skip connections
        e1 = self.enc1(h)  # (B*T, 256, 21, 21) - save for skip
        e2 = self.enc2(e1)  # (B*T, 128, 10, 10)

        # Bottleneck with optional attention
        if self.use_attention:
            b, c, h_size, w_size = e2.shape
            e2_flat = e2.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
            e2_attn, _ = self.attention(e2_flat, e2_flat, e2_flat)
            e2 = e2_attn.permute(0, 2, 1).view(b, c, h_size, w_size)

        bottleneck = self.bottleneck(e2)  # (B*T, 128, 10, 10)

        # Decoder with skip connections
        d1 = self.up1(bottleneck)  # (B*T, 128, 20, 20)
        # Resize e1 to match d1 for skip connection
        e1_resized = F.interpolate(
            e1, size=d1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = torch.cat([d1, e1_resized], dim=1)  # (B*T, 128+256, 20, 20)
        d1 = self.dec1(d1)  # (B*T, 128, 20, 20)

        d2 = self.up2(d1)  # (B*T, 64, 40, 40)
        d2 = self.dec2(d2)  # (B*T, 64, 40, 40)

        images = self.final(d2)  # (B*T, 3, 40, 40)

        # Resize to exact target size
        if images.shape[-1] != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style-based generation."""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style = self.style(style)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


class StyleBlock(nn.Module):
    """Style-modulated convolution block."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adain(x, style)
        return self.activation(x)


class StyleImageDecoder(nn.Module):
    """
    StyleGAN-inspired decoder with Adaptive Instance Normalization.

    Uses style vectors to modulate the feature maps, allowing for
    better control over generated image characteristics.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        style_dim: int = 256,
    ):
        super().__init__()

        self.image_size = image_size
        self.init_size = image_size // 4

        # Mapping network (latent -> style)
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initial constant input
        self.const = nn.Parameter(torch.randn(1, 256, 1, 1))

        # Style blocks
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)

        self.style1 = StyleBlock(256, 128, style_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.style2 = StyleBlock(128, 64, style_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.style3 = StyleBlock(64, 32, style_dim)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

        self.action_head = nn.Linear(latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        has_time_dim = x.dim() == 3

        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        # Generate style vector
        style = self.mapping(x_flat)

        # Project to initial feature map
        h = self.fc(x_flat)
        h = h.view(-1, 256, self.init_size, self.init_size)

        # Style-modulated upsampling
        h = self.style1(h, style)
        h = self.up1(h)

        h = self.style2(h, style)
        h = self.up2(h)

        h = self.style3(h, style)

        images = self.to_rgb(h)
        actions = self.action_head(x_flat)

        if has_time_dim:
            images = images.view(B, T, 3, self.image_size, self.image_size)
            actions = actions.view(B, T, -1)

        return images, actions


class VQTokenDecoder(nn.Module):
    """
    VQ-GAN Token Prediction Decoder with Spatial Architecture.

    Instead of predicting raw pixels, predicts discrete visual tokens
    from a pretrained VQ-VAE codebook. Uses a spatial convolutional
    network to predict tokens with local context awareness.

    Architecture:
        latent (D,) -> project to (C, 7, 7) spatial grid
        -> ResBlocks for spatial reasoning -> 1x1 conv -> (codebook_size, 7, 7)

    This decouples:
    - Gradient interpretation (trainable spatial token predictor)
    - Visual generation (frozen pretrained VQ decoder)

    Args:
        latent_dim: Input latent dimension from transformer
        vqvae_checkpoint: Path to trained VQ-VAE checkpoint
        num_tokens: Number of tokens per image (spatial_h * spatial_w)
        codebook_size: VQ-VAE codebook size
        num_actions: Number of action classes
        gumbel_tau: Gumbel-Softmax temperature (lower = harder)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        image_size: int = 84,
        num_actions: int = 5,
        vqvae_checkpoint: str = "ckpts/vqvae/best_model.pt",
        num_tokens: int = 49,
        codebook_size: int = 512,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.gumbel_tau = gumbel_tau
        self.spatial_size = int(num_tokens**0.5)  # 7 for 49 tokens

        # Load frozen VQ-VAE
        from attacker.models.vqvae import load_vqvae

        self.vqvae = load_vqvae(vqvae_checkpoint, device="cpu")
        for p in self.vqvae.parameters():
            p.requires_grad = False

        embedding_dim = self.vqvae.config["embedding_dim"]

        # Spatial token prediction network
        # Step 1: Project latent -> spatial feature grid (C, 7, 7)
        spatial_channels = 256
        self.spatial_proj = nn.Sequential(
            nn.Linear(
                latent_dim, spatial_channels * self.spatial_size * self.spatial_size
            ),
            nn.GELU(),
        )

        # Step 2: Spatial reasoning with residual conv blocks
        # Neighboring tokens share information through 3x3 convolutions
        self.spatial_refine = nn.Sequential(
            ResidualBlock2d(spatial_channels),
            ResidualBlock2d(spatial_channels),
            ResidualBlock2d(spatial_channels),
        )

        # Step 3: Per-position token prediction via 1x1 conv
        # Each spatial position independently predicts its codebook token
        self.token_head = nn.Sequential(
            nn.Conv2d(spatial_channels, spatial_channels, 1),
            nn.GELU(),
            nn.Conv2d(spatial_channels, codebook_size, 1),
        )

        # Action prediction head
        self.action_head = nn.Linear(latent_dim, num_actions)

        # Store embedding dim for forward pass
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, latent_dim) or (B*T, latent_dim)

        Returns:
            images: (B, T, 3, H, W) decoded images
            actions: (B, T, num_actions) action logits
            token_logits: (B*T, num_tokens, codebook_size) for CE loss
        """
        has_time_dim = x.dim() == 3
        if has_time_dim:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            B = x.shape[0]
            T = 1
            x_flat = x

        # Project to spatial grid: (B*T, 256, 7, 7)
        spatial = self.spatial_proj(x_flat)
        spatial = spatial.view(-1, 256, self.spatial_size, self.spatial_size)

        # Spatial reasoning with local context: (B*T, 256, 7, 7)
        spatial = self.spatial_refine(spatial)

        # Per-position token logits: (B*T, codebook_size, 7, 7)
        logits_2d = self.token_head(spatial)

        # Reshape to (B*T, num_tokens, codebook_size)
        token_logits = logits_2d.view(-1, self.codebook_size, self.num_tokens)
        token_logits = token_logits.permute(0, 2, 1)  # (B*T, num_tokens, codebook_size)

        # Get quantized vectors via Gumbel-Softmax (differentiable) or argmax
        if self.training:
            soft_tokens = F.gumbel_softmax(
                token_logits, tau=self.gumbel_tau, hard=True, dim=-1
            )
            # Lookup embeddings: (B*T, num_tokens, embed_dim)
            quantized = soft_tokens @ self.vqvae.quantizer.embedding.weight
        else:
            indices = token_logits.argmax(dim=-1)  # (B*T, num_tokens)
            quantized = self.vqvae.quantizer.embedding(indices)

        # Reshape to spatial grid: (B*T, embed_dim, H', W')
        quantized = quantized.view(
            -1, self.spatial_size, self.spatial_size, self.embedding_dim
        ).permute(0, 3, 1, 2)

        # Decode with frozen VQ decoder
        # Gradients flow through Gumbel-Softmax -> embedding lookup -> decoder
        # (decoder weights frozen, but quantized input carries grad)
        images_with_grad = self.vqvae.decoder(quantized)

        # Actions
        actions = self.action_head(x_flat)

        if has_time_dim:
            images_with_grad = images_with_grad.view(B, T, *images_with_grad.shape[1:])
            actions = actions.view(B, T, -1)

        return images_with_grad, actions, token_logits

    def set_tau(self, tau: float):
        """Update Gumbel-Softmax temperature for annealing."""
        self.gumbel_tau = tau


# ============================================================================
# Flow Matching Decoder
# ============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FlowResBlock(nn.Module):
    """Residual block with time+condition modulation for flow matching U-Net."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Modulation from time+condition embedding
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels * 2),  # scale and shift
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), emb: (B, emb_dim)"""
        h = self.block[0](x)  # GroupNorm
        h = self.block[1](h)  # SiLU
        h = self.block[2](h)  # Conv

        # Apply scale+shift from embedding
        scale_shift = self.emb_proj(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.block[3](h) * (1 + scale) + shift  # GroupNorm with modulation
        h = self.block[4](h)  # SiLU
        h = self.block[5](h)  # Conv

        return h + x


class FlowImageDecoder(nn.Module):
    """
    Conditional Flow Matching decoder.

    Instead of directly regressing pixels, this decoder learns a velocity field
    v(x_t, t, condition) that transforms noise into images via an ODE.

    Architecture: Small conditional U-Net with time+condition modulation.

    Training: predict velocity v = x_1 - x_0 given interpolation x_t = (1-t)*x_0 + t*x_1
    Inference: integrate ODE from noise x_0 to image x_1 using learned velocity field
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        image_size: int = 84,
        num_actions: int = 5,
        base_channels: int = 64,
        num_sampling_steps: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_sampling_steps = num_sampling_steps
        self.is_flow_decoder = True  # Flag for model to detect

        ch = base_channels
        emb_dim = 256

        # Time embedding: scalar -> emb_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Condition embedding: latent_dim -> emb_dim
        self.cond_embed = nn.Sequential(
            nn.Linear(latent_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # U-Net encoder
        self.enc_conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.enc_res1 = FlowResBlock(ch, emb_dim)
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # 84->42

        self.enc_res2 = FlowResBlock(ch, emb_dim)
        self.down2 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)  # 42->21

        self.enc_res3 = FlowResBlock(ch * 2, emb_dim)
        self.down3 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)  # 21->11

        # Bottleneck
        self.mid_res1 = FlowResBlock(ch * 4, emb_dim)
        self.mid_res2 = FlowResBlock(ch * 4, emb_dim)

        # U-Net decoder (with skip connections)
        self.up3 = nn.ConvTranspose2d(
            ch * 4, ch * 2, 4, stride=2, padding=1, output_padding=1
        )  # 11->22->crop to 21
        self.dec_res3 = FlowResBlock(ch * 2, emb_dim)  # after reduce: ch*2
        self.dec_reduce3 = nn.Conv2d(ch * 4, ch * 2, 1)  # reduce after concat

        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)  # 21->42
        self.dec_res2 = FlowResBlock(ch, emb_dim)  # after reduce: ch
        self.dec_reduce2 = nn.Conv2d(ch * 2, ch, 1)

        self.up1 = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)  # 42->84
        self.dec_res1 = FlowResBlock(ch, emb_dim)  # after reduce: ch
        self.dec_reduce1 = nn.Conv2d(ch * 2, ch, 1)

        # Output: velocity (no sigmoid — velocity can be any value)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

        # Action head (from condition, not from velocity)
        self.action_head = nn.Linear(latent_dim, num_actions)

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field v(x_t, t, condition).

        Args:
            x_t: (B, 3, H, W) noisy image at timestep t
            t: (B,) timestep in [0, 1]
            condition: (B, latent_dim) gradient latent

        Returns:
            v: (B, 3, H, W) predicted velocity
        """
        # Compute combined embedding
        t_emb = self.time_embed(t)  # (B, emb_dim)
        c_emb = self.cond_embed(condition)  # (B, emb_dim)
        emb = t_emb + c_emb  # (B, emb_dim)

        # U-Net encoder
        h = self.enc_conv_in(x_t)  # (B, ch, 84, 84)
        h1 = self.enc_res1(h, emb)  # (B, ch, 84, 84)
        h = self.down1(h1)  # (B, ch, 42, 42)

        h2 = self.enc_res2(h, emb)  # (B, ch, 42, 42)
        h = self.down2(h2)  # (B, ch*2, 21, 21)

        h3 = self.enc_res3(h, emb)  # (B, ch*2, 21, 21)
        h = self.down3(h3)  # (B, ch*4, 11, 11)

        # Bottleneck
        h = self.mid_res1(h, emb)
        h = self.mid_res2(h, emb)

        # U-Net decoder with skip connections
        h = self.up3(h)  # (B, ch*2, 22, 22) or (B, ch*2, 21, 21)
        # Crop to match skip connection size
        h = h[:, :, : h3.shape[2], : h3.shape[3]]
        h = torch.cat([h, h3], dim=1)  # (B, ch*4, 21, 21)
        h = self.dec_reduce3(h)  # (B, ch*2, 21, 21)
        h = self.dec_res3(h, emb)

        h = self.up2(h)  # (B, ch, 42, 42)
        h = h[:, :, : h2.shape[2], : h2.shape[3]]
        h = torch.cat([h, h2], dim=1)  # (B, ch*2, 42, 42)
        h = self.dec_reduce2(h)
        h = self.dec_res2(h, emb)

        h = self.up1(h)  # (B, ch, 84, 84)
        h = h[:, :, : h1.shape[2], : h1.shape[3]]
        h = torch.cat([h, h1], dim=1)  # (B, ch*2, 84, 84)
        h = self.dec_reduce1(h)
        h = self.dec_res1(h, emb)

        return self.conv_out(h)  # (B, 3, 84, 84)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Generate images via ODE integration (Euler method).

        Args:
            condition: (B, latent_dim)
            num_steps: number of ODE steps (default: self.num_sampling_steps)

        Returns:
            images: (B, 3, H, W) in [0, 1]
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps

        B = condition.shape[0]
        device = condition.device

        # Start from noise
        x = torch.randn(B, 3, self.image_size, self.image_size, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.predict_velocity(x, t, condition)
            x = x + dt * v

        return x.clamp(0, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard decoder interface for compatibility.

        During inference (eval mode), generates images via ODE sampling.
        During training, this is NOT called — the training loop calls
        predict_velocity() directly.

        Args:
            x: (B, T, latent_dim) or (B, latent_dim)

        Returns:
            images: (B, T, 3, H, W)
            actions: (B, T, num_actions)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            # Process each timestep
            images = []
            for t in range(T):
                img = self.sample(x[:, t])
                images.append(img)
            images = torch.stack(images, dim=1)  # (B, T, 3, H, W)
            actions = self.action_head(x)  # (B, T, num_actions)
        else:
            images = self.sample(x).unsqueeze(1)
            actions = self.action_head(x).unsqueeze(1)

        return images, actions


def get_decoder(
    decoder_type: str,
    latent_dim: int = 512,
    image_size: int = 84,
    num_actions: int = 5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get decoder by type.

    Args:
        decoder_type: One of 'basic', 'residual', 'unet', 'style', 'vqgan', 'flow'
        latent_dim: Input latent dimension
        image_size: Output image size
        num_actions: Number of actions for action head
        **kwargs: Additional decoder-specific arguments

    Returns:
        Decoder module
    """
    decoders = {
        "basic": ImageDecoder,
        "residual": ResidualImageDecoder,
        "unet": UNetImageDecoder,
        "style": StyleImageDecoder,
        "vqgan": VQTokenDecoder,
        "flow": FlowImageDecoder,
    }

    # Lazy import for pretrained decoder (requires diffusers)
    if decoder_type == "pretrained":
        from attacker.models.pretrained_decoder import PretrainedVAEDecoder

        decoders["pretrained"] = PretrainedVAEDecoder

    if decoder_type not in decoders:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. Available: {list(decoders.keys()) + ['pretrained']}"
        )

    return decoders[decoder_type](
        latent_dim=latent_dim,
        image_size=image_size,
        num_actions=num_actions,
        **kwargs,
    )

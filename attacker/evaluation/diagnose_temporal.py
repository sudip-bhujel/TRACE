"""
Temporal Diagnosis Script for TRACE.

Produces four diagnostic outputs:

1. **Gradient cosine similarity** between consecutive frames
2. **Attention weight heatmaps** (Transformer only)
3. **Latent similarity** before vs. after the temporal model
4. **Per-timestep PSNR** comparing frame 0 (no context) vs frame 7 (full context)

Usage:
    uv run -m attacker.evaluation.diagnose_temporal <config.yaml>

The config should include:
    data.h5_path          — test H5 dataset
    model.*               — architecture params (used to load checkpoint)
    eval.checkpoint       — path to trained checkpoint
    output.save_dir       — where to save diagnostic figures
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacker.data.dataset import TemporalGradientDataset
from attacker.models.model import TemporalGradientInversion


# ============================================================================
# 1. Gradient cosine similarity
# ============================================================================


def gradient_cosine_similarity(
    dataset: TemporalGradientDataset,
    num_sequences: int = 100,
    save_dir: Path = Path("diagnostics"),
) -> Dict[str, float]:
    """Compute pairwise cosine similarity between consecutive gradients."""
    similarities: List[float] = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (gradients, _images, _actions) in enumerate(dataloader):
        if i >= num_sequences:
            break
        # gradients: (1, T, D)
        grads = gradients.squeeze(0)  # (T, D)
        T = grads.shape[0]
        for t in range(T - 1):
            g1 = grads[t]
            g2 = grads[t + 1]
            cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            similarities.append(cos_sim)

    sims = np.array(similarities)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=50, edgecolor="black", alpha=0.7, color="#3498db")
    ax.axvline(sims.mean(), color="red", linestyle="--", label=f"Mean: {sims.mean():.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Gradient Cosine Similarity Between Consecutive Frames")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "gradient_cosine_similarity.png", dpi=150)
    plt.close()

    print(f"  Gradient cosine similarity: mean={sims.mean():.4f}, "
          f"std={sims.std():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")

    return {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
    }


# ============================================================================
# 2. Attention weight heatmaps
# ============================================================================


def _extract_attention_weights(
    model: TemporalGradientInversion,
    gradients: torch.Tensor,
) -> List[torch.Tensor]:
    """Extract attention weights from each Transformer layer.

    Returns list of (B, heads, T, T) tensors, one per layer.
    """
    if model.skip_transformer:
        return []

    temporal = model.temporal_model
    if not hasattr(temporal, "layers"):
        return []

    # Encode
    latents = model.gradient_encoder(gradients)

    # Add positional embeddings if present
    if hasattr(temporal, "pos_embedding") and temporal.pos_embedding is not None:
        T = latents.shape[1]
        latents = latents + temporal.pos_embedding[:, :T, :]

    attn_weights = []
    x = latents
    for layer in temporal.layers:
        if not hasattr(layer, "attention"):
            break
        attn = layer.attention
        B, T, _ = x.shape
        # Compute Q, K manually to get attention weights
        qkv = attn.qkv_proj(layer.norm1(x))
        qkv = qkv.reshape(B, T, 3, attn.num_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if present
        if hasattr(attn, "use_rope") and attn.use_rope and hasattr(attn, "rope"):
            from attacker.models.transformer import apply_rotary_emb
            cos, sin = attn.rope(T)
            cos = cos.to(q.dtype).to(q.device)
            sin = sin.to(q.dtype).to(q.device)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Manual attention scores
        scale = attn.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask
        if attn.is_causal:
            mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, heads, T, T)
        attn_weights.append(weights.detach().cpu())

        # Forward through the full layer for next iteration
        x = layer(x)

    return attn_weights


def visualize_attention(
    model: TemporalGradientInversion,
    dataset: TemporalGradientDataset,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 3,
):
    """Extract and visualize attention weights from Transformer layers."""
    if model.skip_transformer:
        print("  Skipping attention visualization (no Transformer)")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for seq_idx, (gradients, _images, _actions) in enumerate(dataloader):
        if seq_idx >= num_sequences:
            break

        gradients = gradients.to(device)

        with torch.no_grad():
            attn_weights = _extract_attention_weights(model, gradients)

        if not attn_weights:
            print("  Could not extract attention weights")
            return

        num_layers = len(attn_weights)
        fig, axes = plt.subplots(
            1, num_layers, figsize=(5 * num_layers, 4), squeeze=False
        )

        for layer_idx, weights in enumerate(attn_weights):
            # Average over heads: (T, T)
            avg_weights = weights[0].mean(dim=0).numpy()
            im = axes[0, layer_idx].imshow(avg_weights, cmap="viridis", vmin=0, vmax=1)
            axes[0, layer_idx].set_title(f"Layer {layer_idx}")
            axes[0, layer_idx].set_xlabel("Key (t)")
            axes[0, layer_idx].set_ylabel("Query (t)")
            plt.colorbar(im, ax=axes[0, layer_idx], fraction=0.046)

        plt.suptitle(f"Attention Weights (Seq {seq_idx + 1}, avg over heads)")
        plt.tight_layout()
        plt.savefig(save_dir / f"attention_seq_{seq_idx + 1:03d}.png", dpi=150)
        plt.close()

    print(f"  Saved attention visualizations for {num_sequences} sequences")


# ============================================================================
# 3. Latent similarity analysis
# ============================================================================


def latent_similarity_analysis(
    model: TemporalGradientInversion,
    dataset: TemporalGradientDataset,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 50,
) -> Dict[str, float]:
    """Compare latents before and after the temporal model."""
    if model.skip_transformer:
        print("  Skipping latent similarity (no temporal model)")
        return {}

    cos_sims: List[float] = []
    l2_dists: List[float] = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (gradients, _images, _actions) in enumerate(dataloader):
            if i >= num_sequences:
                break

            gradients = gradients.to(device)
            latents_before = model.gradient_encoder(gradients)  # (1, T, D)
            latents_after = model.temporal_model(latents_before)  # (1, T, D)

            # Per-frame cosine similarity
            cos = F.cosine_similarity(
                latents_before.squeeze(0), latents_after.squeeze(0), dim=-1
            )
            cos_sims.extend(cos.cpu().tolist())

            # L2 distance
            l2 = (latents_after - latents_before).pow(2).sum(dim=-1).sqrt()
            l2_dists.extend(l2.squeeze(0).cpu().tolist())

    cos_arr = np.array(cos_sims)
    l2_arr = np.array(l2_dists)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(cos_arr, bins=50, edgecolor="black", alpha=0.7, color="#2ecc71")
    ax1.axvline(cos_arr.mean(), color="red", linestyle="--",
                label=f"Mean: {cos_arr.mean():.4f}")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_title("Latent Similarity: Before vs After Temporal Model")
    ax1.legend()

    ax2.hist(l2_arr, bins=50, edgecolor="black", alpha=0.7, color="#e74c3c")
    ax2.axvline(l2_arr.mean(), color="blue", linestyle="--",
                label=f"Mean: {l2_arr.mean():.4f}")
    ax2.set_xlabel("L2 Distance")
    ax2.set_title("Latent L2 Distance: Before vs After Temporal Model")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "latent_similarity.png", dpi=150)
    plt.close()

    print(f"  Latent cos-sim: mean={cos_arr.mean():.4f}, std={cos_arr.std():.4f}")
    print(f"  Latent L2 dist: mean={l2_arr.mean():.4f}, std={l2_arr.std():.4f}")

    return {
        "cosine_sim_mean": float(cos_arr.mean()),
        "cosine_sim_std": float(cos_arr.std()),
        "l2_dist_mean": float(l2_arr.mean()),
        "l2_dist_std": float(l2_arr.std()),
    }


# ============================================================================
# 4. Per-timestep PSNR
# ============================================================================


def per_timestep_psnr(
    model: TemporalGradientInversion,
    dataset: TemporalGradientDataset,
    device: torch.device,
    save_dir: Path,
    num_sequences: int = 50,
) -> Dict[int, float]:
    """Compare PSNR at each timestep position."""
    psnr_per_t: Dict[int, List[float]] = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (gradients, images, _actions) in enumerate(dataloader):
            if i >= num_sequences:
                break

            gradients = gradients.to(device)
            images = images.to(device)

            pred_images, _, _, _ = model(gradients)
            pred_images = pred_images.clamp(0, 1)

            T = images.shape[1]
            for t in range(T):
                mse = F.mse_loss(pred_images[0, t], images[0, t]).item()
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
                psnr_per_t.setdefault(t, []).append(psnr)

    # Average over sequences
    mean_psnr = {t: float(np.mean(vals)) for t, vals in sorted(psnr_per_t.items())}
    std_psnr = {t: float(np.std(vals)) for t, vals in sorted(psnr_per_t.items())}

    timesteps = sorted(mean_psnr.keys())
    means = [mean_psnr[t] for t in timesteps]
    stds = [std_psnr[t] for t in timesteps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(timesteps, means, yerr=stds, capsize=3, color="#9b59b6", alpha=0.8,
           edgecolor="black")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-Timestep PSNR\n(Frame 0: no context → Frame T-1: full context)")
    ax.set_xticks(timesteps)
    plt.tight_layout()
    plt.savefig(save_dir / "per_timestep_psnr.png", dpi=150)
    plt.close()

    print("  Per-timestep PSNR:")
    for t in timesteps:
        print(f"    t={t}: {mean_psnr[t]:.2f} ± {std_psnr[t]:.2f} dB")

    return mean_psnr


# ============================================================================
# Main
# ============================================================================


def diagnose(
    h5_path: str,
    checkpoint: str,
    save_dir: str = "diagnostics",
    num_sequences: int = 50,
    sequence_length: int = 8,
    stride: int = 8,
    gradient_dim: Optional[int] = None,
    gradient_layers: Optional[List[str]] = None,
    device: str = "auto",
    # Model params
    latent_dim: int = 512,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    encoder_hidden_dim: Optional[int] = None,
    encoder_num_blocks: Optional[int] = None,
    encoder_expansion: Optional[int] = None,
    encoder_projection_rank: Optional[int] = None,
    encoder_type: str = "basic",
    decoder_type: str = "basic",
    skip_transformer: bool = False,
    temporal_model_type: str = "transformer",
    ff_multiplier: int = 4,
    use_rope: bool = False,
):
    """Run all temporal diagnostics."""
    # Device
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Save dir: {save_path}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
    )
    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = len(dataset.actions.unique())

    # --- Diagnostic 1: Gradient correlation (no model needed) ---
    print("\n[1/4] Gradient Cosine Similarity")
    grad_sim = gradient_cosine_similarity(dataset, num_sequences, save_path)

    # Load model
    print("\nLoading model...")
    model = TemporalGradientInversion(
        gradient_dim=actual_gradient_dim,
        latent_dim=latent_dim,
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_blocks=encoder_num_blocks,
        encoder_expansion=encoder_expansion,
        encoder_projection_rank=encoder_projection_rank,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        skip_transformer=skip_transformer,
        temporal_model_type=temporal_model_type,
        ff_multiplier=ff_multiplier,
        use_rope=use_rope,
    ).to(dev)

    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # --- Diagnostic 2: Attention weights ---
    print("\n[2/4] Attention Weight Visualization")
    visualize_attention(model, dataset, dev, save_path, min(num_sequences, 5))

    # --- Diagnostic 3: Latent similarity ---
    print("\n[3/4] Latent Similarity Analysis")
    latent_sim = latent_similarity_analysis(
        model, dataset, dev, save_path, num_sequences
    )

    # --- Diagnostic 4: Per-timestep PSNR ---
    print("\n[4/4] Per-Timestep PSNR")
    psnr = per_timestep_psnr(model, dataset, dev, save_path, num_sequences)

    print(f"\n{'=' * 60}")
    print("Diagnostics complete!")
    print(f"All figures saved to: {save_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python -m attacker.evaluation.diagnose_temporal <config.yaml>"

    cfg = OmegaConf.load(sys.argv[1])
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    gradient_dim = data_cfg.get("gradient_dim", None)
    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    diagnose(
        h5_path=eval_cfg.get("h5_path", data_cfg.get("h5_path")),
        checkpoint=eval_cfg.get("checkpoint",
                                str(Path(output_cfg.get("save_dir", "ckpts")) / "best_model.pt")),
        save_dir=str(Path(output_cfg.get("save_dir", "diagnostics")) / "diagnostics"),
        num_sequences=eval_cfg.get("num_sequences", 50),
        sequence_length=model_cfg.get("sequence_length", 8),
        stride=model_cfg.get("stride", 8),
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
        device=cfg.get("device", "auto"),
        latent_dim=model_cfg.get("latent_dim", 512),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", None),
        encoder_num_blocks=model_cfg.get("encoder_num_blocks", None),
        encoder_expansion=model_cfg.get("encoder_expansion", None),
        encoder_projection_rank=model_cfg.get("encoder_projection_rank", None),
        encoder_type=model_cfg.get("encoder_type", "basic"),
        decoder_type=model_cfg.get("decoder_type", "basic"),
        skip_transformer=model_cfg.get("skip_transformer", False),
        temporal_model_type=model_cfg.get("temporal_model_type", "transformer"),
        ff_multiplier=model_cfg.get("ff_multiplier", 4),
        use_rope=model_cfg.get("use_rope", False),
    )

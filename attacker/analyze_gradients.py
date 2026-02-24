"""
Gradient Layer Informativeness Analysis.

Analyzes which gradient layers from the victim ActorCritic model carry the
most useful information for image reconstruction. This helps identify the
optimal gradient subset to reduce input dimensionality and noise.

The gradient vector is a concatenation of all model parameters' gradients,
sorted alphabetically by parameter name (see victim/capture.py:flatten_gradients).

Usage:
    uv run -m attacker.analyze_gradients \
        --h5_path trajectory_data/gradients_108_augmented.h5 \
        --save_dir ckpts/gradient_analysis
"""

import argparse
import os
from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np

from victim.model import ActorCritic


def get_layer_map(num_actions: int = 5, hidden_size: int = 512):
    """
    Build a map from model parameter names to their gradient dimension ranges.

    Returns:
        OrderedDict of {param_name: (start_idx, end_idx, shape, num_params)}
        sorted alphabetically (matching flatten_gradients order).
    """
    model = ActorCritic(in_channels=3, num_actions=num_actions, hidden_size=hidden_size)

    layer_map = OrderedDict()
    offset = 0

    for name in sorted(model.state_dict().keys()):
        param = model.state_dict()[name]
        num_params = param.numel()
        layer_map[name] = {
            "start": offset,
            "end": offset + num_params,
            "shape": tuple(param.shape),
            "num_params": num_params,
        }
        offset += num_params

    return layer_map, offset


def categorize_layer(name: str) -> str:
    """Categorize a layer by its role in the model."""
    if "encoder.0" in name:
        return "Conv1 (3→32, k=8, s=4)"
    elif "encoder.1" in name:
        return "Conv2 (32→64, k=4, s=2)"
    elif "encoder.2" in name:
        return "Conv3 (64→64, k=3, s=1)"
    elif "fc" in name:
        return "FC (6400→512)"
    elif "policy" in name:
        return "Policy Head (512→5)"
    elif "value" in name:
        return "Value Head (512→1)"
    return "Unknown"


def analyze_gradients(
    h5_path: str,
    save_dir: str = "ckpts/gradient_analysis",
    num_samples: int = 2000,
    gradient_dim: int = None,
):
    """Analyze gradient informativeness per model layer."""
    os.makedirs(save_dir, exist_ok=True)

    # Build layer map
    layer_map, total_params = get_layer_map()

    print("=" * 70)
    print("ActorCritic Gradient Layer Map")
    print("=" * 70)
    print(f"{'Parameter Name':<30} {'Shape':<20} {'Params':>10} {'Range':>20}")
    print("-" * 70)
    for name, info in layer_map.items():
        print(
            f"{name:<30} {str(info['shape']):<20} {info['num_params']:>10,} "
            f"{info['start']:>8,}-{info['end']:>8,}"
        )
    print("-" * 70)
    print(f"{'Total':<30} {'':<20} {total_params:>10,}")

    # Load gradient data
    print(f"\nLoading gradients from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        grad_shape = f["gradients"].shape
        img_shape = f["images"].shape
        actual_grad_dim = grad_shape[1]

        print(f"  Gradients: {grad_shape} (dtype={f['gradients'].dtype})")
        print(f"  Images: {img_shape}")

        if gradient_dim:
            actual_grad_dim = min(gradient_dim, actual_grad_dim)

        # Sample subset for analysis
        n = min(num_samples, grad_shape[0])
        indices = np.random.RandomState(42).choice(grad_shape[0], n, replace=False)
        indices.sort()

        print(f"  Analyzing {n} samples (gradient_dim={actual_grad_dim:,})...")

        gradients = f["gradients"][indices, :actual_grad_dim].astype(np.float32)
        images = f["images"][indices].astype(np.float32) / 255.0

    # === Analysis 1: Per-Layer Gradient Statistics ===
    print("\n" + "=" * 70)
    print("Per-Layer Gradient Statistics")
    print("=" * 70)

    layer_stats = []
    for name, info in layer_map.items():
        start, end = info["start"], info["end"]
        if start >= actual_grad_dim:
            # This layer is beyond our gradient_dim cutoff
            layer_stats.append(
                {
                    "name": name,
                    "category": categorize_layer(name),
                    "num_params": info["num_params"],
                    "captured": 0,
                    "mean_abs": 0,
                    "std": 0,
                    "variance": 0,
                    "max_abs": 0,
                    "sparsity": 1.0,
                    "snr": 0,
                }
            )
            continue

        # Clamp end to actual gradient dim
        end_clamped = min(end, actual_grad_dim)
        captured = end_clamped - start
        layer_grads = gradients[:, start:end_clamped]

        mean_abs = np.mean(np.abs(layer_grads))
        std = np.std(layer_grads)
        var = np.var(layer_grads)
        max_abs = np.max(np.abs(layer_grads))
        sparsity = np.mean(np.abs(layer_grads) < 1e-6)

        # Signal-to-noise ratio: variance across samples / mean variance within samples
        per_dim_var_across_samples = np.var(layer_grads, axis=0).mean()
        per_sample_var = np.var(layer_grads, axis=1).mean()
        snr = per_dim_var_across_samples / (per_sample_var + 1e-10)

        layer_stats.append(
            {
                "name": name,
                "category": categorize_layer(name),
                "num_params": info["num_params"],
                "captured": captured,
                "mean_abs": float(mean_abs),
                "std": float(std),
                "variance": float(var),
                "max_abs": float(max_abs),
                "sparsity": float(sparsity),
                "snr": float(snr),
            }
        )

    print(
        f"{'Layer':<30} {'Captured':>8} {'MeanAbs':>10} {'Std':>10} {'SNR':>8} {'Sparsity':>10}"
    )
    print("-" * 80)
    for s in layer_stats:
        print(
            f"{s['name']:<30} {s['captured']:>8,} {s['mean_abs']:>10.6f} "
            f"{s['std']:>10.6f} {s['snr']:>8.4f} {s['sparsity']:>9.1%}"
        )

    # === Analysis 2: Per-Layer Correlation with Image Features ===
    print("\n" + "=" * 70)
    print("Per-Layer Correlation with Image Variation")
    print("=" * 70)

    # Compute image feature: flatten and compute variation per sample
    img_flat = images.reshape(n, -1)
    img_mean = img_flat.mean(axis=1)
    img_std_per_sample = img_flat.std(axis=1)

    for s in layer_stats:
        if s["captured"] == 0:
            s["corr_with_img_mean"] = 0
            s["corr_with_img_std"] = 0
            continue

        start = layer_map[s["name"]]["start"]
        end_clamped = min(layer_map[s["name"]]["end"], actual_grad_dim)
        layer_grads = gradients[:, start:end_clamped]

        # Gradient magnitude per sample
        grad_magnitude = np.linalg.norm(layer_grads, axis=1)

        # Correlation with image brightness
        corr_mean = np.corrcoef(grad_magnitude, img_mean)[0, 1]
        corr_std = np.corrcoef(grad_magnitude, img_std_per_sample)[0, 1]

        s["corr_with_img_mean"] = float(corr_mean) if not np.isnan(corr_mean) else 0
        s["corr_with_img_std"] = float(corr_std) if not np.isnan(corr_std) else 0

    print(f"{'Layer':<30} {'|Corr(brightness)|':>18} {'|Corr(contrast)|':>18}")
    print("-" * 70)
    for s in layer_stats:
        print(
            f"{s['name']:<30} {abs(s['corr_with_img_mean']):>18.4f} "
            f"{abs(s['corr_with_img_std']):>18.4f}"
        )

    # === Analysis 3: Category-Level Summary ===
    print("\n" + "=" * 70)
    print("Category-Level Summary")
    print("=" * 70)

    categories = OrderedDict()
    for s in layer_stats:
        cat = s["category"]
        if cat not in categories:
            categories[cat] = {
                "params": 0,
                "captured": 0,
                "total_variance": 0,
                "max_snr": 0,
                "max_corr": 0,
                "layers": [],
            }
        categories[cat]["params"] += s["num_params"]
        categories[cat]["captured"] += s["captured"]
        categories[cat]["total_variance"] += s["variance"] * s["captured"]
        categories[cat]["max_snr"] = max(categories[cat]["max_snr"], s["snr"])
        categories[cat]["max_corr"] = max(
            categories[cat]["max_corr"],
            abs(s["corr_with_img_mean"]) + abs(s["corr_with_img_std"]),
        )
        categories[cat]["layers"].append(s["name"])

    print(
        f"{'Category':<28} {'Params':>10} {'Captured':>10} {'AvgVar':>12} {'MaxSNR':>8} {'ImgCorr':>8}"
    )
    print("-" * 80)
    for cat, info in categories.items():
        avg_var = info["total_variance"] / max(info["captured"], 1)
        print(
            f"{cat:<28} {info['params']:>10,} {info['captured']:>10,} "
            f"{avg_var:>12.8f} {info['max_snr']:>8.4f} {info['max_corr']:>8.4f}"
        )

    # === Analysis 4: Gradient Dimension Importance via Variance ===
    print("\n" + "=" * 70)
    print("Top Gradient Dimensions by Variance")
    print("=" * 70)

    per_dim_variance = np.var(gradients, axis=0)

    # Map each dimension to its layer
    dim_to_layer = {}
    for name, info in layer_map.items():
        for d in range(info["start"], min(info["end"], actual_grad_dim)):
            dim_to_layer[d] = name

    # Find top-k dimensions
    top_k = 100
    top_dims = np.argsort(per_dim_variance)[-top_k:][::-1]

    layer_representation = {}
    for d in top_dims:
        layer = dim_to_layer.get(d, "unknown")
        cat = categorize_layer(layer)
        layer_representation[cat] = layer_representation.get(cat, 0) + 1

    print(f"Top {top_k} highest-variance dimensions belong to:")
    for cat, count in sorted(layer_representation.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}/{top_k} ({count / top_k:.0%})")

    # === Visualization ===
    # 1. Per-dimension variance heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Variance across all dimensions
    axes[0].plot(per_dim_variance, linewidth=0.3, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Gradient Dimension Index")
    axes[0].set_ylabel("Variance")
    axes[0].set_title("Per-Dimension Gradient Variance")
    axes[0].set_yscale("log")

    # Add layer boundaries
    colors = ["red", "orange", "green", "blue", "purple", "brown"]
    for i, (name, info) in enumerate(layer_map.items()):
        if info["start"] < actual_grad_dim:
            end_vis = min(info["end"], actual_grad_dim)
            axes[0].axvspan(
                info["start"],
                end_vis,
                alpha=0.1,
                color=colors[i % len(colors)],
                label=categorize_layer(name)
                if ".weight" in name and "2" not in name
                else None,
            )
    axes[0].legend(loc="upper right", fontsize=7)

    # 2. Category bar chart
    cat_names = list(categories.keys())
    cat_vars = [
        categories[c]["total_variance"] / max(categories[c]["captured"], 1)
        for c in cat_names
    ]
    cat_captured = [categories[c]["captured"] for c in cat_names]

    bar_colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6", "#95a5a6"]
    axes[1].bar(range(len(cat_names)), cat_vars, color=bar_colors[: len(cat_names)])
    axes[1].set_xticks(range(len(cat_names)))
    axes[1].set_xticklabels(cat_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Average Variance per Dimension")
    axes[1].set_title("Gradient Variance by Layer Category")
    axes[1].set_yscale("log")

    # Add param count labels
    for i, (v, c) in enumerate(zip(cat_vars, cat_captured)):
        axes[1].text(i, v * 1.2, f"{c:,}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "gradient_analysis.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"\nSaved gradient analysis plot to {save_dir}/gradient_analysis.png")

    # === Save Recommendations ===
    # Compute cumulative variance coverage
    sorted_var_idx = np.argsort(per_dim_variance)[::-1]
    cumulative_var = np.cumsum(per_dim_variance[sorted_var_idx])
    total_var = cumulative_var[-1]

    thresholds = [0.90, 0.95, 0.99]
    print("\n" + "=" * 70)
    print("Dimensionality Reduction Recommendations")
    print("=" * 70)
    for t in thresholds:
        n_dims = np.searchsorted(cumulative_var, t * total_var) + 1
        print(
            f"  {t:.0%} variance coverage: {n_dims:,} / {actual_grad_dim:,} dims ({n_dims / actual_grad_dim:.1%})"
        )

    # Encoder-only recommendation
    encoder_end = 0
    for name, info in layer_map.items():
        if "encoder" in name:
            encoder_end = max(encoder_end, info["end"])

    encoder_var = per_dim_variance[: min(encoder_end, actual_grad_dim)].sum()
    print(f"\n  Encoder-only (first {encoder_end:,} dims):")
    print(f"    Variance coverage: {encoder_var / total_var:.1%}")
    print(
        f"    Dimension reduction: {actual_grad_dim:,} → {encoder_end:,} ({encoder_end / actual_grad_dim:.1%})"
    )

    # Save results
    with open(os.path.join(save_dir, "analysis_results.txt"), "w") as f:
        f.write(f"h5_path: {h5_path}\n")
        f.write(f"total_model_params: {total_params}\n")
        f.write(f"actual_gradient_dim: {actual_grad_dim}\n")
        f.write(f"num_samples_analyzed: {n}\n\n")

        f.write("Layer Stats:\n")
        for s in layer_stats:
            f.write(
                f"  {s['name']}: params={s['num_params']}, captured={s['captured']}, "
                f"var={s['variance']:.8f}, snr={s['snr']:.4f}, "
                f"corr_brightness={s['corr_with_img_mean']:.4f}, "
                f"corr_contrast={s['corr_with_img_std']:.4f}\n"
            )

        f.write(f"\nEncoder-only dims: {encoder_end}\n")
        f.write(f"Encoder variance coverage: {encoder_var / total_var:.4f}\n")

        for t in thresholds:
            n_dims = np.searchsorted(cumulative_var, t * total_var) + 1
            f.write(f"{t:.0%} variance: {n_dims} dims\n")

    print(f"Saved analysis results to {save_dir}/analysis_results.txt")

    return layer_stats, categories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze gradient layer informativeness"
    )
    parser.add_argument(
        "--h5_path", type=str, default="trajectory_data/gradients_108_augmented.h5"
    )
    parser.add_argument("--save_dir", type=str, default="ckpts/gradient_analysis")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to analyze (subset for speed)",
    )
    parser.add_argument(
        "--gradient_dim",
        type=int,
        default=None,
        help="Limit gradient dimensions (None = use all available)",
    )
    args = parser.parse_args()

    analyze_gradients(
        h5_path=args.h5_path,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        gradient_dim=args.gradient_dim,
    )

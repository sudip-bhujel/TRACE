"""
Defense Evaluation Script.

Evaluates the impact of gradient defense mechanisms on reconstruction
quality. Loads a trained attacker model, applies each configured defense
to the test gradients, and reports per-defense metrics.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.evaluate import load_model
from attacker.evaluation.metrics import (
    METRIC_FORMATS,
    METRIC_KEYS,
    METRIC_LABELS,
    MetricsComputer,
    print_results,
)
from defense.base import GradientDefense, get_defense
from defense.strategies.dp_sgd import DPSGDDefense


def evaluate_with_defense(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    defense: Optional[GradientDefense],
    num_sequences: int = 20,
    enable_fid: bool = False,
    num_actions: int = 5,
    collect_first_sequence: bool = False,
    model_type: str = "temporal",
) -> Dict[str, object]:
    """
    Run evaluation with a specific defense applied to gradients.

    Args:
        model: Trained attacker model.
        dataloader: Test data loader yielding (gradients, images, actions).
        device: Torch device.
        defense: Defense to apply, or ``None`` for the undefended baseline.
        num_sequences: Number of sequences to evaluate.
        enable_fid: Whether to compute FID (slow).
        num_actions: Number of discrete actions.
        collect_first_sequence: If ``True``, include the first sequence's
            ground-truth images and predicted images in the returned dict
            (keys ``"gt_images"`` and ``"pred_images"``).

    Returns:
        Dict of metric name to value (and optionally image tensors).
    """
    model.eval()
    metrics = MetricsComputer(device, compute_fid_flag=enable_fid)

    defense_name = defense.name if defense is not None else "none"

    data_iter = iter(dataloader)
    evaluated = 0
    first_gt = None
    first_pred = None

    for seq_idx in range(num_sequences):
        try:
            gradients, images, actions = next(data_iter)
        except StopIteration:
            break

        gradients = gradients[:1].to(device)
        images = images[:1]
        actions = actions[:1]

        if defense is not None:
            gradients = defense.apply(gradients)

        with torch.no_grad():
            if model_type == "autoregressive":
                pred_images, pred_actions, _, _ = model(
                    gradients, teacher_forcing=False
                )
            else:
                pred_images, pred_actions, _, _ = model(gradients)

        metrics.update(pred_images, images, pred_actions, actions)
        evaluated += 1

        if collect_first_sequence and seq_idx == 0:
            first_gt = images[0].cpu()
            first_pred = pred_images[0].cpu()

    results = metrics.compute()
    results["defense"] = defense_name
    results["num_sequences"] = evaluated

    if collect_first_sequence:
        results["gt_images"] = first_gt
        results["pred_images"] = first_pred

    return results


def build_defense_configs(cfg) -> List[Optional[GradientDefense]]:
    """
    Build a list of defense instances from the YAML config.

    The config ``defenses`` section is a list of dicts, each with a
    ``type`` key and defense-specific parameters.  An implicit baseline
    (``None``) is always prepended.

    Args:
        cfg: OmegaConf config with a ``defenses`` key.

    Returns:
        List starting with ``None`` (baseline) followed by configured
        defenses.
    """
    defenses: List[Optional[GradientDefense]] = [None]

    for entry in cfg.get("defenses", []):
        entry = dict(entry)
        defense_type = entry.pop("type")
        defenses.append(get_defense(defense_type, **entry))

    return defenses


def _save_reconstruction_grid(
    all_results: List[Dict[str, object]],
    save_dir: Path,
):
    """
    Save a reconstruction grid comparing defenses on the same sequence.

    Layout::

        Row 0:  Ground Truth   (T frames)
        Row 1:  No defense     (T reconstructed frames)
        Row 2:  Defense A      ...
        Row 3:  Defense B      ...
        ...
    """
    rows_with_images = [r for r in all_results if r.get("pred_images") is not None]
    if not rows_with_images:
        return

    grid_rows = _pick_representative_rows(rows_with_images)

    gt_images = grid_rows[0]["gt_images"]
    T = gt_images.shape[0]
    num_rows = 1 + len(grid_rows)

    row_labels = ["Ground\nTruth"] + [
        _short_defense_label(r.get("defense", "?")) for r in grid_rows
    ]

    fig, axes = plt.subplots(
        num_rows,
        T,
        figsize=(1.8 * T, 1.8 * num_rows),
    )
    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    if T == 1:
        axes = axes[:, None]

    for t in range(T):
        ax = axes[0, t]
        ax.imshow(gt_images[t].permute(1, 2, 0).numpy().clip(0, 1))
        ax.axis("off")

    for row_idx, r in enumerate(grid_rows, start=1):
        pred = r["pred_images"]
        for t in range(T):
            ax = axes[row_idx, t]
            ax.imshow(pred[t].permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis("off")

    for row_idx, label in enumerate(row_labels):
        ax0 = axes[row_idx, 0]
        pos = ax0.get_position()
        y = (pos.y0 + pos.y1) / 2
        fig.text(
            pos.x0 - 0.018,
            y,
            label,
            va="center",
            ha="center",
            fontsize=14,
            fontfamily="serif",
            rotation=90,
        )

    for t in range(T):
        col_pos = axes[0, t].get_position()
        x = (col_pos.x0 + col_pos.x1) / 2
        fig.text(
            x,
            col_pos.y1 + 0.005,
            f"t={t + 1}",
            ha="center",
            fontfamily="serif",
            fontsize=14,
        )

    for ext in ("png", "pdf"):
        path = save_dir / f"defense_reconstruction_grid.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Reconstruction grid saved to: {path}")
    plt.close(fig)


def _pick_representative_rows(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Select one representative result per defense category.

    Picks the middle configuration when multiple are available
    (e.g. for 3 pruning levels, picks the 2nd).
    """
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        name = r.get("defense", "")
        if name == "none":
            cat = "baseline"
        elif "pruning" in name:
            cat = "pruning"
        elif "noise" in name:
            cat = "noise"
        elif "dpsgd" in name:
            cat = "dpsgd"
        else:
            cat = name
        buckets.setdefault(cat, []).append(r)

    selected: List[Dict[str, object]] = []
    for cat in ("baseline", "pruning", "noise", "dpsgd"):
        entries = buckets.get(cat, [])
        if entries:
            selected.append(entries[len(entries) // 2])

    for cat, entries in buckets.items():
        if cat not in ("baseline", "pruning", "noise", "dpsgd"):
            selected.append(entries[len(entries) // 2])

    return selected


def _short_defense_label(name: str) -> str:
    """Convert internal defense names to short readable labels."""
    if name == "none":
        return "No\nDefense"
    if name.startswith("pruning_keep"):
        ratio = name.split("keep")[-1]
        pct = int(float(ratio) * 100)
        return f"Pruning\n({pct}%)"
    if name.startswith("noise_abs_sigma"):
        sigma = name.split("sigma")[-1]
        return f"Noise\n($\\sigma$={sigma})"
    if name.startswith("noise_rel_sigma"):
        sigma = name.split("sigma")[-1]
        return f"Noise rel\n($\\sigma$={sigma})"
    if name.startswith("dpsgd_eps"):
        parts = name.replace("dpsgd_eps", "").split("_delta")
        eps = parts[0]
        # Convert e.g. "1.0e-05" → "$10^{-5}$"
        delta_str = parts[1]
        try:
            exp = int(float(delta_str.split("e")[1]))
            delta_fmt = f"$10^{{{exp}}}$"
        except (IndexError, ValueError):
            delta_fmt = delta_str
        return f"DP-SGD\n($\\epsilon$={eps}, $\\delta$={delta_fmt})"
    if name.startswith("dpsgd_C"):
        parts = name.replace("dpsgd_C", "").split("_sigma")
        return f"DP-SGD\n($C$={parts[0]}, $\\sigma$={parts[1]})"
    return name.replace("_", "\n")


def _save_comparison_chart(
    all_results: List[Dict[str, float]],
    save_dir: Path,
):
    """Generate bar charts comparing metrics across defenses."""
    metric_keys = [k for k in METRIC_KEYS if k != "fid"]
    metric_labels = {
        k: f"{METRIC_LABELS[k]} ({'higher' if k == 'lpips' else 'lower'}=better defended)"
        for k in metric_keys
    }

    names = [r["defense"] for r in all_results]
    present_keys = [k for k in metric_keys if k in all_results[0]]

    fig, axes = plt.subplots(1, len(present_keys), figsize=(4 * len(present_keys), 5))
    if len(present_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, present_keys):
        values = [r.get(key, 0.0) for r in all_results]
        std_values = [r.get(f"{key}_std", 0.0) for r in all_results]

        colors = ["#4CAF50"] + ["#2196F3"] * (len(values) - 1)
        bars = ax.bar(
            range(len(values)),
            values,
            color=colors,
            yerr=std_values,
            capsize=3,
            alpha=0.85,
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric_labels.get(key, key))
        ax.set_title(key.upper())

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format(val, METRIC_FORMATS.get(key, (".3f", ""))[0])
                + METRIC_FORMATS.get(key, ("", ""))[1],
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.suptitle("Defense Impact on Reconstruction Quality", fontsize=13)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        chart_path = save_dir / f"defense_comparison.{ext}"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison chart saved to: {chart_path}")
    plt.close()


def run_defense_evaluation(cfg):
    """
    Main evaluation loop driven by an OmegaConf config.

    Args:
        cfg: Full configuration (data, model, eval, defenses, output).
    """
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    h5_path = data_cfg.get("h5_path")
    gradient_dim = data_cfg.get("gradient_dim", None)
    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    sequence_length = model_cfg.get("sequence_length", 8)
    stride = model_cfg.get("stride", 8)

    print(f"\nLoading test data from: {h5_path}")
    dataset = TemporalGradientDataset(
        h5_path,
        sequence_length=sequence_length,
        stride=stride,
        gradient_dim=gradient_dim,
        gradient_layers=gradient_layers,
    )
    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = len(dataset.actions.unique())

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Sequences: {len(dataset)}")
    print(f"  Gradient dim: {actual_gradient_dim}")

    checkpoint_path = eval_cfg.get("checkpoint")
    print(f"\nLoading model from: {checkpoint_path}")
    model_type = model_cfg.get("model_type", "temporal")
    model = load_model(
        checkpoint_path,
        gradient_dim=actual_gradient_dim,
        device=device,
        num_actions=num_actions,
        latent_dim=model_cfg.get("latent_dim", 512),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        encoder_hidden_dims=model_cfg.get("encoder_hidden_dims", None),
        encoder_type=model_cfg.get("encoder_type", "basic"),
        decoder_type=model_cfg.get("decoder_type", "basic"),
        model_type=model_type,
    )

    defenses = build_defense_configs(cfg)
    num_sequences = eval_cfg.get("num_sequences", 20)
    enable_fid = eval_cfg.get("enable_fid", False)

    save_dir = Path(output_cfg.get("save_dir", "eval_results/defense"))
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, float]] = []

    for defense in defenses:
        tag = defense.name if defense is not None else "none (baseline)"
        print(f"\n{'=' * 60}")
        print(f"Evaluating defense: {tag}")
        if defense is not None:
            print(f"  Config: {defense.summary()}")
            if isinstance(defense, DPSGDDefense) and defense.epsilon is not None:
                print(
                    f"  Privacy budget: (ε={defense.epsilon:.4f}, δ={defense.delta:.1e})"
                    f" | σ={defense.noise_multiplier:.6g}"
                )
        print(f"{'=' * 60}")

        results = evaluate_with_defense(
            model=model,
            dataloader=dataloader,
            device=device,
            defense=defense,
            num_sequences=num_sequences,
            enable_fid=enable_fid,
            num_actions=num_actions,
            collect_first_sequence=True,
            model_type=model_type,
        )

        if isinstance(defense, DPSGDDefense) and defense.epsilon is not None:
            results["epsilon"] = defense.epsilon
            results["delta"] = defense.delta

        _print_results(results)
        all_results.append(results)

    results_path = save_dir / "defense_results.json"
    skip_keys = {"gt_images", "pred_images"}
    _safe = []
    for r in all_results:
        clean = {
            k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
            for k, v in r.items()
            if k not in skip_keys
        }
        _safe.append(clean)
    with open(results_path, "w") as f:
        json.dump(_safe, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    _save_comparison_chart(all_results, save_dir)
    _save_reconstruction_grid(all_results, save_dir)

    _save_summary_csv(all_results, save_dir)


def _print_results(results: Dict[str, float]):
    """Pretty-print a single defense's metrics."""
    print_results(results)


def _save_summary_csv(all_results: List[Dict[str, float]], save_dir: Path):
    """Save a compact comparison table as CSV."""
    columns = ["defense"] + METRIC_KEYS + ["epsilon", "delta"]
    csv_path = save_dir / "defense_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({col: r.get(col, "") for col in columns})

    print(f"  Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: uv run -m defense.evaluate <config_path>"

    cfg = OmegaConf.load(sys.argv[1])
    print(f"Loaded config from {sys.argv[1]}")

    run_defense_evaluation(cfg)

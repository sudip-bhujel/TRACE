"""Compare two attacker models trained on different victims (e.g. PPO vs A2C)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from attacker.data.dataset import TemporalGradientDataset
from attacker.evaluation.evaluate import ACTION_NAMES, load_model
from attacker.evaluation.metrics import (
    MetricsComputer,
    print_per_timestep_results,
    print_results,
)

@dataclass
class VictimSpec:
    """Everything needed to evaluate a single attacker model."""

    name: str
    checkpoint_path: str
    h5_path: str
    gradient_dim: Optional[int]
    gradient_layers: Optional[List[str]]
    sequence_length: int
    stride: int
    model_kwargs: Dict[str, Any]
    teacher_forcing: bool
    enable_fid: bool


def _default_checkpoint_from_cfg(cfg: DictConfig) -> str:
    save_dir = cfg.get("output", {}).get("save_dir", None)
    if save_dir is None:
        raise ValueError(
            "Config has no output.save_dir; pass --<victim>-checkpoint explicitly."
        )
    return str(Path(save_dir) / "best_model.pt")


def _build_spec(
    name: str,
    config_path: str,
    checkpoint_override: Optional[str],
    h5_override: Optional[str],
    enable_fid: bool,
) -> VictimSpec:
    cfg = OmegaConf.load(config_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})

    checkpoint = checkpoint_override or _default_checkpoint_from_cfg(cfg)

    # Prefer explicit test h5 from eval section, else fall back to data.h5_path
    h5_path = h5_override or eval_cfg.get("h5_path", data_cfg.get("h5_path", None))
    if h5_path is None:
        raise ValueError(f"[{name}] No h5 path found in config or CLI override.")

    gradient_layers = data_cfg.get("gradient_layers", None)
    if gradient_layers is not None:
        gradient_layers = list(gradient_layers)

    model_kwargs: Dict[str, Any] = {
        "latent_dim": model_cfg.get("latent_dim", 512),
        "num_transformer_layers": model_cfg.get("num_transformer_layers", 4),
        "num_heads": model_cfg.get("num_heads", 8),
        "encoder_hidden_dims": model_cfg.get("encoder_hidden_dims", None),
        "encoder_hidden_dim": model_cfg.get("encoder_hidden_dim", None),
        "encoder_num_blocks": model_cfg.get("encoder_num_blocks", None),
        "encoder_expansion": model_cfg.get("encoder_expansion", None),
        "encoder_projection_rank": model_cfg.get("encoder_projection_rank", None),
        "encoder_type": model_cfg.get("encoder_type", "basic"),
        "decoder_type": model_cfg.get("decoder_type", "basic"),
        "skip_transformer": model_cfg.get("skip_transformer", False),
        "temporal_model_type": model_cfg.get("temporal_model_type", "transformer"),
        "ff_multiplier": model_cfg.get("ff_multiplier", 4),
        "use_rope": model_cfg.get("use_rope", False),
        "model_type": model_cfg.get("model_type", "temporal"),
    }

    return VictimSpec(
        name=name,
        checkpoint_path=checkpoint,
        h5_path=h5_path,
        gradient_dim=data_cfg.get("gradient_dim", None),
        gradient_layers=gradient_layers,
        sequence_length=model_cfg.get("sequence_length", 8),
        stride=model_cfg.get("stride", 8),
        model_kwargs=model_kwargs,
        teacher_forcing=eval_cfg.get("teacher_forcing", False),
        enable_fid=enable_fid,
    )



def _select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_victim(
    spec: VictimSpec, device: torch.device
) -> Tuple[torch.nn.Module, DataLoader, int, int]:
    """Load (model, dataloader, num_actions, effective_gradient_dim) for one victim."""
    print(f"\n[{spec.name}] Loading test data: {spec.h5_path}")
    dataset = TemporalGradientDataset(
        spec.h5_path,
        sequence_length=spec.sequence_length,
        stride=spec.stride,
        gradient_dim=spec.gradient_dim,
        gradient_layers=spec.gradient_layers,
    )
    actual_gradient_dim = dataset.effective_gradient_dim
    num_actions = int(len(dataset.actions.unique()))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(
        f"[{spec.name}]   sequences={len(dataset)}  grad_dim={actual_gradient_dim}  num_actions={num_actions}"
    )
    print(f"[{spec.name}] Loading checkpoint: {spec.checkpoint_path}")

    model = load_model(
        checkpoint_path=spec.checkpoint_path,
        gradient_dim=actual_gradient_dim,
        device=device,
        num_actions=num_actions,
        **spec.model_kwargs,
    )
    return model, dataloader, num_actions, actual_gradient_dim



def _run_inference(
    model: torch.nn.Module,
    gradients: torch.Tensor,
    images: torch.Tensor,
    device: torch.device,
    model_type: str,
    teacher_forcing: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_images, pred_actions) on the model's device."""
    with torch.no_grad():
        if model_type == "autoregressive":
            pred_images, pred_actions, _, _ = model(
                gradients,
                images=images.to(device) if teacher_forcing else None,
                teacher_forcing=teacher_forcing,
            )
        else:
            pred_images, pred_actions, _, _ = model(gradients)
    return pred_images, pred_actions



def _render_comparison_figure(
    gt_images: torch.Tensor,  # (T, 3, H, W) on CPU
    ppo_pred: torch.Tensor,  # (T, 3, H, W) on CPU
    a2c_pred: torch.Tensor,  # (T, 3, H, W) on CPU
    gt_actions: torch.Tensor,  # (T,) integer labels
    ppo_pred_labels: torch.Tensor,  # (T,)
    a2c_pred_labels: torch.Tensor,  # (T,)
    save_stem: Path,
    title: Optional[str] = None,
    action_names: Optional[List[str]] = None,
):
    """Save a 3-row figure (GT / PPO / A2C) as PNG+PDF."""
    plt.rcParams["font.family"] = "serif"

    T = gt_images.shape[0]
    fig, axes = plt.subplots(3, T, figsize=(2 * T, 6), squeeze=False)

    row_labels = ["GT", "PPO", "A2C"]
    rows = [gt_images, ppo_pred, a2c_pred]
    row_label_tensors: List[Optional[torch.Tensor]] = [
        gt_actions,
        ppo_pred_labels,
        a2c_pred_labels,
    ]

    for r, (row_img, row_label) in enumerate(zip(rows, row_labels)):
        for t in range(T):
            ax = axes[r, t]
            img = row_img[t].permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0:
                ax.set_title(f"$t={t + 1}$", fontsize=12)

            if t == 0:
                ax.set_ylabel(row_label, fontsize=14, rotation=90, labelpad=8)

            # Optional: per-cell action annotation (small, below image)
            action_idx_tensor = row_label_tensors[r]
            if action_idx_tensor is not None and action_names is not None:
                idx = int(action_idx_tensor[t].item())
                if 0 <= idx < len(action_names):
                    ax.text(
                        0.5,
                        -0.08,
                        action_names[idx],
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                    )

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.savefig(save_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def _clean_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN with None and coerce nested structures for JSON safety."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, dict):
            out[k] = _clean_for_json(v)
        else:
            out[k] = v
    return out


def _save_combined_metrics(
    save_dir: Path,
    ppo_results: Dict[str, Any],
    a2c_results: Dict[str, Any],
    ppo_per_t: Dict[str, Dict[int, float]],
    a2c_per_t: Dict[str, Dict[int, float]],
):
    """Write a consolidated ``metrics.json`` and ``metrics.csv`` covering both victims."""
    save_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "ppo": _clean_for_json(
            {
                **ppo_results,
                "per_timestep": {
                    m: {str(t): v for t, v in d.items()} for m, d in ppo_per_t.items()
                },
            }
        ),
        "a2c": _clean_for_json(
            {
                **a2c_results,
                "per_timestep": {
                    m: {str(t): v for t, v in d.items()} for m, d in a2c_per_t.items()
                },
            }
        ),
    }

    json_path = save_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Flat CSV for easy table import: one row per victim, columns for aggregate metrics.
    csv_path = save_dir / "metrics.csv"
    agg_keys_seen: List[str] = []
    for d in (ppo_results, a2c_results):
        for k in d.keys():
            if k != "per_timestep" and k not in agg_keys_seen:
                agg_keys_seen.append(k)

    fieldnames = ["victim"] + agg_keys_seen
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, res in (("ppo", ppo_results), ("a2c", a2c_results)):
            row: Dict[str, Any] = {"victim": name}
            for k in agg_keys_seen:
                v = res.get(k)
                if isinstance(v, float) and math.isnan(v):
                    v = ""
                row[k] = v
            writer.writerow(row)

    # Per-timestep CSV (one row per (victim, metric, timestep))
    per_t_csv = save_dir / "metrics_per_timestep.csv"
    with open(per_t_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["victim", "metric", "timestep", "value"])
        for name, per_t in (("ppo", ppo_per_t), ("a2c", a2c_per_t)):
            for metric, t_map in per_t.items():
                for t in sorted(t_map.keys()):
                    writer.writerow([name, metric, t, per_t[metric][t]])

    print(f"  Combined metrics: {json_path}")
    print(f"  Combined metrics: {csv_path}")
    print(f"  Per-timestep:     {per_t_csv}")


def compare(
    ppo_spec: VictimSpec,
    a2c_spec: VictimSpec,
    save_dir: Path,
    num_sequences: int = 5,
    device_str: str = "auto",
):
    device = _select_device(device_str)
    print(f"Using device: {device}")

    ppo_model, ppo_loader, ppo_num_actions, _ = _load_victim(ppo_spec, device)
    a2c_model, a2c_loader, a2c_num_actions, _ = _load_victim(a2c_spec, device)

    num_actions = max(ppo_num_actions, a2c_num_actions)
    action_names = (
        ACTION_NAMES[:num_actions]
        if num_actions <= len(ACTION_NAMES)
        else [f"a{i}" for i in range(num_actions)]
    )

    ppo_metrics = MetricsComputer(device, compute_fid_flag=ppo_spec.enable_fid)
    a2c_metrics = MetricsComputer(device, compute_fid_flag=a2c_spec.enable_fid)

    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ppo_iter = iter(ppo_loader)
    a2c_iter = iter(a2c_loader)

    # Metric accumulation uses the full test sets (or until exhausted).
    # Figures are generated only for the first ``num_sequences`` pairs.
    figures_generated = 0
    step = 0
    while True:
        step += 1
        ppo_batch = next(ppo_iter, None)
        a2c_batch = next(a2c_iter, None)

        if ppo_batch is None and a2c_batch is None:
            break

        # --- PPO side ---
        if ppo_batch is not None:
            g_p, img_p, act_p = ppo_batch
            g_p = g_p.to(device)
            img_p_device = img_p.to(device)
            pred_p, pred_a_p = _run_inference(
                ppo_model,
                g_p,
                img_p,
                device,
                model_type=ppo_spec.model_kwargs["model_type"],
                teacher_forcing=ppo_spec.teacher_forcing,
            )
            ppo_metrics.update(pred_p, img_p_device, pred_a_p, act_p.to(device))
        else:
            pred_p = None
            img_p = None
            act_p = None
            pred_a_p = None

        # --- A2C side ---
        if a2c_batch is not None:
            g_a, img_a, act_a = a2c_batch
            g_a = g_a.to(device)
            img_a_device = img_a.to(device)
            pred_a, pred_a_a = _run_inference(
                a2c_model,
                g_a,
                img_a,
                device,
                model_type=a2c_spec.model_kwargs["model_type"],
                teacher_forcing=a2c_spec.teacher_forcing,
            )
            a2c_metrics.update(pred_a, img_a_device, pred_a_a, act_a.to(device))
        else:
            pred_a = None
            img_a = None
            act_a = None
            pred_a_a = None

        if (
            figures_generated < num_sequences
            and ppo_batch is not None
            and a2c_batch is not None
        ):
            gt_cpu = img_p[0].detach().cpu()
            ppo_cpu = pred_p[0].detach().clamp(0, 1).cpu()
            a2c_cpu = pred_a[0].detach().clamp(0, 1).cpu()
            gt_act = act_p[0].detach().cpu()
            ppo_act = pred_a_p[0].argmax(dim=-1).detach().cpu()
            a2c_act = pred_a_a[0].argmax(dim=-1).detach().cpu()

            stem = figures_dir / f"compare_seq_{figures_generated + 1:03d}"
            _render_comparison_figure(
                gt_images=gt_cpu,
                ppo_pred=ppo_cpu,
                a2c_pred=a2c_cpu,
                gt_actions=gt_act,
                ppo_pred_labels=ppo_act,
                a2c_pred_labels=a2c_act,
                save_stem=stem,
                title=None,
                action_names=action_names,
            )
            print(
                f"  Saved figure: {stem.with_suffix('.png').name} / {stem.with_suffix('.pdf').name}"
            )
            figures_generated += 1

    ppo_results = ppo_metrics.compute()
    a2c_results = a2c_metrics.compute()
    ppo_per_t = ppo_metrics.compute_per_timestep()
    a2c_per_t = a2c_metrics.compute_per_timestep()

    print_results(
        ppo_results, header=f"\nPPO victim ({ppo_results.get('num_images', 0)} images)"
    )
    if ppo_per_t:
        print("\n  Per-timestep (PPO):")
        print_per_timestep_results(ppo_per_t)
    print_results(
        a2c_results, header=f"\nA2C victim ({a2c_results.get('num_images', 0)} images)"
    )
    if a2c_per_t:
        print("\n  Per-timestep (A2C):")
        print_per_timestep_results(a2c_per_t)

    _save_combined_metrics(save_dir, ppo_results, a2c_results, ppo_per_t, a2c_per_t)
    print(f"\nFigures saved to: {figures_dir}")



def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare PPO vs A2C attacker models (3-row GT/PPO/A2C figures + combined metrics).",
    )
    p.add_argument(
        "--ppo-config",
        default="attacker/config/ppo/base.yaml",
        help="Path to PPO base config.",
    )
    p.add_argument(
        "--a2c-config",
        default="attacker/config/a2c/base.yaml",
        help="Path to A2C base config.",
    )
    p.add_argument(
        "--ppo-checkpoint", default=None, help="Override PPO checkpoint path."
    )
    p.add_argument(
        "--a2c-checkpoint", default=None, help="Override A2C checkpoint path."
    )
    p.add_argument("--ppo-h5", default=None, help="Override PPO test H5 path.")
    p.add_argument("--a2c-h5", default=None, help="Override A2C test H5 path.")
    p.add_argument(
        "--save-dir",
        default="ckpts/trace/compare_victims",
        help="Where to write figures + metrics.",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of 3-row figures to render.",
    )
    p.add_argument("--device", default="auto", help="Device: auto | cuda | mps | cpu.")
    p.add_argument(
        "--no-fid", action="store_true", help="Disable FID computation (faster)."
    )
    return p


def main(argv: Optional[List[str]] = None):
    args = _build_arg_parser().parse_args(argv)

    enable_fid = not args.no_fid

    ppo_spec = _build_spec(
        name="PPO",
        config_path=args.ppo_config,
        checkpoint_override=args.ppo_checkpoint,
        h5_override=args.ppo_h5,
        enable_fid=enable_fid,
    )
    a2c_spec = _build_spec(
        name="A2C",
        config_path=args.a2c_config,
        checkpoint_override=args.a2c_checkpoint,
        h5_override=args.a2c_h5,
        enable_fid=enable_fid,
    )

    print("PPO spec:")
    print(f"  config={args.ppo_config}")
    print(f"  ckpt={ppo_spec.checkpoint_path}")
    print(f"  h5={ppo_spec.h5_path}")
    print("A2C spec:")
    print(f"  config={args.a2c_config}")
    print(f"  ckpt={a2c_spec.checkpoint_path}")
    print(f"  h5={a2c_spec.h5_path}")

    compare(
        ppo_spec=ppo_spec,
        a2c_spec=a2c_spec,
        save_dir=Path(args.save_dir),
        num_sequences=args.num_sequences,
        device_str=args.device,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

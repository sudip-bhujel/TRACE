"""
Evaluation metrics for temporal gradient inversion.

Computes PSNR, SSIM, MS-SSIM, LPIPS, and FID for image reconstruction
quality, along with action prediction accuracy and confusion matrix.

This module is the single source of truth for metric keys, display
formats, and column ordering used by both ``attacker.evaluate`` and
``defense.evaluate``.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm

# ---- Canonical metric definitions (single source of truth) ----

METRIC_KEYS: List[str] = [
    "mse",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "fid",
    "action_accuracy",
]
"""Ordered list of metric keys reported in evaluations."""

METRIC_FORMATS: Dict[str, Tuple[str, str]] = {
    "mse": (".6f", ""),
    "psnr": (".2f", "dB"),
    "ssim": (".4f", ""),
    "ms_ssim": (".4f", ""),
    "lpips": (".4f", ""),
    "fid": (".2f", ""),
    "action_accuracy": (".1f", "%"),
}
"""Mapping from metric key to ``(format_spec, unit_suffix)``."""

METRIC_LABELS: Dict[str, str] = {
    "mse": "MSE",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "ms_ssim": "MS-SSIM",
    "lpips": "LPIPS",
    "fid": "FID",
    "action_accuracy": "Action Accuracy",
}
"""Short human-readable label for each metric."""


def format_metric(results: Dict[str, float], key: str, include_std: bool = True) -> str:
    """
    Format a single metric value with optional ``+/- std``.

    Returns ``"N/A"`` when the value is missing or NaN.
    """
    val = results.get(key)
    if val is None or (isinstance(val, float) and (val != val)):  # NaN check
        return "N/A"
    fmt, unit = METRIC_FORMATS.get(key, (".4f", ""))
    s = format(val, fmt)
    if include_std:
        std = results.get(f"{key}_std")
        if std is not None:
            s += f" +/- {format(std, fmt)}"
    if unit:
        s += f" {unit}"
    return s


def print_results(results: Dict[str, float], header: str = ""):
    """Pretty-print all metrics from a results dict."""
    if header:
        print(header)
    for key in METRIC_KEYS:
        label = METRIC_LABELS.get(key, key)
        print(f"  {label + ':':<18} {format_metric(results, key)}")


def print_per_timestep_results(per_timestep: Dict[str, Dict[int, float]]):
    """Pretty-print per-timestep metrics as a table."""
    if not per_timestep:
        return
    timesteps = sorted(next(iter(per_timestep.values())).keys())
    metrics = list(per_timestep.keys())

    # Header
    header = f"  {'t':<4}" + "".join(f"{m.upper():<12}" for m in metrics)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for t in timesteps:
        row = f"  {t:<4}"
        for m in metrics:
            fmt = METRIC_FORMATS.get(m, (".4f", ""))[0]
            val = per_timestep[m].get(t, float("nan"))
            row += f"{format(val, fmt):<12}"
        print(row)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-image PSNR between predicted and target images.

    Args:
        pred: (N, C, H, W) in [0, 1]
        target: (N, C, H, W) in [0, 1]

    Returns:
        (N,) PSNR values in dB.
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def _gaussian_kernel_2d(
    size: int = 11, sigma: float = 1.5, channels: int = 3
) -> torch.Tensor:
    """Create 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Per-image SSIM using Gaussian-weighted statistics (Wang et al. 2004).

    Args:
        pred: (N, C, H, W) in [0, 1]
        target: (N, C, H, W) in [0, 1]

    Returns:
        (N,) SSIM values.
    """
    C1 = 0.01**2
    C2 = 0.03**2
    channels = pred.shape[1]
    pad = window_size // 2
    kernel = _gaussian_kernel_2d(window_size, sigma, channels).to(
        pred.device, pred.dtype
    )

    mu_pred = F.conv2d(pred, kernel, groups=channels, padding=pad)
    mu_target = F.conv2d(target, kernel, groups=channels, padding=pad)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = (
        F.conv2d(pred.pow(2), kernel, groups=channels, padding=pad) - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target.pow(2), kernel, groups=channels, padding=pad) - mu_target_sq
    )
    sigma_cross = (
        F.conv2d(pred * target, kernel, groups=channels, padding=pad) - mu_cross
    )

    num = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    den = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    return (num / den).mean(dim=(1, 2, 3))


def _ssim_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel: torch.Tensor,
    channels: int,
    pad: int,
) -> tuple:
    """
    Compute the luminance, contrast-structure, and full SSIM maps.

    Returns:
        (luminance, cs, ssim_map) each of shape (N, C, H', W').
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mu_pred = F.conv2d(pred, kernel, groups=channels, padding=pad)
    mu_target = F.conv2d(target, kernel, groups=channels, padding=pad)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = (
        F.conv2d(pred.pow(2), kernel, groups=channels, padding=pad) - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target.pow(2), kernel, groups=channels, padding=pad) - mu_target_sq
    )
    sigma_cross = (
        F.conv2d(pred * target, kernel, groups=channels, padding=pad) - mu_cross
    )

    luminance = (2 * mu_cross + C1) / (mu_pred_sq + mu_target_sq + C1)
    cs = (2 * sigma_cross + C2) / (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = luminance * cs

    return luminance, cs, ssim_map


def compute_ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Multi-Scale SSIM (Wang et al. 2003).

    Computes SSIM contrast-structure at multiple downsampled scales and
    combines them with the luminance term at the coarsest scale.

    Args:
        pred: (N, C, H, W) in [0, 1]
        target: (N, C, H, W) in [0, 1]
        window_size: Gaussian window size.
        sigma: Gaussian standard deviation.
        weights: Per-scale weights.  Defaults to the 5-level weights from
            the original paper: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].

    Returns:
        (N,) MS-SSIM values.
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    levels = len(weights)
    channels = pred.shape[1]
    pad = window_size // 2
    kernel = _gaussian_kernel_2d(window_size, sigma, channels).to(
        pred.device, pred.dtype
    )

    cs_per_level: List[torch.Tensor] = []
    luminance_final = None

    for i in range(levels):
        # Images must be large enough for the Gaussian window
        if pred.shape[2] < window_size or pred.shape[3] < window_size:
            break

        lum, cs, _ = _ssim_components(pred, target, kernel, channels, pad)

        cs_mean = cs.mean(dim=(1, 2, 3)).clamp(min=1e-10)
        cs_per_level.append(cs_mean)

        if i == levels - 1:
            luminance_final = lum.mean(dim=(1, 2, 3)).clamp(min=1e-10)
        else:
            pred = F.avg_pool2d(pred, kernel_size=2)
            target = F.avg_pool2d(target, kernel_size=2)

    # If the image was too small for all levels, fall back to single-scale
    if luminance_final is None:
        lum, cs, _ = _ssim_components(pred, target, kernel, channels, pad)
        luminance_final = lum.mean(dim=(1, 2, 3)).clamp(min=1e-10)
        if not cs_per_level:
            cs_per_level.append(cs.mean(dim=(1, 2, 3)).clamp(min=1e-10))

    # Trim weights to actual number of computed levels
    actual_levels = len(cs_per_level)
    w = weights[:actual_levels]
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]

    result = luminance_final ** w[-1]
    for cs_val, wi in zip(cs_per_level, w):
        result = result * (cs_val**wi)

    return result


def compute_fid(
    real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Fréchet Inception Distance between two feature distributions.

    Args:
        real_features: (N, D) features from real images.
        fake_features: (M, D) features from generated images.
        eps: Small constant added to covariance diagonals for numerical
            stability of the matrix square root.

    Returns:
        FID score (lower is better).
    """
    mu_r = real_features.mean(axis=0)
    mu_f = fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    # Regularise covariance matrices for numerical stability (especially
    # important when the number of samples is smaller than the feature dim).
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_f += np.eye(sigma_f.shape[0]) * eps

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            raise ValueError(
                "Imaginary component of sqrtm is too large; "
                "FID computation is unreliable."
            )
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


class InceptionFeatureExtractor(nn.Module):
    """Extract 2048-dim pool features from InceptionV3 for FID computation."""

    # ImageNet statistics expected by InceptionV3
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device

        self.register_buffer(
            "mean", torch.tensor(self._MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self._STD, device=device).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """
        Args:
            images: (N, 3, H, W) in [0, 1]

        Returns:
            (N, 2048) numpy feature array.
        """
        x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x).cpu().numpy()


class MetricsComputer:
    """
    Accumulates image reconstruction and action prediction metrics
    across batches, then computes aggregate statistics.

    Supported metrics: MSE, PSNR, SSIM, LPIPS, FID, action accuracy.

    Usage::

        mc = MetricsComputer(device, compute_fid=True)
        for pred, target, pred_act, tgt_act in loader:
            mc.update(pred, target, pred_act, tgt_act)
        results = mc.compute()
        mc.save(Path("metrics.json"))
    """

    def __init__(self, device: torch.device, compute_fid_flag: bool = True):
        self.device = device

        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        self.lpips_fn.eval()

        self._compute_fid = compute_fid_flag
        if compute_fid_flag:
            self.inception = InceptionFeatureExtractor(device)

        self.real_features: List[np.ndarray] = []
        self.fake_features: List[np.ndarray] = []

        self.psnr_values: List[float] = []
        self.ssim_values: List[float] = []
        self.ms_ssim_values: List[float] = []
        self.lpips_values: List[float] = []
        self.mse_values: List[float] = []

        # Per-timestep tracking (keyed by timestep index)
        self.psnr_per_timestep: Dict[int, List[float]] = defaultdict(list)
        self.ssim_per_timestep: Dict[int, List[float]] = defaultdict(list)
        self.lpips_per_timestep: Dict[int, List[float]] = defaultdict(list)

        self.all_pred_actions: List[torch.Tensor] = []
        self.all_target_actions: List[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ):
        """
        Accumulate metrics for one batch.

        Args:
            pred_images: (B, T, 3, H, W) or (N, 3, H, W)
            target_images: same shape as pred_images
            pred_actions: (B, T, num_actions) or (N, num_actions) logits
            target_actions: (B, T) or (N,) integer labels
        """
        has_time_dim = pred_images.ndim == 5
        if has_time_dim:
            B, T = pred_images.shape[:2]
            pred_flat = pred_images.reshape(B * T, *pred_images.shape[2:])
            tgt_flat = target_images.reshape(B * T, *target_images.shape[2:])
        else:
            pred_flat = pred_images
            tgt_flat = target_images

        pred_flat = pred_flat.clamp(0, 1).to(self.device)
        tgt_flat = tgt_flat.to(self.device)

        # MSE & PSNR (per-image)
        mse = F.mse_loss(pred_flat, tgt_flat, reduction="none").mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
        self.mse_values.extend(mse.cpu().tolist())
        self.psnr_values.extend(psnr.cpu().tolist())

        # SSIM & MS-SSIM (per-image)
        ssim_vals = compute_ssim(pred_flat, tgt_flat)
        self.ssim_values.extend(ssim_vals.cpu().tolist())

        ms_ssim_vals = compute_ms_ssim(pred_flat, tgt_flat)
        self.ms_ssim_values.extend(ms_ssim_vals.cpu().tolist())

        # LPIPS (per-image; expects [-1, 1] input)
        lp = self.lpips_fn(pred_flat * 2 - 1, tgt_flat * 2 - 1)
        if lp.numel() > 1:
            lp_list = lp.squeeze().cpu().tolist()
        else:
            lp_list = [lp.item()]
        self.lpips_values.extend(lp_list)

        # Per-timestep tracking
        if has_time_dim:
            psnr_2d = psnr.reshape(B, T)
            ssim_2d = ssim_vals.reshape(B, T)
            lp_tensor = torch.tensor(lp_list, dtype=torch.float32).reshape(B, T)
            for t in range(T):
                self.psnr_per_timestep[t].extend(psnr_2d[:, t].cpu().tolist())
                self.ssim_per_timestep[t].extend(ssim_2d[:, t].cpu().tolist())
                self.lpips_per_timestep[t].extend(lp_tensor[:, t].tolist())

        # FID features
        if self._compute_fid:
            self.real_features.append(self.inception(tgt_flat))
            self.fake_features.append(self.inception(pred_flat))

        # Actions
        if pred_actions.ndim == 3:
            pred_labels = pred_actions.argmax(dim=-1).reshape(-1)
            tgt_labels = target_actions.reshape(-1)
        else:
            pred_labels = pred_actions.argmax(dim=-1)
            tgt_labels = target_actions
        self.all_pred_actions.append(pred_labels.cpu())
        self.all_target_actions.append(tgt_labels.cpu())

    def compute(self) -> Dict[str, float]:
        """Return aggregate metrics with mean and standard deviation."""
        results: Dict[str, float] = {}
        n = len(self.mse_values)

        results["mse"] = float(np.mean(self.mse_values)) if n else 0.0
        results["psnr"] = float(np.mean(self.psnr_values)) if n else 0.0
        results["ssim"] = float(np.mean(self.ssim_values)) if n else 0.0
        results["ms_ssim"] = float(np.mean(self.ms_ssim_values)) if n else 0.0
        results["lpips"] = float(np.mean(self.lpips_values)) if n else 0.0

        if n >= 2:
            results["mse_std"] = float(np.std(self.mse_values, ddof=1))
            results["psnr_std"] = float(np.std(self.psnr_values, ddof=1))
            results["ssim_std"] = float(np.std(self.ssim_values, ddof=1))
            results["ms_ssim_std"] = float(np.std(self.ms_ssim_values, ddof=1))
            results["lpips_std"] = float(np.std(self.lpips_values, ddof=1))

        if self._compute_fid and self.real_features:
            real = np.concatenate(self.real_features, axis=0)
            fake = np.concatenate(self.fake_features, axis=0)
            results["fid"] = (
                compute_fid(real, fake) if real.shape[0] >= 2 else float("nan")
            )
        else:
            results["fid"] = float("nan")

        if self.all_pred_actions:
            pred_all = torch.cat(self.all_pred_actions)
            tgt_all = torch.cat(self.all_target_actions)
            results["action_accuracy"] = (
                pred_all == tgt_all
            ).float().mean().item() * 100
        else:
            results["action_accuracy"] = 0.0

        results["num_images"] = n
        return results

    def compute_per_timestep(self) -> Dict[str, Dict[int, float]]:
        """Return per-timestep mean metrics: {metric: {t: mean_value}}."""
        results: Dict[str, Dict[int, float]] = {}
        if not self.psnr_per_timestep:
            return results

        for metric_name, storage in [
            ("psnr", self.psnr_per_timestep),
            ("ssim", self.ssim_per_timestep),
            ("lpips", self.lpips_per_timestep),
        ]:
            per_t = {}
            for t in sorted(storage.keys()):
                vals = storage[t]
                if vals:
                    per_t[t] = float(np.mean(vals))
            if per_t:
                results[metric_name] = per_t
        return results

    def confusion_matrix(self, num_actions: int = 5) -> np.ndarray:
        """Row = true action, column = predicted action."""
        pred = torch.cat(self.all_pred_actions).numpy()
        tgt = torch.cat(self.all_target_actions).numpy()
        cm = np.zeros((num_actions, num_actions), dtype=np.int64)
        for t, p in zip(tgt, pred):
            cm[t, p] += 1
        return cm

    def save(self, path: Path):
        """Save aggregate metrics to both JSON and CSV.

        Writes ``<path>`` as JSON and ``<path>.csv`` (same stem) as CSV.
        Includes per-timestep metrics when temporal data is available.
        """
        results = self.compute()
        clean = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in results.items()
        }

        per_timestep = self.compute_per_timestep()
        if per_timestep:
            clean["per_timestep"] = {
                metric: {str(t): v for t, v in t_vals.items()}
                for metric, t_vals in per_timestep.items()
            }

        path.parent.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)

        # CSV (same directory, same stem) — aggregate only
        csv_path = path.with_suffix(".csv")
        csv_clean = {k: v for k, v in clean.items() if k != "per_timestep"}
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_clean.keys()))
            writer.writeheader()
            writer.writerow(csv_clean)

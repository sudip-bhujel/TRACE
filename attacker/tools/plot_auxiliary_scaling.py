"""Plot auxiliary-data scaling from saved metrics, without running evaluation."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FRACTIONS = (10, 30, 50, 70, 90)
METRICS = {"mse": "MSE", "psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}


def load_metrics(root):
    # The existing full-data reference uses a different validation split.
    records = [json.loads((root / str(f) / "eval_results/metrics.json").read_text())
               for f in FRACTIONS]
    if len({r["num_images"] for r in records}) != 1 or records[0]["num_images"] <= 0:
        raise ValueError("Evaluation image counts must be positive and match across runs")
    values = {metric: [float(r[metric]) for r in records] for metric in METRICS}
    if any(not math.isfinite(v) or v < 0 for series in values.values() for v in series):
        raise ValueError("Metrics must be finite and nonnegative")
    return values


@plt.rc_context({"font.family": "DejaVu Serif", "font.size": 8,
                 "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
def plot_metrics(values, output):
    output.mkdir(parents=True, exist_ok=True)
    for metric, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(1.65, 1.5))
        ax.plot(FRACTIONS, values[metric], "o-", color="#0072B2",
                linewidth=1.2, markersize=3.5)
        ax.set(xlabel="Auxiliary data (%)", ylabel=label, xticks=FRACTIONS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(length=2.5, pad=2)
        ax.margins(x=0.08, y=0.15)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=0.34, right=0.98, bottom=0.26, top=0.96)
        for ext in ("pdf", "png"):
            fig.savefig(output / f"auxiliary_scaling_{metric}.{ext}", dpi=240)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path,
                        default=Path("ckpts/trace/ppo/auxiliary_scaling"))
    parser.add_argument("--output", type=Path,
                        default=Path("ckpts/trace/ppo/auxiliary_scaling/figures"))
    args = parser.parse_args()
    plot_metrics(load_metrics(args.results), args.output)
    print(f"Saved four auxiliary-scaling plots to {args.output}")


if __name__ == "__main__":
    main()

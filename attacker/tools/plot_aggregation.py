"""Plot saved aggregation exports without loading models or using a GPU."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_histograms(path):
    export = json.loads(path.read_text())
    shape = (export["sequence_length"], len(export["action_names"]))
    indexed = {}
    for item in export["sequences"]:
        index = item["sequence_index"]
        if not isinstance(index, int) or index < 0 or index in indexed:
            raise ValueError(f"Invalid or duplicate sequence index: {index}")
        arrays = [np.asarray(item[key], dtype=float) for key in ("target", "prediction")]
        for arr in arrays:
            if (arr.shape != shape or not np.isfinite(arr).all() or (arr < 0).any()
                    or not np.allclose(arr.sum(-1), 1, atol=1e-6)):
                raise ValueError(f"Invalid action proportions in sequence {index + 1}")
        indexed[index] = arrays
    if not indexed:
        raise ValueError("Export sequences must be nonempty")
    return export, indexed


def load_exports(root, comparison_path=None):
    (before, left), (after, right) = [load_histograms(path) for path in (
        root / "before/action_histograms.json",
        comparison_path if comparison_path is not None else root / "after/action_histograms.json")]
    for key in ("h5_path", "sequence_length", "stride", "batch_size",
                "aggregation_size", "teacher_forcing", "action_names"):
        if before[key] != after[key]:
            raise ValueError(f"Before/after export metadata differs: {key}")
    if left.keys() != right.keys():
        raise ValueError("Before/after sequence IDs must match")
    for index in left:
        if not np.array_equal(left[index][0], right[index][0]):
            raise ValueError(f"Ground-truth actions differ in sequence {index + 1}")
    return before["action_names"], left, right


def action_prior(k, prior):
    prior = np.full(k, 1 / k) if prior is None else np.asarray(prior, dtype=float)
    if (prior.shape != (k,) or not np.isfinite(prior).all() or (prior < 0).any()
            or not np.isclose(prior.sum(), 1)):
        raise ValueError("The action prior must contain one probability per action and sum to one")
    return prior


def load_images(root, number, t_count):
    images = []
    for side in ("before", "after"):
        path = root / side / f"reconstruction_seq_{number:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            images.append((data["target_images"], data["predicted_images"]))
    gt = images[0][0]
    if (gt.ndim != 4 or gt.shape[:2] != (t_count, 3)
            or any(arr.shape != gt.shape or not np.isfinite(arr).all()
                   for pair in images for arr in pair)):
        raise ValueError(f"Invalid RGB arrays in sequence {number}")
    if not np.array_equal(gt, images[1][0]):
        raise ValueError(f"Ground-truth images differ in sequence {number}")
    return gt, images[0][1], images[1][1]


def plot_sequence(root, output, number, names, before, after, prior=None):
    t_count = len(before[0])
    images = load_images(root, number, t_count)
    plot_windows(output, f"seq_{number:03d}", names, before[0], before[1], after[1],
                 images, [f"Window {t + 1}" for t in range(t_count)], prior)


@plt.rc_context({"font.family": "DejaVu Serif"})
def plot_windows(output, stem, names, target, pred_before, pred_after, images, window_labels, prior=None):
    t_count, k = target.shape
    prior_label = "Uniform baseline" if prior is None else "Training-frequency baseline"
    prior = action_prior(k, prior)

    output.mkdir(parents=True, exist_ok=True)
    tv_before = 0.5 * np.abs(pred_before - target).sum(-1)
    tv_after = 0.5 * np.abs(pred_after - target).sum(-1)
    columns = min(4, t_count)
    rows = (t_count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows + 1), squeeze=False)
    x = np.arange(k)
    labels = [re.sub(r"([a-z])([A-Z])", r"\1\n\2", name) for name in names]
    for t, ax in enumerate(axes.flat):
        if t >= t_count:
            ax.axis("off")
            continue
        for offset, values, label, color in (
            (-0.24, target[t], "Ground truth", "#525252"),
            (0, pred_before[t], "Before fine-tuning", "#D55E00"),
            (0.24, pred_after[t], "After fine-tuning", "#0072B2"),
        ):
            ax.bar(x + offset, values, width=0.23, label=label, color=color)
        ax.scatter(x, prior, marker="D", s=20, color="#009E73", zorder=4,
                   label=prior_label)
        ax.set_title(f"{window_labels[t]}\nTV: {tv_before[t]:.3f} / {tv_after[t]:.3f}", fontsize=10)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.05), yticks=[0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if t % columns == 0:
            ax.set_ylabel("Action proportion")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.subplots_adjust(top=1 - 1.05 / fig.get_figheight(), bottom=0.10, hspace=0.70, wspace=0.28)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"actions_{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, t_count, figsize=(1.65 * t_count + 0.7, 5.2), squeeze=False)
    for row, (label, frames) in enumerate(zip(
        ("Ground truth\nlast frame", "Before fine-tuning", "After fine-tuning"),
        images,
    )):
        for t, ax in enumerate(axes[row]):
            ax.imshow(frames[t].transpose(1, 2, 0).clip(0, 1), interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(window_labels[t], fontsize=10)
        axes[row, 0].text(-0.14, 0.5, label, ha="center", va="center", rotation=90,
                          transform=axes[row, 0].transAxes, fontsize=10)
    top = 0.90 if any("\n" in label for label in window_labels) else 0.95
    fig.subplots_adjust(left=0.045, right=0.99, top=top, bottom=0.02, wspace=0.04, hspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"reconstructions_{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_panel(ax, errors):
    averages = [values.mean(-1) for values in errors.values()]
    positions = np.arange(len(errors)) * 0.7
    boxes = ax.boxplot(averages, positions=positions, widths=0.45,
                       patch_artist=True, showfliers=False, medianprops={"color": "black"})
    rng = np.random.default_rng(0)
    colors = {"Before fine-tuning": "#D55E00", "After fine-tuning": "#0072B2",
              "From scratch": "#CC79A7", "Uniform baseline": "#009E73",
              "Training-frequency baseline": "#9467BD"}
    for x, label, values, box in zip(positions, errors, averages, boxes["boxes"]):
        color = colors[label]
        box.set(facecolor=color, alpha=0.25)
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(values)), values, s=12, color=color, alpha=0.5)
        ax.scatter(x, values.mean(), marker="D", s=30, color="black", zorder=4)
        ax.text(x, 0.97, rf"$\mu={values.mean():.3f}$", ha="center", va="top", fontsize=9)
    ax.set(xticks=positions,
           xticklabels=[label.replace(" fine-tuning", "\nfine-tuning").replace(" baseline", "\nbaseline").replace(" scratch", "\nscratch")
                        for label in errors], ylim=(0, 1))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


@plt.rc_context({"font.family": "DejaVu Serif"})
def plot_summary(root, output, names, before, after, prior=None, scratch_path=None, agg8_path=None):
    ids = sorted(before)
    target, pre, post = [np.stack(arrays) for arrays in (
        [before[i][0] for i in ids], [before[i][1] for i in ids], [after[i][1] for i in ids])]
    errors = {label: 0.5 * np.abs(prediction - target).sum(-1) for label, prediction in (
        ("Before fine-tuning", pre), ("After fine-tuning", post))}
    if scratch_path is not None:
        _, _, scratch = load_exports(root, scratch_path)
        prediction = np.stack([scratch[i][1] for i in ids])
        errors["From scratch"] = 0.5 * np.abs(prediction - target).sum(-1)
    errors["Uniform baseline"] = 0.5 * np.abs(np.full(len(names), 1 / len(names)) - target).sum(-1)
    if prior is not None:
        errors["Training-frequency baseline"] = 0.5 * np.abs(action_prior(len(names), prior) - target).sum(-1)
    metadata = json.loads((root / "before/action_histograms.json").read_text())
    panels = [("action_error_summary", errors)]
    agg8_summary = None
    if agg8_path is not None:
        agg8, records = load_histograms(agg8_path)
        if metadata["aggregation_size"] != 4 or agg8["aggregation_size"] != 8:
            raise ValueError("Separate summary requires Agg4 and Agg8 exports")
        for key in ("action_names", "sequence_length", "teacher_forcing"):
            if agg8[key] != metadata[key]:
                raise ValueError(f"Agg4/Agg8 export metadata differs: {key}")
        agg8_ids = sorted(records)
        truth, prediction = [np.stack([records[i][j] for i in agg8_ids]) for j in (0, 1)]
        agg8_errors = {"From scratch": 0.5 * np.abs(prediction - truth).sum(-1),
                       "Uniform baseline": 0.5 * np.abs(1 / len(names) - truth).sum(-1)}
        panels.append(("action_error_summary_agg8", agg8_errors))
        agg8_summary = {
            "source_export": str(agg8_path.resolve()), "aggregation_size": 8,
            "num_sequences": len(agg8_ids), "num_windows": int(truth.shape[0] * truth.shape[1]),
            "mean_tv": {label: float(values.mean()) for label, values in agg8_errors.items()},
            "per_sequence": [{"sequence_number": i + 1,
                              **{label: float(values[s].mean()) for label, values in agg8_errors.items()}}
                             for s, i in enumerate(agg8_ids)],
        }
    output.mkdir(parents=True, exist_ok=True)
    for stem, values in panels:
        fig, ax = plt.subplots(figsize=(2.6 if len(values) == 2 else 4.2, 2.8))
        plot_error_panel(ax, values)
        ax.set_ylabel("Mean TV distance")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    percentiles = (10, 50, 90)
    order = np.argsort(errors["After fine-tuning"].ravel(), kind="stable")
    ranks = np.rint(np.asarray(percentiles) / 100 * (len(order) - 1)).astype(int)
    positions = [np.unravel_index(int(order[rank]), target.shape[:2]) for rank in ranks]
    selected_images = [load_images(root, ids[s] + 1, target.shape[1]) for s, t in positions]
    frames = [np.stack([image[row][t] for image, (_, t) in zip(selected_images, positions)])
              for row in range(3)]
    labels = [f"{p}th percentile\nSeq. {ids[s] + 1}, window {t + 1}"
              for p, (s, t) in zip(percentiles, positions)]
    plot_windows(output, "percentiles", names,
                 *[np.stack([arr[s, t] for s, t in positions]) for arr in (target, pre, post)],
                 frames, labels, prior)
    (output / "summary.json").write_text(json.dumps({
        "source_exports": str(root.resolve()),
        "scratch_export": None if scratch_path is None else str(scratch_path.resolve()),
        "agg8": agg8_summary,
        "aggregation_size": metadata["aggregation_size"],
        "num_sequences": len(ids), "num_windows": int(len(order)),
        "action_names": names,
        "training_prior": None if prior is None else action_prior(len(names), prior).tolist(),
        "boxplot": "One dot per sequence; box is IQR, line is median, diamond is mean; whiskers are 1.5 IQR. No confidence intervals.",
        "mean_tv": {label: float(values.mean()) for label, values in errors.items()},
        "per_sequence": [{"sequence_number": i + 1,
                          **{label: float(values[s].mean()) for label, values in errors.items()}}
                         for s, i in enumerate(ids)],
        "selection_rule": "Nearest ranks to the 10th, 50th, and 90th percentiles of post-fine-tuning window TV; ties ordered by sequence and window. Selection does not use image quality.",
        "examples": [{"percentile": p, "sequence_number": ids[s] + 1, "window_number": int(t + 1),
                      "tv": {label: float(values[s, t]) for label, values in errors.items()}}
                     for p, (s, t) in zip(percentiles, positions)],
        "example_tv_order": ["before fine-tuning", "after fine-tuning"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="?", type=Path,
                        default=Path("ckpts/trace/ppo/aggregation/exports"))
    parser.add_argument("--sequences", type=int, nargs="+", help="One-based sequence numbers; default: all")
    parser.add_argument("--summary", action="store_true", help="Plot all-sequence TV and three percentile-selected windows")
    parser.add_argument("--scratch", type=Path, help="Add from-scratch action_histograms.json to the summary plot")
    parser.add_argument("--agg8-scratch", type=Path, help="Add a separate Agg8 figure from action_histograms.json")
    parser.add_argument("--output", type=Path, help="Default: <exports>/figures (or figures/summary with --summary)")
    parser.add_argument("--prior", type=float, nargs="+",
                        help="Training action frequencies, in exported action order; default: uniform")
    args = parser.parse_args()
    if args.summary and args.sequences:
        parser.error("--summary uses all exported sequences; omit --sequences")
    if args.scratch and not args.summary:
        parser.error("--scratch requires --summary")
    if args.agg8_scratch and not args.summary:
        parser.error("--agg8-scratch requires --summary")
    names, before, after = load_exports(args.exports)
    if args.summary:
        output = args.output or args.exports / "figures/summary"
        plot_summary(args.exports, output, names, before, after, args.prior, args.scratch, args.agg8_scratch)
        print(f"Saved summary and percentile examples to {output}")
        return
    numbers = args.sequences if args.sequences is not None else [i + 1 for i in sorted(before)]
    if any(n - 1 not in before for n in numbers):
        parser.error("Requested sequence number is not present in both exports")
    output = args.output or args.exports / "figures"
    print("No-gradient baseline:", "training frequencies" if args.prior is not None else "uniform")
    for number in numbers:
        plot_sequence(args.exports, output, number, names, before[number - 1], after[number - 1], args.prior)
        print(f"Saved sequence {number:03d} to {output}")
    (output / "plot_settings.json").write_text(json.dumps({
        "sequences": numbers,
        "selection": "explicit sequence numbers" if args.sequences else "all exported sequences",
        "action_names": names,
        "prior_source": "user-supplied training frequencies" if args.prior is not None else "uniform",
        "prior": args.prior if args.prior is not None else [1 / len(names)] * len(names),
    }, indent=2))


if __name__ == "__main__":
    main()

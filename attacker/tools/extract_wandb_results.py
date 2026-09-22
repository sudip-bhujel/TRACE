"""Collect paper-ready experiment details from exported W&B run folders."""

import argparse
import csv
import json
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _required(pattern: str, text: str, field: str, cast=str):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {field}")
    return cast(match.group(1).replace(",", ""))


def _metric(text: str, label: str, unit: str | None = None) -> dict:
    match = re.search(
        rf"^\s*{re.escape(label)}:\s*({NUMBER})(?:\s*\+/-\s*({NUMBER}))?",
        text,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Missing evaluation metric {label}")
    result = {"mean": float(match.group(1))}
    if match.group(2) is not None:
        result["std"] = float(match.group(2))
    if unit:
        result["unit"] = unit
    return result


def parse_output_log(text: str) -> dict:
    dataset_pattern = re.compile(
        r"Loading data from (.+?) \(lazy loading for gradients\)\.\.\.\n"
        r"(?:[ \t]+[^\n]+\n)?"
        r"Dataset initialized: ([\d,]+) sequences, seq_len=(\d+), stride=(\d+)"
    )
    datasets = dataset_pattern.findall(text)
    if len(datasets) < 2:
        raise ValueError("Expected training and evaluation dataset details")

    def dataset(values: tuple[str, str, str, str]) -> dict:
        path, sequences, seq_len, stride = values
        return {
            "path": path,
            "available_sequences": int(sequences.replace(",", "")),
            "sequence_length": int(seq_len),
            "stride": int(stride),
        }

    evaluation = text[text.index("Evaluation Summary") :]
    timestep_section = evaluation.split("Per-timestep metrics:", 1)[1]
    timesteps = [
        {
            "timestep": int(t),
            "psnr_db": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips),
        }
        for t, psnr, ssim, lpips in re.findall(
            rf"^\s*(\d+)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$",
            timestep_section,
            re.MULTILINE,
        )
    ]
    if not timesteps:
        raise ValueError("Missing per-timestep evaluation metrics")

    train_dataset = dataset(datasets[0])
    train_dataset.update(
        {
            "train_sequences": _required(r"^\s*Train:\s*([\d,]+)", text, "train split", int),
            "validation_sequences": _required(
                r"^\s*Val:\s*([\d,]+)", text, "validation split", int
            ),
            "augmented": "augmented" in datasets[0][0],
        }
    )

    return {
        "hardware": {
            "device": _required(r"^Using device:\s*(.+)$", text, "device"),
            "world_size": _required(
                r"^DDP enabled: world_size=(\d+)", text, "DDP world size", int
            ),
            "accumulation_steps": _required(
                r"^DDP enabled:.*accumulation_steps=(\d+)",
                text,
                "accumulation steps",
                int,
            ),
            "flash_attention": _required(
                r"^Flash Attention available:\s*(True|False)",
                text,
                "Flash Attention status",
                lambda value: value == "True",
            ),
        },
        "dataset": {
            "training": train_dataset,
            "evaluation": dataset(datasets[1]),
            "number_of_actions": _required(
                r"^\s*Number of actions:\s*(\d+)", text, "number of actions", int
            ),
        },
        "model": {
            "attacker_parameters": _required(
                r"^Model parameters:\s*([\d,]+)", text, "model parameters", int
            )
        },
        "training": {
            "best_validation_loss": _required(
                rf"^Training complete[.!] Best val loss:\s*({NUMBER})",
                text,
                "best validation loss",
                float,
            )
        },
        "evaluation": {
            "images": _required(
                r"Evaluation Summary \(([\d,]+) images", text, "evaluation images", int
            ),
            "sequences": _required(
                r"Evaluation Summary \([\d,]+ images from ([\d,]+) sequences\)",
                text,
                "evaluation sequences",
                int,
            ),
            "metrics": {
                "mse": _metric(evaluation, "MSE"),
                "psnr": _metric(evaluation, "PSNR", "dB"),
                "ssim": _metric(evaluation, "SSIM"),
                "ms_ssim": _metric(evaluation, "MS-SSIM"),
                "lpips": _metric(evaluation, "LPIPS"),
                "fid": _metric(evaluation, "FID"),
                "action_accuracy": _metric(evaluation, "Action Accuracy", "percent"),
            },
            "per_timestep": timesteps,
        },
    }


def _training_runtime(history_path: Path) -> tuple[float, int]:
    with history_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    runtimes = [float(row["_runtime"]) for row in rows if row.get("_runtime")]
    epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
    if not runtimes or not epochs:
        raise ValueError(f"Missing _runtime or epoch in {history_path}")
    return max(runtimes), max(epochs)


def parse_run(run_dir: Path) -> dict:
    result = parse_output_log((run_dir / "output.log").read_text())
    runtime_seconds, epochs = _training_runtime(run_dir / "history.csv")
    world_size = result["hardware"]["world_size"]
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    metadata = json.loads((run_dir / "wandb-metadata.json").read_text())

    result["hardware"].update(
        {
            "gpu_model": metadata.get("gpu"),
            "gpu_count": metadata.get("gpu_count"),
            "cpu_count": metadata.get("cpu_count"),
            "system_memory_bytes": (
                int(metadata["memory"]["total"])
                if metadata.get("memory", {}).get("total")
                else None
            ),
            "python": metadata.get("python"),
            "operating_system": metadata.get("os"),
        }
    )

    result["run"] = json.loads((run_dir / "run.json").read_text())
    result["configuration"] = {
        key: config[key]
        for key in ("data", "model", "training", "loss", "eval")
        if key in config
    }
    result["training"].update(
        {
            "epochs": epochs,
            "wall_time_seconds": runtime_seconds,
            "wall_time_hours": round(runtime_seconds / 3600, 3),
            "gpu_hours": round(runtime_seconds * world_size / 3600, 3),
            "duration_source": "maximum _runtime in history.csv",
            "final_metrics": {
                key: value
                for key, value in summary.items()
                if key.startswith(("train/", "val/")) or key in ("epoch", "lr")
            },
        }
    )
    result["source_files"] = {
        "output_log": str(run_dir / "output.log"),
        "history": str(run_dir / "history.csv"),
        "wandb_summary": str(run_dir / "summary.json"),
        "wandb_config": str(run_dir / "config.json"),
        "wandb_metadata": str(run_dir / "wandb-metadata.json"),
    }
    return result


def _result_key(result: dict) -> str:
    run = result["run"]
    return f"{run.get('group') or 'ungrouped'}/{run['name']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/wandb"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logs = sorted(args.input_dir.rglob("output.log"))
    if not logs:
        raise RuntimeError(f"No output.log files found under {args.input_dir}")

    results = {}
    skipped = 0
    for log in logs:
        if "Evaluation Summary" not in log.read_text():
            skipped += 1
            continue
        result = parse_run(log.parent)
        key = _result_key(result)
        current = results.get(key)
        if current is None or str(result["run"]["created_at"]) > str(
            current["run"]["created_at"]
        ):
            results[key] = result
    output = args.output or args.input_dir / "paper_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": results}, indent=2, sort_keys=True) + "\n")
    print(f"Extracted {len(results)} evaluation runs; skipped {skipped} without summaries -> {output}")


if __name__ == "__main__":
    main()

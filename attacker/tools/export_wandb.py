"""Export W&B runs selected by attacker config files or directories."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import wandb

from victim.config_utils import load_config


def _latest_finished_run(
    api: wandb.Api, project_path: str, run_name: str, group: str | None = None
):
    filters = {"display_name": run_name, "state": "finished"}
    if group:
        filters["group"] = group
    runs = list(
        api.runs(
            project_path,
            filters=filters,
        )
    )
    if not runs:
        raise RuntimeError(f"No finished W&B run named {run_name!r} in {project_path}")
    return max(runs, key=lambda run: str(run.created_at or ""))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _safe_parts(value: str) -> tuple[str, ...]:
    parts = []
    for part in value.split("/"):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._")
        if safe:
            parts.append(safe)
    return tuple(parts)


def _config_paths(inputs: list[str]) -> list[Path]:
    paths = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("*.yaml"))
            paths.update(path.rglob("*.yml"))
        elif path.is_file() and path.suffix in {".yaml", ".yml"}:
            paths.add(path)
        else:
            raise ValueError(f"Not a YAML file or directory: {path}")
    return sorted(paths)


def _wandb_key(config_path: Path) -> tuple[str, str, str, str] | None:
    wandb_cfg = load_config(str(config_path)).get("wandb")
    if not wandb_cfg or wandb_cfg.get("enabled") is False:
        return None
    if any(not wandb_cfg.get(key) for key in ("entity", "project", "name")):
        return None
    return tuple(
        str(wandb_cfg.get(key) or "") for key in ("entity", "project", "group", "name")
    )


def export_run(api: wandb.Api, config_path: str | Path, output_root: Path) -> Path:
    cfg = load_config(str(config_path))
    wandb_cfg = cfg.get("wandb")
    if not wandb_cfg:
        raise ValueError(f"Missing wandb section in {config_path}")

    missing = [key for key in ("entity", "project", "name") if not wandb_cfg.get(key)]
    if missing:
        raise ValueError(f"Missing wandb {', '.join(missing)} in {config_path}")

    entity = str(wandb_cfg.entity)
    project = str(wandb_cfg.project)
    run_name = str(wandb_cfg.name)
    group = str(wandb_cfg.get("group") or "")
    run = _latest_finished_run(api, f"{entity}/{project}", run_name, group)

    hierarchy = (group or "ungrouped", run_name)
    output_dir = output_root.joinpath(*(part for value in hierarchy for part in _safe_parts(value)))
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(run.scan_history())
    history.to_csv(output_dir / "history.csv", index=False)
    for filename in ("output.log", "wandb-metadata.json"):
        remote_file = run.file(filename)
        if remote_file is None:
            raise RuntimeError(f"Run {run_name!r} has no {filename}")
        remote_file.download(root=str(output_dir), replace=True)
    _write_json(output_dir / "summary.json", dict(run.summary))
    _write_json(output_dir / "config.json", dict(run.config))
    _write_json(
        output_dir / "run.json",
        {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "entity": entity,
            "project": project,
            "created_at": run.created_at,
            "group": run.group,
            "tags": run.tags,
            "source_config": str(Path(config_path)),
        },
    )
    print(
        f"Exported {run_name} ({run.id}): {len(history)} history rows, log, and metadata"
        f" -> {output_dir}"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Attacker YAML files or directories")
    parser.add_argument("--output-dir", default="results/wandb")
    args = parser.parse_args()

    api = wandb.Api()
    output_root = Path(args.output_dir)
    configs = _config_paths(args.configs)
    directory_mode = any(Path(value).is_dir() for value in args.configs)
    seen = set()
    exported = skipped = 0
    for config_path in configs:
        key = _wandb_key(config_path)
        if key is None or key in seen:
            continue
        seen.add(key)
        try:
            export_run(api, config_path, output_root)
            exported += 1
        except RuntimeError as error:
            if not directory_mode:
                raise
            skipped += 1
            print(f"Skipped {config_path}: {error}")
    print(f"Exported {exported} runs; skipped {skipped} unavailable runs")


if __name__ == "__main__":
    main()

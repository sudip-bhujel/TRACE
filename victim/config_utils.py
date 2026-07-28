from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str,
    cli_overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    """Load a config, recursively merge its optional relative ``_base_`` config."""
    path = Path(config_path)
    cfg = OmegaConf.load(path)
    base_ref = cfg.get("_base_")

    if base_ref is not None:
        del cfg["_base_"]
        base_path = (path.parent / str(base_ref)).resolve()
        base_cfg = load_config(str(base_path))
        cfg = OmegaConf.merge(base_cfg, cfg)

    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(cli_overrides)))

    return cfg

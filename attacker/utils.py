from pathlib import Path

from omegaconf import OmegaConf


def load_config_with_defaults(config_path: str) -> OmegaConf:
    """Load a YAML config and merge it over any base configs listed in ``defaults``."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config file not found: {config_path}, using defaults")
        return OmegaConf.create({})

    cfg = OmegaConf.load(config_path)

    if "defaults" in cfg:
        base_cfgs = []
        for default in cfg.defaults:
            base_path = config_path.parent / f"{default}.yaml"
            if base_path.exists():
                base_cfgs.append(OmegaConf.load(base_path))
            else:
                print(f"Warning: Base config not found: {base_path}")

        cfg_without_defaults = OmegaConf.create(
            {k: v for k, v in cfg.items() if k != "defaults"}
        )
        return OmegaConf.merge(*base_cfgs, cfg_without_defaults)

    return cfg

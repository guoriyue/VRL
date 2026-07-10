"""Config loading utilities (verl-style YAML overlay via OmegaConf)."""

from typing import Any

from omegaconf import DictConfig

from vrl.config.loading import bundled_config_resource, list_bundled_configs, load_config


def build_configs(cfg: DictConfig) -> dict[str, Any]:
    """Build typed configs without loading the training stack during discovery."""

    from vrl.config.builders import build_configs as _build_configs

    return _build_configs(cfg)


__all__ = ["build_configs", "bundled_config_resource", "list_bundled_configs", "load_config"]

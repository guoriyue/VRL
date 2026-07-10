"""Config loading utilities (verl-style YAML overlay via OmegaConf)."""

from vrl.config.builders import build_configs
from vrl.config.loading import bundled_config_resource, list_bundled_configs, load_config

__all__ = ["build_configs", "bundled_config_resource", "list_bundled_configs", "load_config"]

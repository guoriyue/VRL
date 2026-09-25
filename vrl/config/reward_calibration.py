"""Public configuration reference to a qualified reward deployment receipt."""

from pathlib import Path

from vrl.config.base import ConfigBase


class RewardCalibrationConfig(ConfigBase):
    deployment_path: Path

"""Public configuration reference to a qualified, frozen reward deployment."""

from pathlib import Path

from pydantic import Field

from vrl.config.base import ConfigBase


class RewardCalibrationConfig(ConfigBase):
    deployment_path: Path
    deployment_id: str = Field(pattern=r"^[0-9a-f]{64}$")

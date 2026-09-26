"""Immutable image-text requirements shared by scoring and visual task contracts."""

from __future__ import annotations

from pydantic import ConfigDict, Field, StrictInt, model_validator

from vrl.config.base import ConfigBase


class TextRegion(ConfigBase):
    """An ordered expected transcript inside an absolute, half-open pixel box."""

    model_config = ConfigDict(frozen=True)

    region_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    text: str = Field(min_length=1)
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]


class TextLayout(ConfigBase):
    """Fixed canvas and uniquely named regions; resizing is never implicit."""

    model_config = ConfigDict(frozen=True)

    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)
    regions: tuple[TextRegion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_regions(self) -> TextLayout:
        if len({region.region_id for region in self.regions}) != len(self.regions):
            raise ValueError("text region IDs must be unique")
        for region in self.regions:
            left, top, right, bottom = region.box
            if not 0 <= left < right <= self.width or not 0 <= top < bottom <= self.height:
                raise ValueError("text region box must be nonempty and inside the canvas")
            if not region.text.strip():
                raise ValueError("text region target must contain visible characters")
        return self

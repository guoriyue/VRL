from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vrl.scripts.cosmos.train import _load_required_reference_image


def test_cosmos_reference_image_is_required() -> None:
    with pytest.raises(ValueError, match="model.reference_image"):
        _load_required_reference_image("")


def test_cosmos_reference_image_path_must_exist() -> None:
    with pytest.raises(FileNotFoundError, match="model.reference_image"):
        _load_required_reference_image("/tmp/does-not-exist-cosmos-reference.png")


def test_cosmos_reference_image_loads_as_rgb(tmp_path: Path) -> None:
    path = tmp_path / "reference.png"
    Image.new("RGBA", (8, 8), color=(255, 0, 0, 128)).save(path)

    image = _load_required_reference_image(str(path))

    assert image.mode == "RGB"
    assert image.size == (8, 8)

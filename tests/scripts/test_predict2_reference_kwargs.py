"""Predict2 reference inputs are normalized at the dataset boundary."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.scripts.diffusion.cosmos.train import _predict2_collector_kwargs


def _cfg(reference_image: str) -> object:
    return OmegaConf.create(
        {
            "data": {
                "manifest": "manifest.jsonl",
                "preprocessing": {
                    "conditioning": "reference_image",
                    "reference_image": reference_image,
                },
            },
        },
    )


def test_dataset_default_fills_missing_row_reference(tmp_path) -> None:
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    example = type("Example", (), {"reference_image": None})()

    kwargs = _predict2_collector_kwargs(_cfg(str(ref)), examples=[example])

    assert kwargs == {}
    assert example.reference_image == str(ref.resolve())


def test_global_mode_requires_existing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _predict2_collector_kwargs(
            _cfg(str(tmp_path / "missing.png")),
            examples=[type("Example", (), {"reference_image": None})()],
        )


def test_missing_row_and_default_raise() -> None:
    with pytest.raises(ValueError, match="missing required field reference_image"):
        _predict2_collector_kwargs(
            _cfg(""),
            examples=[type("Example", (), {"reference_image": None})()],
        )

"""Predict2 global reference mode must ship a path string, not a PIL image."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.scripts.diffusion.cosmos.train import _predict2_collector_kwargs


def _cfg(reference_image: str) -> object:
    return OmegaConf.create(
        {
            "cosmos": {"reference_mode": "global"},
            "model": {"reference_image": reference_image},
        },
    )


def test_global_mode_passes_validated_path_string(tmp_path) -> None:
    """The kwarg must survive the launch contract's primitive-only check."""

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")

    kwargs = _predict2_collector_kwargs(_cfg(str(ref)), examples=[])

    assert kwargs == {"reference_image": str(ref)}
    GenerationRuntimeLaunchContract._validate_serializable_config(
        kwargs,
        "executor_kwargs",
    )


def test_global_mode_requires_existing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _predict2_collector_kwargs(_cfg(str(tmp_path / "missing.png")), examples=[])


def test_global_mode_requires_non_empty_path() -> None:
    with pytest.raises(ValueError, match=r"requires model\.reference_image"):
        _predict2_collector_kwargs(_cfg(""), examples=[])

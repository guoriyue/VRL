"""Wan I2V collector hook: validation only, no metadata rewriting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.scripts.diffusion.wan_2_1.train import _i2v_collector_kwargs


def _cfg(global_reference: str | None = None) -> object:
    return OmegaConf.create(
        {
            "data": {
                "manifest": "manifest.jsonl",
                "preprocessing": {
                    "conditioning": "reference_image",
                    "reference_image": global_reference,
                },
            },
        },
    )


def test_valid_rows_pass_and_metadata_stays_untouched(tmp_path: Path) -> None:
    """The hook validates; it must not write markers into example metadata.

    Regression for the removed dead write: ``conditioning=reference_image`` had
    no reader anywhere (the batch-context marker comes from the wan model's
    ``export_batch_context``, not from example metadata).
    """
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    metadata = {"source_episode": "episode_0"}
    example = SimpleNamespace(reference_image=str(ref), metadata=metadata)

    assert _i2v_collector_kwargs(_cfg(), [example]) == {}

    assert example.metadata is metadata
    assert example.metadata == {"source_episode": "episode_0"}


def test_missing_row_reference_falls_back_to_global(tmp_path: Path) -> None:
    ref = tmp_path / "g.png"
    ref.write_bytes(b"png")
    example = SimpleNamespace(reference_image=None, metadata={})
    assert _i2v_collector_kwargs(_cfg(str(ref)), [example]) == {}
    assert example.reference_image == str(ref.resolve())


def test_missing_row_reference_without_global_raises() -> None:
    example = SimpleNamespace(reference_image=None, metadata={})
    with pytest.raises(ValueError, match="missing required field"):
        _i2v_collector_kwargs(_cfg(), [example])


def test_nonexistent_row_reference_raises(tmp_path: Path) -> None:
    example = SimpleNamespace(reference_image=str(tmp_path / "missing.png"), metadata={})
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _i2v_collector_kwargs(_cfg(), [example])

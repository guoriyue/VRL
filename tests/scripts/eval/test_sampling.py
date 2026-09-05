"""Tests for the shared eval sampling projection."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.scripts.eval._sampling import resolve_eval_sampling


def _root(*, sampling: dict, executor: dict | None = None):
    model = {"family": "cosmos-predict2-anima"}
    if executor is not None:
        model["executor"] = executor
    return parse_config(OmegaConf.create({"model": model, "sampling": sampling}))


_IMAGE = {"width": 512, "height": 512, "num_steps": 10, "guidance_scale": 4.5}


def test_family_key_declared_in_sampling_wins_over_executor() -> None:
    root = _root(
        sampling={**_IMAGE, "max_sequence_length": 256},
        executor={"max_sequence_length": 128},
    )

    assert resolve_eval_sampling(root)["max_sequence_length"] == 256


def test_family_key_falls_back_to_model_executor_like_the_runtime() -> None:
    root = _root(sampling=_IMAGE, executor={"max_sequence_length": 128})

    assert resolve_eval_sampling(root)["max_sequence_length"] == 128


def test_family_key_missing_everywhere_names_the_config_path() -> None:
    root = _root(sampling=_IMAGE)

    with pytest.raises(
        ValueError, match=r"config missing required field: sampling\.max_sequence_length"
    ):
        resolve_eval_sampling(root)


def test_cli_override_wins_and_explicit_zero_guidance_is_kept() -> None:
    root = _root(sampling=_IMAGE, executor={"max_sequence_length": 128})

    out = resolve_eval_sampling(root, overrides={"num_steps": 3, "guidance_scale": 0.0})

    assert out["num_steps"] == 3
    assert out["guidance_scale"] == 0.0

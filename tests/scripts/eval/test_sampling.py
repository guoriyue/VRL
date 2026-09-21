"""Tests for the shared eval sampling projection."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.scripts.eval._sampling import resolve_eval_sampling


def _root(*, sampling: dict, executor: dict | None = None, eval: dict | None = None):
    model = {"family": "sd3_5"}
    if executor is not None:
        model["executor"] = executor
    payload = {"model": model, "sampling": sampling}
    if eval is not None:
        payload["eval"] = eval
    return parse_config(OmegaConf.create(payload))


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


def test_eval_sampling_num_steps_overrides_training_and_yields_to_the_cli() -> None:
    """Flow-GRPO shape: train on 10 steps, evaluate on 40; a CLI value still wins."""
    root = _root(
        sampling=_IMAGE,
        executor={"max_sequence_length": 128},
        eval={"num_steps": 40},
    )

    assert resolve_eval_sampling(root)["num_steps"] == 40
    assert resolve_eval_sampling(root, overrides={"num_steps": 3})["num_steps"] == 3
    assert resolve_eval_sampling(root)["guidance_scale"] == 4.5


def test_eval_sampling_rejects_a_non_positive_step_count() -> None:
    with pytest.raises(ValueError, match=r"eval\.num_steps must be >= 1"):
        _root(sampling=_IMAGE, eval={"num_steps": 0})


def test_eval_sampling_resolution_overrides_training_geometry() -> None:
    """Train at 512px, evaluate at the checkpoint's native 1024px."""
    root = _root(
        sampling={**_IMAGE, "max_sequence_length": 256},
        eval={"width": 1024, "height": 1024, "num_steps": 40},
    )

    out = resolve_eval_sampling(root)

    assert (out["width"], out["height"], out["num_steps"]) == (1024, 1024, 40)


def test_eval_sampling_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match=r"eval\.width must be >= 1"):
        _root(sampling=_IMAGE, eval={"width": 0})

"""Online recipe controller configuration ownership tests."""

from __future__ import annotations

import pytest

from vrl.config.schema import RootConfig
from vrl.run import OnlineRunConfig


def _root(**trainer: object) -> RootConfig:
    return RootConfig.model_validate({"trainer": trainer})


def test_online_run_config_uses_controller_defaults() -> None:
    run = OnlineRunConfig.from_root(_root(total_epochs=0))

    assert run.total_epochs == 0
    assert run.save_freq == 50
    assert run.seed == 0


def test_online_run_config_preserves_explicit_values() -> None:
    run = OnlineRunConfig.from_root(
        _root(total_epochs=12, save_freq=3, seed=-7),
    )

    assert run == OnlineRunConfig(total_epochs=12, save_freq=3, seed=-7)


@pytest.mark.parametrize(
    ("trainer", "path"),
    [
        ({}, "trainer.total_epochs"),
        ({"total_epochs": -1}, "trainer.total_epochs"),
        ({"total_epochs": 1, "save_freq": -1}, "trainer.save_freq"),
    ],
)
def test_online_run_config_rejects_invalid_controller_values(
    trainer: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        OnlineRunConfig.from_root(_root(**trainer))

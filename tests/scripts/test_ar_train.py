"""Family resolution tests for the generic AR GRPO entrypoint."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from omegaconf import OmegaConf

import vrl.scripts.ar.train as ar_train


def _cfg(algorithm_kind: str) -> Any:
    return OmegaConf.create(
        {
            "model": {"family": "janus_pro"},
            "algorithm": {"kind": algorithm_kind},
        },
    )


@pytest.mark.parametrize(
    ("algorithm_kind", "expected_family"),
    [
        # The shipped R1 recipe keeps model.family=janus_pro and selects the
        # r1 variant through algorithm.kind — the entrypoint must hand the
        # resolved family to the recipe, or the factory's multisegment guard
        # rejects the run before launch.
        ("token_grpo_multisegment", "janus_pro_r1"),
        ("token_grpo", "janus_pro"),
    ],
)
def test_train_ar_grpo_hands_resolved_online_family_to_recipe(
    monkeypatch: pytest.MonkeyPatch,
    algorithm_kind: str,
    expected_family: str,
) -> None:
    """Checks train_ar_grpo resolves the online family, not just model.family."""
    captured: dict[str, str] = {}

    async def fake_run_online_recipe(cfg: Any, definition: Any) -> None:
        del cfg
        captured["family"] = definition.family

    monkeypatch.setattr(ar_train, "run_online_recipe", fake_run_online_recipe)

    asyncio.run(ar_train.train_ar_grpo(_cfg(algorithm_kind)))

    assert captured["family"] == expected_family

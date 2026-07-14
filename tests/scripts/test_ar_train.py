"""Family ownership tests for the generic AR GRPO entrypoint."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from omegaconf import OmegaConf

import vrl.scripts.ar.train as ar_train


def _cfg(family: str, algorithm_kind: str) -> Any:
    return OmegaConf.create(
        {
            "model": {"family": family},
            "algorithm": {"kind": algorithm_kind},
        },
    )


@pytest.mark.parametrize(
    ("family", "algorithm_kind"),
    [
        ("janus_pro_r1", "token_grpo_multisegment"),
        ("janus_pro", "token_grpo"),
    ],
)
def test_train_ar_grpo_keeps_family_owned_by_config(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    algorithm_kind: str,
) -> None:
    """The generic entrypoint passes config through without a family wrapper."""
    captured: dict[str, object] = {}

    async def fake_run_online_recipe(cfg: Any) -> None:
        captured["family"] = str(cfg.model.family)

    monkeypatch.setattr(ar_train, "run_online_recipe", fake_run_online_recipe)

    asyncio.run(ar_train.train_ar_grpo(_cfg(family, algorithm_kind)))

    assert captured == {"family": family}

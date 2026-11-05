"""Fixtures shared by the continuous rollout test family."""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.continuous import types as continuous_types


@pytest.fixture(autouse=True)
def fast_queue_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll ready slots every millisecond: the production 50 ms tick only
    bounds idle CPU, and a tiny-SANA group finishes in far less."""

    monkeypatch.setattr(continuous_types, "QUEUE_POLL_INTERVAL_S", 0.001)

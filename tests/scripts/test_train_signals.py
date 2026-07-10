from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from typing import Any

import pytest

from vrl.scripts import train


class _SignalHandlers:
    def __init__(self) -> None:
        self.original = {
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        self.current = dict(self.original)

    def install(self, signum: signal.Signals, handler: Any) -> Any:
        previous = self.current[signum]
        self.current[signum] = handler
        return previous

    def get(self, signum: signal.Signals) -> Any:
        return self.current[signum]

    def send(self, signum: signal.Signals) -> None:
        handler = self.current[signum]
        assert callable(handler)
        handler(signum, None)


def _install_cli_fakes(
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
) -> _SignalHandlers:
    handlers = _SignalHandlers()
    monkeypatch.setattr("vrl.config.loading.load_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "run_config", lambda cfg: result)
    monkeypatch.setattr(signal, "getsignal", handlers.get)
    monkeypatch.setattr(signal, "signal", handlers.install)
    return handlers


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_main_cancels_async_trainer_and_restores_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    state = {"cleanup_finished": False}
    handlers: _SignalHandlers

    async def run_until_signalled() -> None:
        asyncio.get_running_loop().call_soon(handlers.send, signum)
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            state["cleanup_finished"] = True

    handlers = _install_cli_fakes(monkeypatch, run_until_signalled())

    with pytest.raises(SystemExit) as exc_info:
        train.main(["--config", "test"])

    assert exc_info.value.code == 128 + int(signum)
    assert state["cleanup_finished"] is True
    assert handlers.current == handlers.original


def test_main_restores_signal_handlers_after_async_trainer_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def finish() -> None:
        await asyncio.sleep(0)

    handlers = _install_cli_fakes(monkeypatch, finish())

    train.main(["--config", "test"])

    assert handlers.current == handlers.original


def test_main_restores_signal_handlers_after_async_trainer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail() -> None:
        raise RuntimeError("training failed")

    handlers = _install_cli_fakes(monkeypatch, fail())

    with pytest.raises(RuntimeError, match="training failed"):
        train.main(["--config", "test"])

    assert handlers.current == handlers.original


def test_main_keeps_synchronous_trainer_path_signal_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = object()
    monkeypatch.setattr("vrl.config.loading.load_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "run_config", lambda cfg: result)

    def reject_signal_install(signum: signal.Signals, handler: Callable[..., Any]) -> None:
        del signum, handler
        raise AssertionError("synchronous trainers must not replace process signal handlers")

    monkeypatch.setattr(signal, "signal", reject_signal_install)

    train.main(["--config", "test"])

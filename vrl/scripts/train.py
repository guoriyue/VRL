"""Unified YAML-driven training entry point.

The experiment name and implementation entrypoint belong in the bundled
``vrl/config/presets/experiment/**/*.yaml`` files. This module is only the
CLI/import layer: it loads one YAML config, imports ``trainer.entrypoint``, then
runs it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import signal
from collections.abc import Awaitable
from dataclasses import dataclass
from types import FrameType
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class TrainTarget:
    """Resolved implementation for one merged training config."""

    import_path: str


def resolve_train_target(cfg: DictConfig) -> TrainTarget:
    """Resolve a merged YAML config to its declared training callable."""

    try:
        import_path = cfg.trainer.entrypoint
    except Exception as exc:
        raise ValueError("config missing required field: trainer.entrypoint") from exc
    if not isinstance(import_path, str) or not import_path.strip():
        raise ValueError("trainer.entrypoint must be a non-empty import path")
    return TrainTarget(import_path.strip())


def _import_callable(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(
            "trainer.entrypoint must use 'module:function' import path syntax",
        )
    module_name, attr_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def run_config(cfg: DictConfig) -> Any:
    """Run the family trainer selected by ``cfg``."""

    target = resolve_train_target(cfg)
    trainer = _import_callable(target.import_path)
    return trainer(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YAML-driven VRL training job.")
    parser.add_argument(
        "--config",
        required=True,
        help="Bundled config name or absolute YAML path.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. trainer.seed=42 actor.optim.lr=2e-4",
    )
    return parser


async def _run_async_trainer(result: Awaitable[Any]) -> signal.Signals | None:
    """Run an async trainer until it completes or the CLI receives a stop signal."""

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(result)
    received_signal: signal.Signals | None = None
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        del frame
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signal.Signals(signum)
        # Cancellation unwinds the trainer's own async cleanup; signal handlers
        # must not reach into Ray or other runtime-specific lifecycle APIs.
        loop.call_soon_threadsafe(task.cancel)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
        try:
            await task
        except asyncio.CancelledError:
            if received_signal is None:
                raise
        return received_signal
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def main(argv: list[str] | None = None) -> None:
    from vrl.config.loading import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    result = run_config(cfg)
    if inspect.isawaitable(result):
        received_signal = asyncio.run(_run_async_trainer(result))
        if received_signal is not None:
            raise SystemExit(128 + int(received_signal))


if __name__ == "__main__":
    main()

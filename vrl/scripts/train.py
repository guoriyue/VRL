"""Unified YAML-driven training entry point.

Execution recipes declare ``trainer.entrypoint``. Independent model, reward,
and dataset presets can be composed at launch with ``+group=option`` arguments.
This module is only the CLI/import layer: it loads the composed config, imports
the entrypoint, then runs it.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import signal
from collections.abc import Awaitable
from types import FrameType
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from vrl.scripts.common.launch_environment import narrow_rank_local_cuda_visibility
from vrl.training_run_result import TrainingRunResultWriter
from vrl.utils.config import import_from_path

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


async def train_online(cfg: DictConfig) -> None:
    """Run the registry-selected online recipe for any policy semantics."""

    from vrl.scripts.common.online import run_online_recipe

    await run_online_recipe(cfg)


def resolve_train_target(root: RootConfig) -> str:
    """Return the validated training callable import path declared by ``root``."""

    import_path = root.trainer.entrypoint if root.trainer is not None else None
    if import_path is None:
        raise ValueError("config missing required field: trainer.entrypoint")
    if not isinstance(import_path, str) or not import_path.strip():
        raise ValueError("trainer.entrypoint must be a non-empty import path")
    return import_path.strip()


def run_config(cfg: DictConfig) -> Any:
    """Run the family trainer selected by ``cfg``."""

    from vrl.config.schema import parse_config

    trainer = import_from_path(resolve_train_target(parse_config(cfg)))
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
        help=(
            "Ordered preset overlays (+reward=ocr +dataset=...) and OmegaConf "
            "dotlist overrides (trainer.seed=42 actor.optim.lr=2e-4)."
        ),
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
    from vrl.config.schema import parse_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    root = parse_config(cfg)
    result = TrainingRunResultWriter.from_root(root)
    try:
        selected_cuda = narrow_rank_local_cuda_visibility(root)
        if selected_cuda is not None:
            logging.getLogger(__name__).info(
                "Rank-local CUDA visibility: LOCAL_RANK=%s physical_device=%s logical_device=0",
                os.environ.get("LOCAL_RANK", "0"),
                selected_cuda,
            )
            # Torch sees one logical cuda:0 after the mask, while Ray reports the
            # original physical ID from ray.get_gpu_ids(). Keep the resource plan
            # in Ray's physical ordinal space so placement probing and worker
            # validation agree — as a loader override, not a post-load edit.
            cfg = load_config(
                args.config,
                overrides=[
                    *args.overrides,
                    f"distributed.resources.visible_devices=[{int(selected_cuda)}]",
                ],
            )
        result = run_config(cfg)
        received_signal: signal.Signals | None = None
        if inspect.isawaitable(result):
            received_signal = asyncio.run(_run_async_trainer(result))
    except BaseException as exc:
        result.write(error=exc)
        raise
    if received_signal is not None:
        result.write(received_signal=received_signal)
        raise SystemExit(128 + int(received_signal))
    result.write()


if __name__ == "__main__":
    main()

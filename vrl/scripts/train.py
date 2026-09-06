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
from collections.abc import Awaitable, MutableMapping
from types import FrameType
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

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


def _narrow_rank_local_cuda_visibility(
    root: RootConfig,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """Give each symmetric-colocated torchrun rank one logical CUDA device.

    Every rank owns a separate local Ray cluster. If all ranks retain the host's
    full CUDA view, Ray may place a rank's single GPU bundle on another rank's
    card. Narrow before importing the trainer so Torch, NCCL, and Ray all agree
    that this rank's physical card is logical ``cuda:0``.
    """

    distributed = root.distributed
    training = None if distributed is None else distributed.training
    resources = None if distributed is None else distributed.resources
    strategy = "single_process" if training is None else str(training.strategy)
    rollout_pool = "auto" if resources is None else str(resources.rollout.gpu_pool)
    if strategy not in {"ddp", "fsdp"} or rollout_pool != "trainer":
        return None

    environment = os.environ if environ is None else environ
    local_rank_raw = environment.get("LOCAL_RANK")
    world_size_raw = environment.get("WORLD_SIZE")
    if local_rank_raw is None or world_size_raw is None:
        # DistributedTrainingContext.from_root owns the complete missing torchrun-env error.
        return None
    try:
        local_rank = int(local_rank_raw)
        world_size = int(world_size_raw)
        configured_local_world_size = 1 if training is None else int(training.gpus_per_node)
        local_world_size = int(
            environment.get("LOCAL_WORLD_SIZE", str(configured_local_world_size)),
        )
    except ValueError as exc:
        raise ValueError(
            "torchrun rank sizes must be integers before rank-local CUDA selection: "
            f"LOCAL_RANK={local_rank_raw!r}, WORLD_SIZE={world_size_raw!r}, "
            f"LOCAL_WORLD_SIZE={environment.get('LOCAL_WORLD_SIZE')!r}",
        ) from exc
    if (
        local_rank < 0
        or local_world_size <= 0
        or world_size < local_world_size
        or local_rank >= local_world_size
    ):
        raise ValueError(
            "invalid torchrun rank identity before rank-local CUDA selection: "
            f"LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}, "
            f"WORLD_SIZE={world_size}",
        )
    if local_world_size != configured_local_world_size:
        raise ValueError(
            "LOCAL_WORLD_SIZE must match distributed.training.gpus_per_node before "
            f"rank-local CUDA selection: LOCAL_WORLD_SIZE={local_world_size}, "
            f"gpus_per_node={configured_local_world_size}",
        )

    if "CUDA_VISIBLE_DEVICES" not in environment:
        selected = str(local_rank)
    else:
        raw_visible = environment["CUDA_VISIBLE_DEVICES"].strip()
        if not raw_visible:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES is empty for a GPU-distributed rank-local launch",
            )
        devices = [token.strip() for token in raw_visible.split(",") if token.strip()]
        if len(set(devices)) != len(devices):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES contains duplicate devices before rank-local "
                f"launch: {raw_visible!r}",
            )
        if len(devices) < local_world_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES cannot supply every local torchrun rank: "
                f"LOCAL_WORLD_SIZE={local_world_size}, "
                f"CUDA_VISIBLE_DEVICES={raw_visible!r}",
            )
        selected = devices[local_rank]

    try:
        physical_device = int(selected)
    except ValueError as exc:
        raise ValueError(
            "symmetric-colocated rank-local Ray placement currently requires integer "
            f"CUDA device ordinals, got {selected!r}",
        ) from exc
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return str(physical_device)


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
    verdict_dir = _verdict_dir(root)
    try:
        selected_cuda = _narrow_rank_local_cuda_visibility(root)
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
        write_run_verdict(verdict_dir, error=exc)
        raise
    if received_signal is not None:
        write_run_verdict(verdict_dir, received_signal=received_signal)
        raise SystemExit(128 + int(received_signal))
    write_run_verdict(verdict_dir)


def _verdict_dir(root: RootConfig) -> str | None:
    output_dir = str((root.trainer.output_dir if root.trainer is not None else None) or "").strip()
    return output_dir or None


RUN_VERDICT_NAME = "run_verdict.json"


def rank_run_verdict_name(rank: int) -> str:
    """Return the per-rank verdict file name owned by one torchrun worker."""

    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    return f"run_verdict.rank-{rank}.json"


def _run_verdict_identity(
    environ: MutableMapping[str, str],
) -> tuple[str, int | None, int | None]:
    """Resolve a collision-free verdict name without masking a trainer failure."""

    rank_raw = environ.get("RANK")
    world_size_raw = environ.get("WORLD_SIZE")
    try:
        rank = int(rank_raw) if rank_raw is not None else None
        world_size = int(world_size_raw) if world_size_raw is not None else None
    except ValueError:
        return RUN_VERDICT_NAME, None, None
    if rank is None or world_size is None or world_size <= 1 or rank < 0 or rank >= world_size:
        return RUN_VERDICT_NAME, None, None
    return rank_run_verdict_name(rank), rank, world_size


def write_run_verdict(
    output_dir: str | None,
    *,
    error: BaseException | None = None,
    received_signal: signal.Signals | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Publish this run's outcome as an explicit machine-readable contract.

    A supervisor must never guess the failure cause from the exit code alone:
    the verdict names the error class so restart policy ("same class twice ->
    stop") is decided on facts. A missing verdict file (SIGKILL, OOM-killed
    interpreter) is itself informative: the run died without unwinding. Every
    multi-rank worker owns a separate file; the supervisor publishes the aggregate
    only after torchrun has joined all workers.
    """

    if output_dir is None:
        return
    import json
    from pathlib import Path

    if error is not None:
        from vrl.runtime_errors import failure_identity_cause

        root_error = failure_identity_cause(error)
        verdict = {
            "verdict": "failed",
            # Cleanup wrappers keep the complete outer message, while restart
            # policy keys on the stable operation root that actually failed.
            "error_class": type(root_error).__name__,
            "error_message": str(error)[:2000],
        }
    elif received_signal is not None:
        verdict = {
            "verdict": "terminated",
            "signal": int(received_signal),
            "signal_name": received_signal.name,
        }
    else:
        verdict = {"verdict": "success"}
    environment = os.environ if environ is None else environ
    file_name, rank, world_size = _run_verdict_identity(environment)
    verdict["schema_version"] = 1
    if rank is not None and world_size is not None:
        verdict["rank"] = rank
        verdict["world_size"] = world_size
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        destination = path / file_name
        temporary = path / f".{file_name}.tmp-{os.getpid()}"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(verdict, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except OSError:
        logging.getLogger(__name__).warning(
            "failed to write run verdict to %s",
            output_dir,
            exc_info=True,
        )


if __name__ == "__main__":
    main()

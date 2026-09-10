"""Isolated native weight-delivery acceptance against a real replay export.

Build the replay source on CPU, optionally restore a training checkpoint, then
start a private Ray fleet. Each receiver is poisoned before two exact installs
through the configured production snapshot or bucket transport.
This checks unsharded in-place parameter delivery, not generation quality or
converted/quantized forward equivalence. It never attaches to a live fleet.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import math
import time
from pathlib import Path
from typing import Any

import torch

from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.weight_sync import RayGenerationWeightSync
from vrl.generation.ray.worker import HEALTH_CONCURRENCY_GROUP, RayGenerationWorker
from vrl.ray.actor_group import RayActorGroup
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.ray.dependencies import require_ray
from vrl.run import resolve_model, resolve_online_run
from vrl.trainers.weight_sync import build_trainable_state_sync_getter, to_cpu_snapshot


class WeightDeliveryProbeWorker(RayGenerationWorker):
    """Poisoning exists only on isolated acceptance actors, never production actors."""

    def poison_parameters(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        poison = {
            name: torch.where(value == 0, torch.ones_like(value), torch.zeros_like(value))
            for name, value in snapshot.items()
        }
        self.update_weights(poison, 0, verify_content=True)
        return {
            "worker_id": self.core.worker_id,
            "tensors": len(snapshot),
            "bytes": sum(value.numel() * value.element_size() for value in snapshot.values()),
            "poison_install_verify_s": time.perf_counter() - started,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> None:
    from vrl.config.loading import load_config
    from vrl.trainers.checkpointing import load_training_checkpoint, restore_model_checkpoint

    args = build_parser().parse_args(argv)
    if args.workers < 1 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("workers and timeout must be positive")
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {args.report}")
    cfg = load_config(args.config, overrides=args.overrides)
    resolved = resolve_online_run(cfg)
    if resolved.resources.rollout_gpus_per_engine != 1:
        raise NotImplementedError("this acceptance entrypoint requires single-rank engines")
    replay = resolve_model(
        resolved.family,
        resolved.built.root,
        torch.device("cpu"),
        precision=resolved.built.precision,
        for_rollout=False,
    )
    inputs = resolved.ray_launch_inputs(replay)
    if inputs.launch_contract.versioned_weight_sync:
        raise NotImplementedError("retained-slot acceptance requires a request/activation probe")
    ray = require_ray()
    if ray.is_initialized():
        raise RuntimeError(
            "weight acceptance must run in a fresh process with a private Ray cluster"
        )
    num_gpus = int(bool(resolved.resources.rollout_devices))
    if num_gpus and torch.cuda.device_count() < args.workers:
        raise ValueError("one visible GPU per acceptance worker is required")

    resolved.run.initialize_process_rng()
    source_started = time.perf_counter()
    bundle = replay.materialize(context="weight delivery acceptance source")
    if args.checkpoint is not None:
        checkpoint = load_training_checkpoint(args.checkpoint)
        restore_model_checkpoint(
            checkpoint,
            bundle=bundle,
            family=resolved.family.family,
            expected_model_identity=replay.identity,
            strict=True,
        )
        del checkpoint
    source_ready = time.perf_counter()
    snapshot = to_cpu_snapshot(build_trainable_state_sync_getter(bundle)())
    snapshot_ready = time.perf_counter()
    del bundle
    gc.collect()

    group = None
    report = None
    try:
        ray.init(address="local", include_dashboard=False)
        group = RayActorGroup.launch(
            worker_cls=WeightDeliveryProbeWorker,
            worker_configs=[inputs] * args.workers,
            worker_ids=[f"acceptance-{index}" for index in range(args.workers)],
            num_cpus=1,
            num_gpus=num_gpus,
            rpc_timeout_s=args.timeout_s,
            operation_prefix="acceptance",
            startup_method="load_policy",
            concurrency_groups={HEALTH_CONCURRENCY_GROUP: 1},
        )
        payload = ray.put(snapshot)
        receivers = ray.get(
            [handle.actor.poison_parameters.remote(payload) for handle in group.handles],
            timeout=args.timeout_s,
        )
        del payload
        engines = [RayGenerationEngine(handle.worker_id, [handle]) for handle in group.handles]
        bucket_bytes = resolved.generation.worker.weight_sync_bucket_bytes
        sync = RayGenerationWeightSync(
            engines,
            actor_dispatcher=RayActorDispatcher(tuple(engine.engine_id for engine in engines)),
            worker_rpc_timeout_s=args.timeout_s,
            verify_content=True,
            bucket_bytes=bucket_bytes,
        )

        async def install_snapshots() -> dict[str, float]:
            timings = {}
            for version, phase in ((1, "first"), (2, "repeat")):
                started = time.perf_counter()
                await sync.push_to_rollout_engines(snapshot, version)
                timings[phase] = time.perf_counter() - started
            return timings

        sync_timings = asyncio.run(install_snapshots())
        # Each successful sync required verified content and the version ACK
        # from every receiver. A partial/failing fleet never reaches publication.
        for receiver in receivers:
            receiver["policy_version"] = 2
        report = {
            "schema": "vrl.weight-delivery-acceptance/v2",
            "scope": "single-rank in-place trainable parameter bytes; no forward equivalence claim",
            "model_identity": replay.identity,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "source_initialization": {
                "seed": resolved.run.seed,
                "deterministic": resolved.run.deterministic,
            },
            "source_build_restore_s": source_ready - source_started,
            "snapshot_export_s": snapshot_ready - source_ready,
            "transport": {
                "kind": "staged_buckets" if bucket_bytes is not None else "snapshot",
                "bucket_bytes": bucket_bytes,
                "sync_verify_wall_s": sync_timings,
            },
            "receivers": receivers,
        }
    finally:
        try:
            if group is not None:
                group.shutdown()
        finally:
            ray.shutdown()
    # Publication follows successful fleet cleanup. A failed rank leaves no
    # completion-looking report, and an existing result is never replaced.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    from vrl.trainers.evidence import publish_evidence_record

    publish_evidence_record(args.report, report)


if __name__ == "__main__":
    main()

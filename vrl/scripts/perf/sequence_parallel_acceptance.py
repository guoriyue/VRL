"""Hardware acceptance for multi-rank (Ulysses sequence-parallel) rollout engines.

The CPU+gloo unit test pins the numerics of the sequence-parallel install on a
tiny transformer. This script is the physical twin on real GPUs: it launches the
PRODUCTION Ray fleet (placement owner -> launcher -> runtime) from an experiment
config with ``distributed.resources.rollout.gpus_per_engine=N``, runs one
fixed-seed deterministic request, and dumps the decoded output tensor together
with the per-rank wall/peak-memory telemetry. Run it once with N=1 and once
with N=2 on the same prompts and seed, then ``compare`` the two dumps.

The request forces ``rollout.denoise_mode=native``: the SDE branch draws its
per-step noise from the process RNG, which a multi-rank engine reseeds from
rank 0's entropy per request, so only the deterministic sampler can be compared
across fleet shapes. ``sampling.seed`` fixes the initial latents.

Usage (one GPU per rank; the trainer device is reserved but never built):

    CUDA_VISIBLE_DEVICES=0,1,2 python -m vrl.scripts.perf.sequence_parallel_acceptance \\
        generate --config experiment/sd3_5/online_grpo_ocr \\
        --gpus-per-engine 1 --rollout-devices 1 --seed 7 \\
        --output-dir outputs/sp_acceptance/n1
    CUDA_VISIBLE_DEVICES=0,1,2 python -m vrl.scripts.perf.sequence_parallel_acceptance \\
        generate --config experiment/sd3_5/online_grpo_ocr \\
        --gpus-per-engine 2 --rollout-devices 1 2 --seed 7 \\
        --output-dir outputs/sp_acceptance/n2
    python -m vrl.scripts.perf.sequence_parallel_acceptance \\
        compare outputs/sp_acceptance/n1 outputs/sp_acceptance/n2 --atol 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.families.semantics import task_type_for
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import GlobalRayPlacementOwner
from vrl.run import resolve_model, resolve_online_run
from vrl.utils.json_files import write_json

logger = logging.getLogger(__name__)

# Test-fixture prompts: short, distinct, and family-neutral. The comparison is
# between fleet shapes on identical inputs, so their content is irrelevant.
DEFAULT_PROMPTS = (
    "a red bicycle leaning against a white brick wall",
    "a wooden table with a single green apple",
    "a lighthouse on a rocky shore at dusk",
    "a stack of old books beside a brass lamp",
)

OUTPUT_TENSOR_FILE = "output.pt"
REPORT_FILE = "report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="launch one fleet shape and dump its output")
    generate.add_argument("--config", required=True)
    generate.add_argument("--gpus-per-engine", type=int, required=True)
    generate.add_argument(
        "--rollout-devices",
        type=int,
        nargs="+",
        required=True,
        help="visible GPU indices for the rollout fleet (count = engines x gpus_per_engine)",
    )
    generate.add_argument(
        "--trainer-device",
        type=int,
        default=0,
        help="visible GPU index reserved for the (never built) trainer role",
    )
    generate.add_argument("--seed", type=int, default=20260911)
    generate.add_argument("--prompts", type=Path, help="one prompt per line; default fixture")
    generate.add_argument("--num-prompts", type=int, default=len(DEFAULT_PROMPTS))
    generate.add_argument("--samples-per-prompt", type=int, default=1)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--resolve-only",
        action="store_true",
        help="resolve config and print the fleet plan without launching Ray",
    )
    generate.add_argument("overrides", nargs="*", help="extra dotlist overrides")

    compare = sub.add_parser("compare", help="compare two generate dumps")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--atol",
        type=float,
        default=0.02,
        help="max abs difference allowed per element, on a [0, 1] scale",
    )
    compare.add_argument(
        "--max-mismatch-fraction",
        type=float,
        default=0.0,
        help="fraction of elements allowed beyond --atol",
    )
    compare.add_argument(
        "--min-psnr-db",
        type=float,
        default=0.0,
        help="lowest per-sample PSNR (dB, [0, 1] scale) accepted; set it from the "
        "single-rank self-repeat floor measured on the same host",
    )
    return parser


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts is None:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.num_prompts < 1 or args.num_prompts > len(prompts):
        raise ValueError(
            f"--num-prompts must be in [1, {len(prompts)}] for the given prompt source",
        )
    return prompts[: args.num_prompts]


def _fleet_overrides(args: argparse.Namespace) -> list[str]:
    devices = ",".join(str(d) for d in args.rollout_devices)
    return [
        f"distributed.resources.trainer.devices=[{args.trainer_device}]",
        f"distributed.resources.rollout.devices=[{devices}]",
        f"distributed.resources.rollout.gpus_per_engine={args.gpus_per_engine}",
        # Sequence parallelism mutates the policy core after build; compile is
        # rejected with gpus_per_engine > 1, and both shapes must run eager so
        # the comparison isolates the fleet topology.
        "model.torch_compile.enable=false",
        f"sampling.seed={args.seed}",
        *args.overrides,
    ]


async def _generate(args: argparse.Namespace) -> dict[str, Any]:
    from vrl.config.loading import load_config

    cfg = load_config(args.config, overrides=_fleet_overrides(args))
    resolved = resolve_online_run(cfg)
    resources = resolved.resources
    if resources.rollout_gpus_per_engine != args.gpus_per_engine:
        raise RuntimeError(
            f"resolved gpus_per_engine={resources.rollout_gpus_per_engine} != "
            f"requested {args.gpus_per_engine}",
        )
    if resources.colocated:
        raise ValueError(
            "sequence-parallel acceptance needs dedicated rollout GPUs; the resolved "
            "topology shares the trainer device",
        )
    prompts = _load_prompts(args)
    plan = {
        "config": args.config,
        "overrides": _fleet_overrides(args),
        "family": resolved.family.family,
        "gpus_per_engine": resources.rollout_gpus_per_engine,
        "num_engines": resources.rollout_num_engines,
        "rollout_devices": list(resources.rollout_devices),
        "rollout_mode": resources.lifecycle.rollout_mode,
        "seed": args.seed,
        "prompts": prompts,
        "samples_per_prompt": args.samples_per_prompt,
    }
    logger.info("fleet plan: %s", json.dumps(plan, indent=2))
    if args.resolve_only:
        return plan

    collector = resolved.collector
    denoise = replace(collector.denoise or DenoiseRequestOptions(), denoise_mode="native")
    sampling = dict(collector.request_sampling)
    if sampling.get("seed") != args.seed:
        raise RuntimeError(f"sampling.seed override did not reach the request: {sampling!r}")
    task_type = task_type_for(resolved.family.task)
    request = GenerationRequest(
        request_id=f"sp-acceptance-{uuid.uuid4()}",
        family=resolved.family.family,
        task=resolved.family.task,
        inputs=[GenerationInput(prompt=p, task_type=task_type) for p in prompts],
        samples_per_prompt=args.samples_per_prompt,
        sampling=sampling,
        samples_per_generation_batch=collector.samples_per_generation_batch,
        trajectory_storage=collector.trajectory_storage,
        denoise=denoise,
        runtime_debug=True,
        policy_version=0,
    )

    replay = resolve_model(
        resolved.family,
        resolved.built.root,
        torch.device("cpu"),
        precision=resolved.built.precision,
        for_rollout=False,
    )
    launch_inputs = resolved.ray_launch_inputs(replay)

    ray = require_ray()
    if ray.is_initialized():
        raise RuntimeError("acceptance must run in a fresh process with a private Ray cluster")
    placement_owner = GlobalRayPlacementOwner(resources, resolved.generation.worker)
    ray.init(
        address="local",
        include_dashboard=False,
        num_cpus=placement_owner.required_local_cluster_cpus(),
    )
    runtime = None
    report: dict[str, Any] = dict(plan)
    try:
        placement_owner.create()
        launch_started = time.perf_counter()
        runtime = RayGenerationLauncher().create_runtime(
            resolved.generation,
            launch_inputs,
            placement=placement_owner.rollout_placement,
        )
        await runtime.preflight()
        if resources.lifecycle.rollout_mode == "on_demand":
            await runtime.activate()
        report["launch_s"] = time.perf_counter() - launch_started

        generate_started = time.perf_counter()
        output = await runtime.generate(request)
        report["generate_wall_s"] = time.perf_counter() - generate_started

        tensor = output.output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"GenerationOutput.output must be a tensor, got {type(tensor)}")
        tensor = tensor.detach().cpu()
        report["output_shape"] = list(tensor.shape)
        report["output_dtype"] = str(tensor.dtype)
        report["runtime_debug"] = output.runtime_debug
        report["peak_memory_mb_by_rank"] = _peak_memory_by_rank(output.runtime_debug)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "output": tensor,
                "prompts": prompts,
                "seed": args.seed,
                "gpus_per_engine": args.gpus_per_engine,
            },
            args.output_dir / OUTPUT_TENSOR_FILE,
        )
        write_json(args.output_dir / REPORT_FILE, report)
        logger.info("acceptance dump written to %s", args.output_dir)
        return report
    finally:
        if runtime is not None:
            await runtime.shutdown()
        placement_owner.shutdown()
        ray.shutdown()


def _peak_memory_by_rank(runtime_debug: dict[str, Any] | None) -> dict[str, float]:
    """Max ``peak_memory_mb`` per rank over the request's batches.

    The P6 gate asks for peak memory per rank, and a multi-rank engine reports
    one row per rank in ``ray_chunks`` (``GenerationBatchResult.rank_metrics``).
    """

    peaks: dict[str, float] = {}
    for row in (runtime_debug or {}).get("ray_chunks", []):
        value = row.get("peak_memory_mb")
        if value is None:
            continue
        worker_id = str(row["worker_id"])
        peaks[worker_id] = max(peaks.get(worker_id, 0.0), float(value))
    return peaks


def _as_unit_scale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    reference = torch.load(args.reference / OUTPUT_TENSOR_FILE, weights_only=True)
    candidate = torch.load(args.candidate / OUTPUT_TENSOR_FILE, weights_only=True)
    for key in ("prompts", "seed"):
        if reference[key] != candidate[key]:
            raise ValueError(f"dumps differ in {key}: {reference[key]!r} != {candidate[key]!r}")
    a = _as_unit_scale(reference["output"])
    b = _as_unit_scale(candidate["output"])
    if a.shape != b.shape:
        raise ValueError(f"output shapes differ: {tuple(a.shape)} != {tuple(b.shape)}")
    for name, tensor in (("reference", a), ("candidate", b)):
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} output must be nonempty and finite")
    diff = (a - b).abs()
    mismatch_fraction = float((diff > args.atol).float().mean())
    # Per-sample PSNR on the [0, 1] scale: the number to read against the
    # single-rank self-repeat floor (bf16 kernel noise amplified over the
    # denoise schedule), which an absolute pixel tolerance cannot express.
    per_sample_mse = ((a - b) ** 2).flatten(1).mean(dim=1)
    psnr_db = [
        float("inf") if mse == 0 else float(10 * torch.log10(1 / mse)) for mse in per_sample_mse
    ]
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_gpus_per_engine": reference["gpus_per_engine"],
        "candidate_gpus_per_engine": candidate["gpus_per_engine"],
        "shape": list(a.shape),
        "atol": args.atol,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "mismatch_fraction": mismatch_fraction,
        "psnr_db": psnr_db,
        "passed": mismatch_fraction <= args.max_mismatch_fraction
        and min(psnr_db) >= args.min_psnr_db,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        asyncio.run(_generate(args))
        return
    result = _compare(args)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

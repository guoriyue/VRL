"""Profile quantized SD3.5 transformer forwards with real checkpoint weights.

This is the model-level complement to ``quantized_linear_benchmark``. It loads
the real SD3.5 Medium transformer, installs VRL's production attention
processor, and compares BF16, default rowwise FP8, and MLP-only NVFP4 at the
current 512px recipe shape (CLIP 77 + T5 128 text tokens). Effective batches 2
and 32 represent one and sixteen samples after batched CFG respectively.

``--target-profile`` can instead force FP8 and NVFP4 onto the identical MLP-only
or attention+MLP path/shape manifest. This is an experimental comparison seam;
it does not change either scheme's production targeting default.

The profile reports CUDA-event latency distributions, exact unique tensor
storage with quantized masters kept/dropped, allocator high-water marks, and
noise-prediction drift against BF16. It deliberately excludes prompt encoding,
scheduler/SDE math, VAE decode, Ray, reward, and training replay, so the result
is a real-checkpoint transformer-forward profile rather than an end-to-end
rollout claim.

Usage:
    python -m vrl.scripts.perf.quantized_sd3_forward_profile
    TORCHINDUCTOR_COMPILE_THREADS=1 python -m \
        vrl.scripts.perf.quantized_sd3_forward_profile --compile
    TORCHINDUCTOR_COMPILE_THREADS=1 python -m \
        vrl.scripts.perf.quantized_sd3_forward_profile --compile \
        --target-profile attention_mlp --schemes bf16 fp8 nvfp4
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from vrl.models.families.sd3_5.model import install_sd3_joint_attention_processor
from vrl.nn.quantization import (
    LinearTargetProfile,
    QuantizedLinear,
    drop_quantized_masters,
    nvfp4_available,
    swap_linears_to_fp8,
    swap_linears_to_nvfp4,
)
from vrl.scripts.perf.common.fp8_math import relative_l1_drift


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 32])
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=("bf16", "fp8", "nvfp4"),
        default=None,
    )
    parser.add_argument(
        "--target-profile",
        choices=("scheme_default", *(profile.value for profile in LinearTargetProfile)),
        default="scheme_default",
        help=(
            "Linear scope shared by every quantized scheme. scheme_default preserves "
            "production FP8=attention_mlp and NVFP4=mlp_only; an explicit profile makes "
            "the runner fail unless FP8 and NVFP4 swap the exact same path/shape manifest."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warm forwards per shape; SD3 needs a long warmup for stable clocks/kernels.",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--keep-masters",
        action="store_true",
        help="Keep train-dtype masters (full-parameter weight-sync ownership).",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads; the default requires a local checkpoint.",
    )
    args = parser.parse_args()
    args.schemes = args.schemes or ["bf16", "fp8", "nvfp4"]
    if args.schemes[0] != "bf16":
        raise SystemExit("--schemes must start with bf16 so quantized drift has a reference")
    if any(batch < 1 for batch in args.batches):
        raise SystemExit("--batches values must be >= 1")
    return args


def _load_transformer(model_name: str, *, allow_download: bool) -> torch.nn.Module:
    from diffusers import SD3Transformer2DModel

    model = SD3Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=not allow_download,
    ).to("cuda")
    model.requires_grad_(False).eval()
    if not install_sd3_joint_attention_processor(model):
        raise RuntimeError("SD3 transformer did not accept VRL's joint attention processor")
    return model


def _make_inputs(batch: int) -> dict[str, torch.Tensor | bool]:
    generator = torch.Generator(device="cuda").manual_seed(20260711 + batch)
    return {
        "hidden_states": torch.randn(
            batch,
            16,
            64,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "encoder_hidden_states": torch.randn(
            batch,
            205,
            4096,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "pooled_projections": torch.randn(
            batch,
            2048,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "timestep": torch.full((batch,), 500.0, device="cuda", dtype=torch.float32),
        "return_dict": False,
    }


def _forward_output(model: Callable[..., Any], inputs: dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        output = model(**inputs)
    return output[0] if isinstance(output, tuple) else output.sample


def _latency_and_memory(
    step: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(max(0, warmup)):
            step()
        torch.cuda.synchronize()
        steady = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        samples: list[float] = []
        for _ in range(max(1, iters)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    finally:
        if gc_was_enabled:
            gc.enable()
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]

    peak = torch.cuda.max_memory_allocated()
    return {
        "median_ms": percentile(0.5),
        "p10_ms": percentile(0.1),
        "p90_ms": percentile(0.9),
        "steady_allocated_gib": steady / 2**30,
        "peak_allocated_gib": peak / 2**30,
        "workspace_mib": (peak - steady) / 2**20,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }


def _unique_tensor_bytes(module: torch.nn.Module) -> int:
    storages: dict[tuple[str, int | None, int], int] = {}
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor is None or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())


def _accuracy(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = output.float()
    ref = reference.float()
    diff = out - ref
    cosine = float(F.cosine_similarity(out.flatten(), ref.flatten(), dim=0))
    return {
        "normalized_l1": relative_l1_drift(out, ref),
        "nrmse": float(diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)),
        "cosine": min(1.0, max(-1.0, cosine)),
        "abs_p99": float(diff.abs().quantile(0.99)),
        "abs_max": float(diff.abs().max()),
    }


def _resolve_target_profile(
    scheme: str,
    requested: str,
) -> LinearTargetProfile | None:
    if scheme == "bf16":
        return None
    if requested != "scheme_default":
        return LinearTargetProfile(requested)
    if scheme == "fp8":
        return LinearTargetProfile.ATTENTION_MLP
    if scheme == "nvfp4":
        return LinearTargetProfile.MLP_ONLY
    raise ValueError(f"unsupported profile scheme {scheme!r}")


def _apply_scheme(
    model: torch.nn.Module,
    scheme: str,
    target_profile: LinearTargetProfile | None,
) -> list[str]:
    if scheme == "bf16":
        return []
    if target_profile is None:
        raise ValueError(f"quantized scheme {scheme!r} requires a target profile")
    if scheme == "fp8":
        return swap_linears_to_fp8(model, target_profile=target_profile)
    if scheme == "nvfp4":
        return swap_linears_to_nvfp4(model, target_profile=target_profile)
    raise ValueError(f"unsupported quantized scheme {scheme!r}")


def _target_manifest(
    model: torch.nn.Module,
    paths: list[str],
) -> tuple[tuple[str, int, int], ...]:
    """Build an ordered path/shape manifest after a quantized module swap."""

    manifest: list[tuple[str, int, int]] = []
    for path in paths:
        module = model.get_submodule(path)
        if not isinstance(module, QuantizedLinear):
            raise RuntimeError(f"target path {path!r} is not a QuantizedLinear after swap")
        manifest.append((path, module.in_features, module.out_features))
    return tuple(sorted(manifest))


def _manifest_sha256(manifest: tuple[tuple[str, int, int], ...]) -> str:
    payload = "\n".join(
        f"{path}\t{in_features}\t{out_features}" for path, in_features, out_features in manifest
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_matching_manifest(
    reference_scheme: str,
    reference: tuple[tuple[str, int, int], ...],
    scheme: str,
    actual: tuple[tuple[str, int, int], ...],
) -> None:
    """Fail before timing when two schemes did not quantize identical Linears."""

    if actual == reference:
        return
    reference_set = set(reference)
    actual_set = set(actual)
    only_reference = sorted(reference_set - actual_set)
    only_actual = sorted(actual_set - reference_set)
    raise RuntimeError(
        f"matched target profile diverged: {reference_scheme}={len(reference)} "
        f"{scheme}={len(actual)}; only_{reference_scheme}={only_reference[:8]!r}; "
        f"only_{scheme}={only_actual[:8]!r}",
    )


def _profile_scheme(
    args: argparse.Namespace,
    scheme: str,
    references: dict[int, torch.Tensor],
    *,
    expected_manifest: tuple[tuple[str, int, int], ...] | None = None,
    expected_scheme: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[int, torch.Tensor],
    tuple[tuple[str, int, int], ...],
]:
    before_load = torch.cuda.memory_allocated()
    model = _load_transformer(args.model, allow_download=args.allow_download)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    target_profile = _resolve_target_profile(scheme, args.target_profile)
    swapped = _apply_scheme(model, scheme, target_profile)
    manifest = _target_manifest(model, swapped)
    if expected_manifest is not None:
        if expected_scheme is None:
            raise ValueError("expected_scheme is required with expected_manifest")
        _require_matching_manifest(expected_scheme, expected_manifest, scheme, manifest)
    quantized = [module for module in model.modules() if isinstance(module, QuantizedLinear)]
    if len(quantized) != len(manifest):
        raise RuntimeError(
            f"swap returned {len(manifest)} paths but model contains "
            f"{len(quantized)} QuantizedLinear modules",
        )
    swapped_weights = sum(in_features * out_features for _, in_features, out_features in manifest)
    kept_storage = _unique_tensor_bytes(model)
    master_bytes = sum(
        module.weight.numel() * module.weight.element_size()
        for module in quantized
        if module.weight is not None
    )
    projected_dropped_storage = kept_storage - master_bytes
    freed = 0
    if scheme != "bf16" and not args.keep_masters:
        freed = drop_quantized_masters(model)
        if freed != master_bytes:
            raise RuntimeError(f"master byte mismatch: counted={master_bytes}, dropped={freed}")
    active_storage = _unique_tensor_bytes(model)
    runner: Any = model
    if args.compile:
        runner = torch.compile(model, mode=args.compile_mode)

    rows: list[dict[str, Any]] = []
    produced_references: dict[int, torch.Tensor] = {}
    for batch in args.batches:
        inputs = _make_inputs(batch)
        cold_start_s = 0.0
        if args.compile:
            start = time.perf_counter()
            output = _forward_output(runner, inputs)
            torch.cuda.synchronize()
            cold_start_s = time.perf_counter() - start
        else:
            output = _forward_output(runner, inputs)
            torch.cuda.synchronize()
        output_cpu = output.float().cpu()
        del output
        row: dict[str, Any] = {"batch": batch, "cold_start_s": cold_start_s}
        if scheme == "bf16":
            produced_references[batch] = output_cpu
            row["accuracy"] = None
        else:
            row["accuracy"] = _accuracy(output_cpu, references[batch])
        row.update(
            _latency_and_memory(
                lambda model=runner, kwargs=inputs: _forward_output(model, kwargs),
                warmup=args.warmup,
                iters=args.iters,
            ),
        )
        rows.append(row)
        del inputs, output_cpu
        gc.collect()

    result = {
        "scheme": scheme,
        "profile": target_profile.value if target_profile is not None else "bf16",
        "recipe": ("rowwise" if scheme == "fp8" else None if scheme == "nvfp4" else "bf16"),
        "swapped_linears": len(swapped),
        "swapped_weights": swapped_weights,
        "target_manifest_sha256": _manifest_sha256(manifest) if manifest else None,
        "total_parameters": total_parameters,
        "coverage_percent": 100.0 * swapped_weights / total_parameters,
        "master_kept_storage_gib": kept_storage / 2**30,
        "master_dropped_storage_gib": projected_dropped_storage / 2**30,
        "active_storage_gib": active_storage / 2**30,
        "masters_dropped": bool(freed),
        "rows": rows,
    }
    del quantized, runner, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    remaining = torch.cuda.memory_allocated() - before_load
    if remaining > 64 * 2**20:
        print(f"WARNING: {scheme} cleanup left {remaining / 2**20:.1f} MiB allocated")
    return result, produced_references, manifest


def _add_relative_speedups(results: list[dict[str, Any]]) -> None:
    latency_by_scheme = {
        result["scheme"]: {row["batch"]: row["median_ms"] for row in result["rows"]}
        for result in results
    }
    bf16_latency = latency_by_scheme["bf16"]
    for result in results:
        for row in result["rows"]:
            row["speedup_vs_bf16"] = bf16_latency[row["batch"]] / row["median_ms"]
    fp8_result = next((result for result in results if result["scheme"] == "fp8"), None)
    nvfp4_result = next((result for result in results if result["scheme"] == "nvfp4"), None)
    if fp8_result is None or nvfp4_result is None:
        return
    if fp8_result.get("profile") != nvfp4_result.get("profile") or fp8_result.get(
        "target_manifest_sha256"
    ) != nvfp4_result.get("target_manifest_sha256"):
        return
    fp8_latency = latency_by_scheme["fp8"]
    nvfp4_latency = latency_by_scheme["nvfp4"]
    for result in results:
        if result["scheme"] == "nvfp4":
            for row in result["rows"]:
                row["speedup_vs_fp8"] = fp8_latency[row["batch"]] / row["median_ms"]
        elif result["scheme"] == "fp8":
            for row in result["rows"]:
                row["speedup_vs_nvfp4"] = nvfp4_latency[row["batch"]] / row["median_ms"]


def _print_report(results: list[dict[str, Any]]) -> None:
    _add_relative_speedups(results)
    print("\n== SD3.5 real-checkpoint transformer forward ==")
    for result in results:
        print(
            f"{result['scheme']}: recipe={result['recipe']} profile={result['profile']} "
            f"swapped={result['swapped_linears']} weights={result['swapped_weights']:,} "
            f"coverage={result['coverage_percent']:.2f}% storage(active/kept/dropped)="
            f"{result['active_storage_gib']:.3f}/{result['master_kept_storage_gib']:.3f}/"
            f"{result['master_dropped_storage_gib']:.3f} GiB "
            f"manifest={result['target_manifest_sha256']}"
        )
        for row in result["rows"]:
            speedup = row["speedup_vs_bf16"]
            accuracy = row["accuracy"]
            accuracy_text = "reference"
            if accuracy is not None:
                accuracy_text = (
                    f"nL1={accuracy['normalized_l1']:.5f} "
                    f"NRMSE={accuracy['nrmse']:.5f} cosine={accuracy['cosine']:.6f} "
                    f"abs_p99={accuracy['abs_p99']:.5f} abs_max={accuracy['abs_max']:.5f}"
                )
            print(
                f"  B={row['batch']:>2} median={row['median_ms']:.3f}ms "
                f"p10/p90={row['p10_ms']:.3f}/{row['p90_ms']:.3f}ms speedup={speedup:.3f}x "
                f"allocated steady/peak={row['steady_allocated_gib']:.3f}/"
                f"{row['peak_allocated_gib']:.3f}GiB workspace={row['workspace_mib']:.1f}MiB "
                f"reserved={row['reserved_gib']:.3f}GiB cold={row['cold_start_s']:.2f}s "
                f"{accuracy_text}"
            )
    nvfp4_result = next((result for result in results if result["scheme"] == "nvfp4"), None)
    if nvfp4_result is not None and all("speedup_vs_fp8" in row for row in nvfp4_result["rows"]):
        for row in nvfp4_result["rows"]:
            print(
                f"  MATCHED B={row['batch']:>2}: nvfp4_vs_fp8="
                f"{row['speedup_vs_fp8']:.3f}x "
                "(>1 means NVFP4 is faster)"
            )
    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this profile requires CUDA")
    if "nvfp4" in args.schemes and not nvfp4_available():
        raise SystemExit(
            "nvfp4 profiling requires an NVFP4-capable CUDA build and Blackwell-class GPU",
        )
    torch.manual_seed(0)
    print(
        f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__} cuda={torch.version.cuda} compile={args.compile} "
        f"keep_masters={args.keep_masters} target_profile={args.target_profile} "
        f"batches={args.batches} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    references: dict[int, torch.Tensor] = {}
    results: list[dict[str, Any]] = []
    matched_manifest: tuple[tuple[str, int, int], ...] | None = None
    matched_scheme: str | None = None
    for scheme in args.schemes:
        print(f"loading/profile scheme={scheme} ...", flush=True)
        require_match = args.target_profile != "scheme_default" and scheme != "bf16"
        result, produced, manifest = _profile_scheme(
            args,
            scheme,
            references,
            expected_manifest=matched_manifest if require_match else None,
            expected_scheme=matched_scheme if require_match else None,
        )
        if require_match and matched_manifest is None:
            matched_manifest = manifest
            matched_scheme = scheme
        references.update(produced)
        results.append(result)
    _print_report(results)


if __name__ == "__main__":
    main()

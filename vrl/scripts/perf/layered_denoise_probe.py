"""Where one Qwen-Image-Layered denoise step spends the card, and how far from the floor.

The generic ``generation_bottleneck_profile`` builds a text-to-image request;
the layered variant needs a source image's condition tokens, so this probe
takes one cached scene (``cache.pt`` from ``encode_layered_scenes``), replicates
it to a rollout batch, and profiles the production ``forward_step`` under the
experiment's own rollout build (host-streamed frozen blocks included). It
reports three things the step time alone cannot:

1. the kernel split (GEMM / attention / elementwise / copies) of one step and
   the achieved GEMM throughput against a measured cuBLAS peak;
2. the host-streaming cost: the same block stack run with its weights resident
   on the card gives the compute floor of the stack, and the gap to the
   measured step is what streaming, prefetch misses and the stem/head cost;
3. the prefetch hit/miss counters of the streamed policy.

Usage (inside a checkout that carries the layered experiment):

    python -m vrl.scripts.perf.layered_denoise_probe \\
        --cache outputs/layered_grpo/eval_scenes/luma_cat/test/scene_r00/cache.pt \\
        --batch 4 --layers 9 [--compile] [override ...]
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _median_ms(fn, *, iters: int) -> float:
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _cublas_peak_tflops(device: torch.device, dtype: torch.dtype) -> float:
    """Achievable dense GEMM throughput on this card at a model-like shape."""

    a = torch.randn(32768, 3072, device=device, dtype=dtype)
    w = torch.randn(12288, 3072, device=device, dtype=dtype)
    ms = _median_ms(lambda: a @ w.T, iters=10)
    return 2.0 * a.shape[0] * a.shape[1] * w.shape[0] / ms / 1e9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="qwen_image_layered/online_grpo_layered_grouping")
    parser.add_argument("--cache", type=Path, required=True, help="one encoded scene's cache.pt")
    parser.add_argument("--batch", type=int, default=4, help="samples per generation batch")
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--steps", type=int, default=3, help="profiled denoise steps")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--block-iters", type=int, default=5, help="resident-block timing repeats")
    parser.add_argument(
        "--compile", action="store_true", help="regional torch.compile of the blocks"
    )
    parser.add_argument("--trace", type=Path, default=None, help="chrome trace output")
    parser.add_argument(
        "--dump-transformer-inputs",
        type=Path,
        default=None,
        help="save the first transformer call's args and kwargs (CPU tensors) for group_offload_compare",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("overrides", nargs="*", help="dotlist config overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.generation.types import DenoiseRequest
    from vrl.models.families.registry import get_model_family_entry
    from vrl.nn.offload import HostStreaming
    from vrl.nn.optimization.regional_compile import repeated_block_lists
    from vrl.scripts.perf.common.diffusion_runtime import denoise_step
    from vrl.scripts.perf.generation_bottleneck_profile import _bucket
    from vrl.trainers.data.layered_scenes import load_layered_scene

    device = torch.device(args.device)
    overrides = list(args.overrides)
    if args.compile:
        overrides += ["model.torch_compile.enable=true", "model.torch_compile.regional=true"]
    cfg = load_config(f"experiment/{args.experiment}", overrides=overrides)
    root = parse_config(cfg)
    entry = get_model_family_entry(str(root.model.family))
    build = entry.resolve_model_build(
        root, device, precision=PrecisionPolicy.from_section(root.precision)
    )
    model = entry.build_rollout(build, expected_identity=None).model.eval()
    transformer = model.transformer
    streaming = HostStreaming.of(transformer)
    streamed = streaming is not None
    print(
        f"policy built: streamed={streamed} compile={args.compile} "
        f"dtype={next(transformer.parameters()).dtype}",
        flush=True,
    )

    revision = None if root.model.revision is None else str(root.model.revision)
    cache = load_layered_scene(args.cache, family=entry.family, model_revision=revision)
    batch = int(args.batch)

    def replicate(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(device).expand(batch, *tensor.shape[1:]).contiguous()

    encoded = {
        "prompt_embeds": replicate(cache["prompt_embeds"]),
        "prompt_embeds_mask": replicate(cache["prompt_embeds_mask"]),
        "condition_latents": replicate(cache["condition_latents"]),
    }
    request = DenoiseRequest(
        width=int(cache["width"]),
        height=int(cache["height"]),
        frame_count=1,
        num_steps=int(root.sampling.num_steps),
        guidance_scale=1.0,
        seed=0,
    )
    with torch.no_grad():
        state = model.prepare_sampling(request, encoded, layers=int(args.layers))
    image_tokens = int(state.latents.shape[1]) + int(cache["condition_latents"].shape[1])
    text_tokens = int(cache["prompt_embeds"].shape[1])
    print(
        f"shape: batch {batch}, {image_tokens} image tokens (+{text_tokens} text), "
        f"{int(root.sampling.num_steps)} steps",
        flush=True,
    )

    # -- GEMM FLOPs of one step, counted at the Linear boundary -----------
    linear_flops = 0.0

    def count_linear(module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        nonlocal linear_flops
        x = inputs[0]
        linear_flops += 2.0 * x.numel() / x.shape[-1] * x.shape[-1] * module.out_features

    hooks = [
        module.register_forward_hook(count_linear)
        for module in transformer.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    # -- block-0 inputs, captured for the resident-block floor -------------
    lists = repeated_block_lists(transformer)
    blocks = [block for group in lists.values() for block in group]
    captured: dict[str, Any] = {}

    def capture(_module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if "args" not in captured:
            captured["args"] = inputs
            captured["kwargs"] = kwargs

    capture_hook = blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
    transformer_inputs: dict[str, Any] = {}

    def capture_transformer(
        _module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if transformer_inputs:
            return

        def to_cpu(value: Any) -> Any:
            return value.detach().cpu() if isinstance(value, torch.Tensor) else value

        transformer_inputs["args"] = tuple(to_cpu(value) for value in inputs)
        transformer_inputs["kwargs"] = {key: to_cpu(value) for key, value in kwargs.items()}

    transformer_hook = transformer.register_forward_pre_hook(capture_transformer, with_kwargs=True)

    with torch.no_grad():
        for index in range(args.warmup):
            denoise_step(model, state, index)
    torch.cuda.synchronize(device)
    for hook in hooks:
        hook.remove()
    capture_hook.remove()
    transformer_hook.remove()
    if args.dump_transformer_inputs is not None:
        args.dump_transformer_inputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(transformer_inputs, args.dump_transformer_inputs)
        print(f"transformer inputs -> {args.dump_transformer_inputs}", flush=True)
    if args.compile:
        # The counting hooks' closures made dynamo recompile every call until
        # it gave up on the block frame; forget that and warm the compiled
        # path up again without hooks.
        importlib.import_module("torch._dynamo").reset()
        with torch.no_grad():
            for index in range(args.warmup):
                denoise_step(model, state, index)
        torch.cuda.synchronize(device)
    linear_flops_per_step = linear_flops / max(1, args.warmup)
    n_heads = int(getattr(transformer.config, "num_attention_heads", 24))
    head_dim = int(getattr(transformer.config, "attention_head_dim", 128))
    seq = image_tokens + text_tokens
    attention_flops_per_step = 4.0 * batch * n_heads * seq * seq * head_dim * len(blocks)
    counters_before = streaming.counters if streaming is not None else {}

    # -- profiled steps ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as prof,
        torch.no_grad(),
    ):
        for index in range(args.steps):
            denoise_step(model, state, (args.warmup + index) % int(root.sampling.num_steps))
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    step_ms = wall / args.steps * 1000.0
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20

    bucket_us: dict[str, float] = defaultdict(float)
    top: list[tuple[float, str]] = []
    total_us = 0.0
    for event in prof.key_averages():
        key = str(event.key)
        if key.startswith(("aten::", "cuda", "ProfilerStep", "Command Buffer")):
            continue
        us = float(getattr(event, "self_device_time_total", 0) or 0)
        if us <= 0:
            continue
        total_us += us
        bucket_us[_bucket(key)] += us
        top.append((us, key))
    device_ms = total_us / 1e3 / args.steps
    gemm_ms = bucket_us.get("gemm", 0.0) / 1e3 / args.steps
    attention_ms = bucket_us.get("attention", 0.0) / 1e3 / args.steps

    print(
        f"\n=== {step_ms:.0f} ms per denoise step (wall), {device_ms:.0f} ms of device kernels ==="
    )
    print(f"  device idle inside the step: {max(0.0, step_ms - device_ms):.0f} ms")
    print(f"  peak allocated: {peak_mb:.0f} MiB")
    print("\n--- device time by bucket, per step ---")
    for name in ("gemm", "attention", "norm/elementwise", "reduction", "copy/memset", "other"):
        us = bucket_us.get(name, 0.0)
        if us > 0:
            print(f"  {name:18s} {us / total_us * 100:5.1f}%  {us / 1e3 / args.steps:8.1f} ms")
    peak_tflops = _cublas_peak_tflops(device, next(transformer.parameters()).dtype)
    gemm_tflops = linear_flops_per_step / gemm_ms / 1e9 if gemm_ms else 0.0
    attention_tflops = attention_flops_per_step / attention_ms / 1e9 if attention_ms else 0.0
    print("\n--- achieved throughput ---")
    print(
        f"  GEMM      {linear_flops_per_step / 1e12:7.1f} TFLOP/step  {gemm_tflops:6.1f} TFLOP/s"
    )
    print(
        f"  attention {attention_flops_per_step / 1e12:7.1f} TFLOP/step  {attention_tflops:6.1f} TFLOP/s"
    )
    print(f"  cuBLAS peak at 32768x3072x12288: {peak_tflops:.1f} TFLOP/s")
    floor_ms = (linear_flops_per_step + attention_flops_per_step) / peak_tflops / 1e9
    print(
        f"  FLOP floor of the step at that peak: {floor_ms:.0f} ms ({floor_ms / step_ms * 100:.0f}% of wall)"
    )
    print("\n--- top 12 kernels by self device time ---")
    for us, name in sorted(top, reverse=True)[:12]:
        print(
            f"  {us / total_us * 100:5.1f}%  {us / 1e3 / args.steps:8.1f} ms  [{_bucket(name):16s}] {name[:72]}"
        )

    # -- resident block floor ---------------------------------------------
    if streamed and "args" in captured:
        block = blocks[0]
        streamer = block._streamer
        upload = streamer._upload(streamer._positions[id(block)])
        if upload.ready is not None:
            torch.cuda.current_stream(device).wait_event(upload.ready)
        resident = {**block._resident_state, **upload.copies}
        block._streaming_active = True
        try:
            with torch.no_grad():
                block_ms = _median_ms(
                    lambda: torch.func.functional_call(
                        block, resident, captured["args"], captured["kwargs"]
                    ),
                    iters=args.block_iters,
                )
        finally:
            block._streaming_active = False
        stack_ms = block_ms * len(blocks)
        print("\n--- host streaming ---")
        print(
            f"  one block with resident weights: {block_ms:.1f} ms; x{len(blocks)} blocks = {stack_ms:.0f} ms"
        )
        print(
            f"  measured step minus resident stack: {step_ms - stack_ms:.0f} ms (streaming + stem/head + scheduler)"
        )
        after = streaming.counters
        print(
            "  prefetch over the profiled steps: hits "
            f"{after['prefetch_hits'] - counters_before['prefetch_hits']}, misses "
            f"{after['prefetch_misses'] - counters_before['prefetch_misses']}"
        )

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(f"\nchrome trace -> {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

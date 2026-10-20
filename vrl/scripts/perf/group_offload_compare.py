"""Our block streaming against diffusers' group offloading on one transformer forward.

Both keep a transformer that does not fit the card on the host and move it
block by block with one block of prefetch on a side stream; diffusers ships
the mechanism as ``apply_group_offloading(offload_type="block_level",
use_stream=True)``. This probe times the bare transformer forward under each,
on the inputs ``layered_denoise_probe --dump-transformer-inputs`` captured
from a real denoise step, so the two arms see identical work.

    python -m vrl.scripts.perf.layered_denoise_probe --cache ... \\
        --dump-transformer-inputs outputs/perf/layered_transformer_inputs.pt
    python -m vrl.scripts.perf.group_offload_compare \\
        --inputs outputs/perf/layered_transformer_inputs.pt --arm block_stream
    python -m vrl.scripts.perf.group_offload_compare \\
        --inputs outputs/perf/layered_transformer_inputs.pt --arm group_offload

One arm per process: each pins or registers host memory in its own way and
neither undoes it. ``group_offload`` runs with ``low_cpu_mem_usage=True``
(pinned staging buffers, like our pre-pinning path) unless ``--pin-masters``
asks for diffusers' default, which pins the whole model's host copy.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_SNAPSHOT = (
    "~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Layered/snapshots/"
    "8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf"
)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--arm", choices=["block_stream", "group_offload"], required=True)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--pin-masters", action="store_true", help="group_offload: pin the host copy"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    from diffusers import QwenImageTransformer2DModel

    device = torch.device(args.device)
    captured = torch.load(args.inputs, map_location="cpu", weights_only=False)
    call_args = _to_device(captured["args"], device)
    call_kwargs = _to_device(captured["kwargs"], device)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.expanduser(args.snapshot), subfolder="transformer", torch_dtype=torch.bfloat16
    ).eval()
    transformer.requires_grad_(False)

    started = time.perf_counter()
    if args.arm == "block_stream":
        from vrl.nn.offload import HostStreaming

        streaming = HostStreaming.install(transformer, device=device)
        setup = f"{len(streaming.blocks)} blocks streamed"
    else:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            transformer,
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            non_blocking=True,
            use_stream=True,
            record_stream=True,
            low_cpu_mem_usage=not args.pin_masters,
        )
        setup = f"group offloading, pin_masters={args.pin_masters}"
    print(f"{args.arm}: {setup} in {time.perf_counter() - started:.1f}s", flush=True)

    def forward() -> None:
        with torch.no_grad():
            transformer(*call_args, **call_kwargs)

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        forward()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - tick) * 1000.0)
    times.sort()
    print(
        f"{args.arm}: median {times[len(times) // 2]:.0f} ms per transformer forward "
        f"(min {times[0]:.0f}), peak allocated {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB",
        flush=True,
    )
    if args.arm == "block_stream":
        print(f"{args.arm}: {HostStreaming.of(transformer).counters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

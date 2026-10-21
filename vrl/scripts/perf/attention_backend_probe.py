"""Which SDPA kernels this card can run for one attention shape, and how fast.

The DiT families attend through ``F.scaled_dot_product_attention`` (diffusers'
``dispatch_attention_fn`` with no backend selected), so the kernel that runs is
whatever SDPA picks for the shape, dtype and device. On a new architecture
(sm_120 Blackwell consumer cards) the eligible set is not obvious: this probe
asks the dispatcher directly for each backend, then times the eligible ones.

Usage (the Qwen-Image-Layered 640x640, 9-layer rollout shape by default:
``(layers + 1 + 1 condition) * 1600`` image tokens plus the text tokens):

    python -m vrl.scripts.perf.attention_backend_probe --batch 4 --seq 17700
    python -m vrl.scripts.perf.attention_backend_probe --no-time   # eligibility only

Timing is per SDPA call on an otherwise idle card; read the TFLOP/s column as
``4 * B * H * S^2 * D`` per forward (twice that with ``--backward``). A busy
card makes the numbers meaningless, so the probe refuses one unless
``--allow-busy`` is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_BACKENDS = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _eligible(backend: SDPBackend, params: torch.backends.cuda.SDPAParams) -> bool:
    checks = {
        SDPBackend.FLASH_ATTENTION: torch.backends.cuda.can_use_flash_attention,
        SDPBackend.CUDNN_ATTENTION: torch.backends.cuda.can_use_cudnn_attention,
        SDPBackend.EFFICIENT_ATTENTION: torch.backends.cuda.can_use_efficient_attention,
    }
    check = checks.get(backend)
    return True if check is None else bool(check(params, True))


def _other_processes_on(device: torch.device) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    own = str(os.getpid())
    return [
        line
        for line in result.stdout.splitlines()
        if line.strip() and line.split(",")[0].strip() != own
    ]


def _time_backend(
    backend: SDPBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    iters: int,
    backward: bool,
) -> float:
    """Median milliseconds of one SDPA call (plus its backward when asked)."""

    def one() -> None:
        out = F.scaled_dot_product_attention(q, k, v)
        if backward:
            out.sum().backward()

    with sdpa_kernel([backend]):
        for _ in range(2):
            one()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            one()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq", type=int, default=17700)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backward", action="store_true", help="time forward plus backward")
    parser.add_argument("--no-time", action="store_true", help="report eligibility only")
    parser.add_argument("--allow-busy", action="store_true", help="time even with other GPU jobs")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("the probe needs a CUDA device")
    dtype = _DTYPES[args.dtype]
    capability = torch.cuda.get_device_capability(device)
    print(
        f"{torch.cuda.get_device_name(device)} sm_{capability[0]}{capability[1]} "
        f"torch {torch.__version__}; shape B={args.batch} H={args.heads} "
        f"S={args.seq} D={args.head_dim} {args.dtype}",
        flush=True,
    )
    for module in ("flash_attn", "flash_attn_interface", "sageattention", "xformers"):
        found = importlib.util.find_spec(module) is not None
        print(f"  {module:22s} {'installed' if found else 'not installed'}")

    shape = (args.batch, args.heads, args.seq, args.head_dim)
    requires_grad = args.backward and not args.no_time
    q, k, v = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        for _ in range(3)
    )
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    eligible = {backend: _eligible(backend, params) for backend in _BACKENDS}
    print("eligibility (dispatcher reasons for a refusal print above):")
    for backend, ok in eligible.items():
        print(f"  {backend.name:20s} {'eligible' if ok else 'refused'}")

    if args.no_time:
        return 0
    busy = _other_processes_on(device)
    if busy and not args.allow_busy:
        print(f"other processes hold the card, not timing: {busy}", file=sys.stderr)
        return 2
    flops = 4.0 * args.batch * args.heads * args.seq**2 * args.head_dim
    if args.backward:
        flops *= 3.5  # forward + ~2.5x forward for the backward of flash-style kernels
    print(
        f"median ms per call over {args.iters} iterations{' (fwd+bwd)' if args.backward else ''}:"
    )
    for backend, ok in eligible.items():
        if not ok:
            continue
        if backend is SDPBackend.MATH and args.seq > 8192:
            print(f"  {backend.name:20s} skipped (materialises S^2 scores)")
            continue
        try:
            ms = _time_backend(backend, q, k, v, iters=args.iters, backward=args.backward)
        except (RuntimeError, torch.OutOfMemoryError) as error:
            print(f"  {backend.name:20s} failed: {str(error).splitlines()[0]}")
            continue
        print(f"  {backend.name:20s} {ms:9.2f} ms  {flops / ms / 1e9:8.1f} TFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

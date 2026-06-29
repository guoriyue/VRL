"""Probe whether data movement and CPU work overlap with denoise compute.

Run:
    python -m vrl.scripts.perf.single_gpu_overlap_copy_probe

This is the complementary check to the compute-overlap probe: tensor-heavy
reward compute may serialize on one GPU, while DMA copy work and host-side
orchestration can still hide behind the next denoise forward.
"""

from __future__ import annotations

import argparse
import time

import torch

from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs


def _synchronize() -> None:
    torch.cuda.synchronize()


def _elapsed_ms(fn) -> float:
    started = time.perf_counter()
    fn()
    _synchronize()
    return (time.perf_counter() - started) * 1000.0


def _busy_wait(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="cosmos-predict2.5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--copy-rounds", type=int, default=4)
    parser.add_argument("--host-gib", type=float, default=1.0)
    parser.add_argument("--cpu-seconds", type=float, default=0.4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this overlap probe")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    model, kwargs = build_synthetic_inputs(
        args.family,
        batch=args.batch,
        device=device,
        dtype=dtype,
    )
    batch, channels, _frames, height, width = kwargs["hidden_states"].shape
    kwargs["hidden_states"] = torch.randn(
        batch,
        channels,
        args.frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )

    element_size = torch.tensor([], dtype=torch.float16).element_size()
    elements = max(1, int(args.host_gib * (1024**3) / element_size))
    host_buffer = torch.empty(elements, dtype=torch.float16, pin_memory=True)
    device_buffer = torch.empty_like(host_buffer, device=device)

    def compute() -> torch.Tensor:
        with torch.no_grad():
            return model(**kwargs)[0]

    def data_move_and_cpu(stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for _ in range(args.copy_rounds):
                device_buffer.copy_(host_buffer, non_blocking=True)
                host_buffer.copy_(device_buffer, non_blocking=True)

        # Host orchestration is independent of tensor-core work and is part of
        # the boundary this staged pipeline aims to hide.
        _busy_wait(args.cpu_seconds)

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    for _ in range(args.warmup):
        compute()
        data_move_and_cpu(copy_stream)
    _synchronize()

    def run_compute_only() -> None:
        for _ in range(args.iters):
            compute()

    def run_data_only() -> None:
        for _ in range(args.iters):
            data_move_and_cpu(copy_stream)
            _synchronize()

    def run_serial() -> None:
        for _ in range(args.iters):
            compute()
            _synchronize()
            data_move_and_cpu(copy_stream)
            _synchronize()

    def run_concurrent() -> None:
        for _ in range(args.iters):
            with torch.cuda.stream(compute_stream):
                compute()
            data_move_and_cpu(copy_stream)
            _synchronize()

    compute_ms = _elapsed_ms(run_compute_only) / args.iters
    data_ms = _elapsed_ms(run_data_only) / args.iters
    serial_ms = _elapsed_ms(run_serial) / args.iters
    concurrent_ms = _elapsed_ms(run_concurrent) / args.iters
    hidden_ms = serial_ms - concurrent_ms
    hidden_pct = hidden_ms / serial_ms * 100.0

    print("=== single-GPU compute vs data-move+CPU overlap probe ===")
    print(f"family:                     {args.family}")
    print(f"frames:                     {args.frames}")
    print(f"host buffer:                {args.host_gib:.2f} GiB")
    print(f"copy rounds:                {args.copy_rounds}")
    print(f"CPU busy wait:              {args.cpu_seconds:.3f} s")
    print(f"compute alone:              {compute_ms:.0f} ms/iter")
    print(f"data-move+CPU alone:        {data_ms:.0f} ms/iter")
    print(f"serial compute then data:   {serial_ms:.0f} ms/iter")
    print(f"concurrent overlapped:      {concurrent_ms:.0f} ms/iter")
    print(f"hidden behind compute:      {hidden_ms:+.0f} ms/iter ({hidden_pct:+.0f}%)")
    print(f"data stage hidden:          {hidden_ms / data_ms * 100.0:+.0f}%")


if __name__ == "__main__":
    main()

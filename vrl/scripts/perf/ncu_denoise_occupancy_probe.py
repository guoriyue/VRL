"""Minimal NCU target: cosmos denoise DiT forward (synthetic real dims, 240p_33f).
Measures achieved SM occupancy / SM throughput of the actual denoise kernels to
answer: does denoise leave SM headroom for concurrent single-GPU multi-staging?"""

from __future__ import annotations

import torch

from vrl.scripts.perf.gemm_projection_breakdown import build_synthetic_inputs


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model, kwargs = build_synthetic_inputs(
        "cosmos-predict2.5",
        batch=1,
        device=device,
        dtype=dtype,
    )
    batch, channels, _, height, width = kwargs["hidden_states"].shape
    kwargs["hidden_states"] = torch.randn(
        batch,
        channels,
        33,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        # NCU replays kernels, but warmup is still needed before capture.
        for _ in range(2):
            model(**kwargs)
        torch.cuda.synchronize()
        model(**kwargs)
        torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()

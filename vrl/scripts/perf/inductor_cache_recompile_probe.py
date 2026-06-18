"""Measure whether a persistent TORCHINDUCTOR_CACHE_DIR makes a *recompile*
cheaper across process boundaries — the rollout-worker-recreate scenario.

When a colocated rollout worker is released after a collect cycle and rebuilt
next cycle, it is a fresh process: torch.compile re-runs Dynamo trace + guard
build + inductor codegen from scratch. inductor's on-disk codegen cache
(TORCHINDUCTOR_CACHE_DIR) survives the process, so a rebuilt worker pointed at
the same cache dir should skip codegen (the expensive part) and only pay
trace+guard. This probe quantifies that gap.

It is a representative proxy (a small SDPA transformer stack), not the full
Cosmos Predict2.5 2B load — the inductor codegen-cache *mechanism* is what's
measured, and it is model-agnostic. Run on a CUDA box:

    python -m vrl.scripts.perf.inductor_cache_recompile_probe

One-shot measurement artifact (perf/), not imported by runtime code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time


def _child() -> None:
    """Compile a representative module in THIS process and print compile wall time.

    TORCHINDUCTOR_CACHE_DIR is read from the env set by the parent. The first
    forward triggers Dynamo trace + inductor codegen (cold) or a codegen
    cache-hit (warm).
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    dim, heads, layers, seq, batch = 1024, 16, 6, 256, 2
    head_dim = dim // heads

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, 4 * dim)
            self.fc2 = nn.Linear(4 * dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, n, _ = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).view(b, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(b, n, dim)
            x = x + self.proj(a)
            x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
            return x

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x

    model = Net().to(device, dtype).eval()
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    compiled = torch.compile(model, mode="default", fullgraph=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(x)  # trace + codegen (cold) OR codegen cache-hit (warm)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"COMPILE_WALL_S={t1 - t0:.3f}")


def _run_child(cache_dir: str) -> float:
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    out = subprocess.run(
        [sys.executable, "-m", "vrl.scripts.perf.inductor_cache_recompile_probe", "--child"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("COMPILE_WALL_S="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"child did not report compile time:\nSTDOUT:{out.stdout}\nSTDERR:{out.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child()
        return

    with tempfile.TemporaryDirectory() as shared, tempfile.TemporaryDirectory() as other:
        t_cold = _run_child(shared)         # cold: fresh cache dir
        t_warm = _run_child(shared)         # warm: SAME dir, codegen cached on disk
        t_cold2 = _run_child(other)         # control: another fresh dir (cold again)

    saved = t_cold - t_warm
    pct = (saved / t_cold * 100.0) if t_cold else 0.0
    print("\n=== TORCHINDUCTOR_CACHE_DIR recompile probe (cross-process) ===")
    print(f"cold  (fresh cache dir)        : {t_cold:.3f}s")
    print(f"warm  (same cache dir reused)  : {t_warm:.3f}s")
    print(f"cold2 (different fresh dir)    : {t_cold2:.3f}s  [control — should ~ cold]")
    print(f"--> warm-cache saving vs cold  : {saved:.3f}s ({pct:.0f}% of cold compile)")
    print("Interpretation: a rebuilt rollout worker pointed at a persistent")
    print("TORCHINDUCTOR_CACHE_DIR pays ~warm, not ~cold, on each recreate.")


if __name__ == "__main__":
    main()

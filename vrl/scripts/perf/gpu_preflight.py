"""Perf-only GPU preflight: verify the build matches the silicon and measure the
local bf16/SDPA envelope before interpreting MFU numbers.

This exists because a wrong MFU denominator silently mis-diagnoses everything. The
vendor "TFLOPS"/"AI TOPS" headline is the fp8/fp4 (often sparse) number; a bf16 DiT
runs at the much lower bf16-dense-with-fp32-accumulate rate (consumer GeForce halves
that again). Measuring an RTX 5090 at its real ~232 bf16 TFLOPS — not the 419 headline
— is the difference between "the DiT is saturated" and a phantom "51% MFU, go write a
kernel" goose chase (see SPRINT_lossless_diffusion_rl_research).

What it checks/measures (all on the live GPU, one-time per process):
  1. torch is BUILT for this GPU's SM (else PTX-JIT fallback / failure).
  2. the real achievable bf16-dense GEMM peak (the correct MFU denominator).
  3. the fastest EXACT SDPA backend at a DiT attention shape, selected by
     ``best_sdpa_backend``. On the RTX 5090, flash (the bundled FA-2) and cuDNN are
     within ~5% and the winner flips with thermal/warmup noise (measured: 187-203
     TFLOPS either way), so this is a wash, not a real lever — but the check still
     pins the per-machine best and flags a genuine gap on GPUs where one wins clearly.
     FA-3 is Hopper-only (sm_90); it does NOT exist for Blackwell sm_120.

This is a profiling/probe artifact, not trainer startup logic. It intentionally
lives under ``vrl.scripts.perf`` so production online recipes do not allocate
large benchmark tensors or synchronize CUDA just to start training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# Module cache so repeated perf probes in one process measure once.
_PEAK_CACHE: dict[int, float] = {}
_REPORT_CACHE: GpuPreflight | None = None


@dataclass(frozen=True, slots=True)
class GpuPreflight:
    """Display/provenance-only perf report for interpreting probe output."""

    # display/provenance-only: this dataclass is the report emitted by perf probes,
    # not a resolved runtime capability consumed by training control flow.
    available: bool
    device_name: str
    sm: str
    arch_list: tuple[str, ...]
    arch_ok: bool
    bf16_peak_tflops: float
    sdpa_backend_tflops: dict[str, float]
    best_sdpa_backend: str
    flash_attn_pkg: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _cuda_time_ms(fn, iters: int, warmup: int, trials: int = 3) -> float:
    """Median over ``trials`` of the mean per-call ms — robust to warmup/scheduler noise.

    A single short timing run mis-ranks near-tied kernels (e.g. flash vs cuDNN differ
    by ~8% but a 20-iter run lands inside the noise band and flips the winner). The
    median of a few well-warmed trials is stable.
    """
    import statistics

    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means: list[float] = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) / iters)
    return statistics.median(means)


def measured_bf16_peak_tflops(size: int = 12288) -> float:
    """Achievable dense bf16 (fp32-accumulate) GEMM TFLOPS on the current GPU.

    The correct MFU denominator — measured, not the vendor headline. Cached per size.
    Returns 0.0 with no CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    if size in _PEAK_CACHE:
        return _PEAK_CACHE[size]
    torch.backends.cuda.matmul.allow_tf32 = False
    best = 0.0
    # Two sizes: the big one tops out the tensor cores; the smaller is a fast floor.
    for n in (8192, size):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        ms = _cuda_time_ms(lambda a=a, b=b: a @ b, iters=30, warmup=12)
        best = max(best, 2 * n * n * n / (ms / 1e3) / 1e12)
        del a, b
    torch.cuda.empty_cache()
    _PEAK_CACHE[size] = best
    return best


def _bench_sdpa_backends() -> dict[str, float]:
    """TFLOPS of each EXACT SDPA backend at a representative DiT attention shape.

    Shape ~ a video DiT self-attention (heads=16, head_dim=128, seq~7k). Skips the
    math backend (reference, never the fast path). Returns {backend_name: TFLOPS}.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    b, h, s, d = 1, 16, 7040, 128
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    flops = 4 * b * h * s * s * d
    out: dict[str, float] = {}
    for backend, name in (
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.CUDNN_ATTENTION, "cudnn"),
        (SDPBackend.EFFICIENT_ATTENTION, "mem_efficient"),
    ):
        try:
            with sdpa_kernel(backend):
                ms = _cuda_time_ms(
                    lambda q=q, k=k, v=v: sdpa(q, k, v),
                    iters=60,
                    warmup=15,
                )
            out[name] = flops / (ms / 1e3) / 1e12
        except Exception:  # backend unavailable for this shape/build
            continue
    del q, k, v
    torch.cuda.empty_cache()
    return out


def _flash_attn_version() -> str | None:
    try:
        import flash_attn

        return getattr(flash_attn, "__version__", "installed")
    except Exception:
        return None


def run_gpu_preflight(*, force: bool = False) -> GpuPreflight:
    """Measure the GPU once and return the resolved facts + best kernels (cached)."""
    global _REPORT_CACHE
    if _REPORT_CACHE is not None and not force:
        return _REPORT_CACHE

    import torch

    if not torch.cuda.is_available():
        _REPORT_CACHE = GpuPreflight(
            available=False, device_name="cpu", sm="", arch_list=(), arch_ok=False,
            bf16_peak_tflops=0.0, sdpa_backend_tflops={}, best_sdpa_backend="math",
            flash_attn_pkg=None, warnings=("no CUDA device",),
        )
        return _REPORT_CACHE

    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = tuple(torch.cuda.get_arch_list())
    warns: list[str] = []
    # arch match: sm_120 native needs an exact (or same-major) entry, else PTX-JIT.
    arch_ok = sm in arch_list
    if not arch_ok:
        warns.append(
            f"{sm} NOT in torch build arch_list {list(arch_list)} -> PTX-JIT fallback "
            f"or failure; install a torch build whose CUDA covers {sm}",
        )

    peak = measured_bf16_peak_tflops()
    sdpa = _bench_sdpa_backends()
    best = max(sdpa, key=sdpa.get) if sdpa else "math"
    if best != "flash" and sdpa.get("flash") and sdpa[best] > sdpa["flash"] * 1.03:
        warns.append(
            f"fastest SDPA backend is '{best}' ({sdpa[best]:.0f} TFLOPS) but torch may "
            f"default to 'flash' ({sdpa['flash']:.0f}); prefer '{best}' via sdpa_kernel "
            f"for a free exact speedup on this GPU",
        )
    fa = _flash_attn_version()

    _REPORT_CACHE = GpuPreflight(
        available=True, device_name=torch.cuda.get_device_name(0), sm=sm,
        arch_list=arch_list, arch_ok=arch_ok, bf16_peak_tflops=peak,
        sdpa_backend_tflops=sdpa, best_sdpa_backend=best, flash_attn_pkg=fa,
        warnings=tuple(warns),
    )
    return _REPORT_CACHE


def log_gpu_preflight() -> GpuPreflight:
    """Log the perf preflight report and return it."""
    r = run_gpu_preflight()
    if not r.available:
        logger.info("GPU preflight: no CUDA device (CPU run)")
        return r
    logger.info(
        "GPU preflight: %s (%s) | arch_ok=%s | bf16 peak=%.0f TFLOPS (MFU denominator) "
        "| best SDPA=%s %s | flash_attn_pkg=%s",
        r.device_name, r.sm, r.arch_ok, r.bf16_peak_tflops, r.best_sdpa_backend,
        {k: round(v) for k, v in r.sdpa_backend_tflops.items()}, r.flash_attn_pkg,
    )
    for w in r.warnings:
        logger.warning("GPU preflight: %s", w)
    return r


def main() -> None:
    log_gpu_preflight()


__all__ = [
    "GpuPreflight",
    "log_gpu_preflight",
    "main",
    "measured_bf16_peak_tflops",
    "run_gpu_preflight",
]


if __name__ == "__main__":
    main()

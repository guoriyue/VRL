"""Per-projection GEMM time breakdown for diffusion DiT forward passes.

WHY: profiling (docs/sprints/info/SPRINT_{rollout,cosmos}_performance.md) shows the
DiT forward is GEMM-dominant (~48-51% of kernel time), but it never split *which*
GEMMs own that time. FP8 and projection-fusion pay off in different places
depending on whether FFN, QKV, AdaLN/modulation, attention-out, or the IO
embedders dominate. This tool attributes every ``nn.Linear``'s device time to a
projection category, so SPRINT_gemm_utilization.md P0 ("逐-projection GEMM 拆分")
stops being a guess.

HOW: each ``nn.Linear.forward`` is wrapped in a ``torch.profiler.record_function``
range named ``projgemm/<category>``; the profiler's per-range self device-time is
then the GEMM kernel time directly issued by that linear (linears don't nest, so
self-time == that linear's matmul). PEFT splits cleanly for free: a wrapped LoRA
linear's children ``base_layer`` (category by its name, e.g. qkv) and
``lora_A``/``lora_B`` (category ``lora``) nest, so base-GEMM vs LoRA-GEMM time
separate without extra work.

FAITHFULNESS: GEMM kernel time is *shape*-determined, not weight-value-determined.
A config-init (random-weight) transformer at the REAL config dims + REAL
latent/token shapes therefore yields the same per-projection split as the real
checkpoint -- no weights download needed. Run on the target GPU for real numbers;
the synthetic CPU path is a self-test and relative sanity check.

Usage:
    # Real numbers (needs the cosmos extra / torchvision on the GPU box):
    python -m vrl.scripts.perf.gemm_projection_breakdown --family cosmos-predict2 --device cuda

    # CPU self-test (no torchvision needed for sd3_5 / wan_2_1):
    python -m vrl.scripts.perf.gemm_projection_breakdown --family sd3_5 --device cpu

    # Or wire the reusable core into a REAL training/rollout forward:
    from vrl.scripts.perf.gemm_projection_breakdown import (
        profile_projection_gemms, format_report,
    )
    bd = profile_projection_gemms(real_model.transformer, run_one_denoise_step, device=dev)
    print(format_report(bd))
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vrl.scripts.perf.common.synthetic_diffusion import (
    build_synthetic_forward,
    build_synthetic_inputs,
)
from vrl.utils.logging import init_logger
from vrl.utils.profiling import record_function

logger = init_logger(__name__)

__all__ = [
    "Breakdown",
    "apply_qkv_fusion",
    "build_synthetic_inputs",
    "classify_linear",
    "format_report",
    "instrument_projection_gemms",
    "profile_projection_gemms",
]

# Projection categories, ordered most-specific first; the FIRST match on the
# lowercased module FQN wins. Validated against the real diffusers FQNs of
# CosmosTransformer3DModel / SD3Transformer2DModel / WanTransformer3DModel
# (every linear in all three lands in a named bucket; "other" stays empty).
PROJECTION_ORDER: tuple[str, ...] = (
    "ffn",
    "qkv",
    "attn_out",
    "adaln",
    "io_embed",
    "lora",
    "other",
)


def classify_linear(fqn: str) -> str:
    """Map a Linear's fully-qualified module name to a projection category.

    Order matters: ``lora`` is checked before ``qkv`` so a PEFT ``...to_q.lora_A``
    is a LoRA GEMM, while its sibling ``...to_q.base_layer`` is still a QKV GEMM.
    """

    name = fqn.lower()
    parts = name.split(".")
    leaf = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else ""
    if "lora_a" in name or "lora_b" in name:
        return "lora"
    if any(
        tag in name
        for tag in (".to_q", ".to_k", ".to_v", ".to_qkv", ".to_added_qkv",
                    "add_q_proj", "add_k_proj", "add_v_proj")
    ):
        return "qkv"
    if "to_out" in name or "to_add_out" in name:
        return "attn_out"
    # FFN/MLP, including SD3's dual-stream context FF (ff_context.net.*).
    if ("net." in name and "ff" in name) or ".mlp." in name or "feed_forward" in name:
        return "ffn"
    # AdaLN / modulation: a Linear whose PARENT module is a norm* (normN / norm_out /
    # norm*_context). Parent-based (not substring) so a top-level `norm_out.linear`
    # with no leading dot is still caught, and `t_embedder.linear_1` is not.
    if parent.startswith("norm") and leaf in ("linear", "linear_1", "linear_2"):
        return "adaln"
    if any(
        tag in name
        for tag in ("time_embed", "time_text_embed", "timestep_embedder", "text_embedder",
                    "t_embedder", "condition_embedder", "context_embedder", "patch_embed",
                    "caption", "pos_embed", "time_proj", "proj_out")
    ):
        return "io_embed"
    return "other"


@dataclass(slots=True)
class Breakdown:
    """Per-category GEMM time, plus coverage bookkeeping."""

    device_us: dict[str, float] = field(default_factory=dict)
    cpu_us: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    category_fqns: dict[str, list[str]] = field(default_factory=dict)
    device_kind: str = "cpu"  # "cuda" when device time is real kernel time

    def primary_us(self) -> dict[str, float]:
        """Device time on CUDA (real kernel time), else CPU time on a CPU run."""

        return self.device_us if self.device_kind == "cuda" else self.cpu_us


@contextmanager
def instrument_projection_gemms(model: nn.Module) -> Iterator[dict[str, list[str]]]:
    """Wrap every ``nn.Linear.forward`` in a ``projgemm/<category>`` profiler range.

    Yields the category -> sorted FQN list mapping (so the report can show exactly
    which modules fed each bucket and surface anything that fell to ``other``).
    Restores the original ``forward`` methods on exit.
    """

    originals: list[tuple[nn.Linear, Any]] = []
    category_fqns: dict[str, list[str]] = {cat: [] for cat in PROJECTION_ORDER}

    def make_wrapper(orig_forward: Any, label: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with record_function(label):
                return orig_forward(*args, **kwargs)

        return wrapped

    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        category = classify_linear(fqn)
        category_fqns[category].append(fqn)
        originals.append((module, module.forward))
        module.forward = make_wrapper(module.forward, f"projgemm/{category}")  # type: ignore[method-assign]

    for cat in category_fqns:
        category_fqns[cat].sort()
    try:
        yield category_fqns
    finally:
        for module, orig in originals:
            module.forward = orig  # type: ignore[method-assign]


def _event_self_us(event: Any, *, cuda: bool) -> tuple[float, float]:
    """(device_us, cpu_us) self-time for a profiler event, version-tolerant."""

    cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
    device_us = 0.0
    if cuda:
        # torch >= 2.1 renamed self_cuda_time_total -> self_device_time_total.
        device_us = float(
            getattr(event, "self_device_time_total", None)
            or getattr(event, "self_cuda_time_total", 0.0)
            or 0.0
        )
    return device_us, cpu_us


def profile_projection_gemms(
    model: nn.Module,
    forward_fn: Callable[[], Any],
    *,
    device: torch.device | str = "cpu",
    warmup: int = 3,
    active: int = 5,
) -> Breakdown:
    """Profile ``forward_fn`` and attribute Linear GEMM time per projection category.

    ``model`` is the module whose linears get instrumented (e.g. the transformer);
    ``forward_fn`` is a zero-arg thunk that runs one forward / denoise step on it.
    Run on CUDA for real kernel time; CPU works for self-test (CPU self-time).
    """

    device = torch.device(device)
    cuda = device.type == "cuda"

    for _ in range(max(0, warmup)):
        forward_fn()
    if cuda:
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with (
        instrument_projection_gemms(model) as category_fqns,
        torch.profiler.profile(activities=activities, record_shapes=False) as prof,
    ):
        for _ in range(max(1, active)):
            forward_fn()
        if cuda:
            torch.cuda.synchronize()

    bd = Breakdown(
        category_fqns=category_fqns,
        device_kind="cuda" if cuda else "cpu",
    )
    for cat in PROJECTION_ORDER:
        bd.device_us[cat] = 0.0
        bd.cpu_us[cat] = 0.0
        bd.calls[cat] = 0
    for event in prof.key_averages():
        key = str(event.key)
        if not key.startswith("projgemm/"):
            continue
        cat = key.split("/", 1)[1]
        if cat not in bd.device_us:
            continue
        device_us, cpu_us = _event_self_us(event, cuda=cuda)
        bd.device_us[cat] += device_us
        bd.cpu_us[cat] += cpu_us
        bd.calls[cat] += int(getattr(event, "count", 0) or 0)
    return bd


def format_report(bd: Breakdown) -> str:
    """Render the breakdown as a sorted table with explicit coverage notes."""

    primary = bd.primary_us()
    unit = "CUDA us" if bd.device_kind == "cuda" else "CPU us"
    total = sum(primary.values()) or 1.0

    rows = sorted(
        ((cat, primary[cat], bd.calls[cat]) for cat in PROJECTION_ORDER if bd.calls[cat]),
        key=lambda r: r[1],
        reverse=True,
    )
    lines = [
        f"per-projection GEMM breakdown  ({unit}, self-time; device={bd.device_kind})",
        f"{'category':<10}{unit:>14}{'%':>9}{'calls':>9}   modules",
        "-" * 78,
    ]
    for cat, us, calls in rows:
        n_fqn = len(bd.category_fqns.get(cat, []))
        lines.append(f"{cat:<10}{us:>14,.1f}{100.0 * us / total:>8.1f}%{calls:>9}   {n_fqn} linear(s)")
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<10}{total:>14,.1f}{100.0:>8.1f}%")

    other = bd.category_fqns.get("other", [])
    if other:
        lines.append("")
        lines.append(f"WARNING: {len(other)} linear(s) fell to 'other' (unclassified) -- coverage gap:")
        for fqn in other[:20]:
            lines.append(f"  - {fqn}")
        if len(other) > 20:
            lines.append(f"  ... and {len(other) - 20} more")
    else:
        lines.append("(coverage: every Linear classified; 'other' empty)")
    if bd.device_kind != "cuda":
        lines.append("NOTE: CPU run -- numbers are CPU self-time for self-test; run --device cuda for real GEMM kernel time.")
    return "\n".join(lines)


def apply_qkv_fusion(model: nn.Module, family: str) -> bool:
    """Fuse to_q/to_k/to_v into a single to_qkv GEMM where the model supports it.

    Returns True if fusion was applied. SD3/Wan expose ``fuse_qkv_projections()``;
    Cosmos has no public fuse API (would need a custom fused attention processor).
    Mathematically equivalent -- only the GEMM shape/launch count changes.
    """

    if hasattr(model, "fuse_qkv_projections"):
        model.fuse_qkv_projections()
        logger.info("applied fuse_qkv_projections() to %s", family)
        return True
    logger.warning(
        "%s has no fuse_qkv_projections() -- skipping (e.g. Cosmos needs a custom fused processor)",
        family,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="sd3_5",
        choices=["sd3_5", "cosmos-predict2", "cosmos-predict2.5", "wan_2_1"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch", type=int, default=2, help="2 == cond+uncond CFG pair")
    parser.add_argument(
        "--layers", type=int, default=None,
        help="override transformer depth (default: production; use a small value for a fast CPU self-test)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument(
        "--no-concat-padding-mask", dest="concat_padding_mask", action="store_false",
        help="cosmos only: skip the torchvision padding-mask resize (no GEMM impact)",
    )
    parser.add_argument(
        "--fuse-qkv", action="store_true",
        help="fuse to_q/to_k/to_v into one to_qkv GEMM (SD3/Wan; no-op on Cosmos) before profiling",
    )
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cpu" and dtype is not torch.float32:
        logger.info("CPU run: forcing fp32 (bf16/fp16 matmul is not CPU-accelerated)")
        dtype = torch.float32

    logger.info(
        "building synthetic %s on %s (%s), batch=%d", args.family, device, dtype, args.batch
    )
    model, forward_fn = build_synthetic_forward(
        args.family, batch=args.batch, device=device, dtype=dtype, layers=args.layers,
        concat_padding_mask=args.concat_padding_mask,
    )
    if args.fuse_qkv:
        apply_qkv_fusion(model, args.family)
    bd = profile_projection_gemms(
        model, forward_fn, device=device, warmup=args.warmup, active=args.active
    )
    print(format_report(bd))


if __name__ == "__main__":
    main()

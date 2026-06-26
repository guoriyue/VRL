"""Probe: can Cosmos3-Nano's generator enter this repo's diffusion seam? (Tier 1)

This is a one-shot *contract* probe, not a training entrypoint. It answers the
P0 questions from SPRINT_physical_ai_model_support.md §3.1 / §Tier-1:

  1. Does the official Diffusers path expose a trainable denoiser?
  2. Is there a replayable per-denoise-step logprob?
  3. Is the action output a diffusion token, JSON action, or server API?
  4. Do the Diffusers and vLLM-Omni paths share a tensor/layout?

Acceptance (sprint): generating a short clip is a PASS, but so is cleanly
recording a dependency / permission / VRAM blocker — the only failure is leaving
the contract unexplained. The probe therefore always writes a decision note and
never crashes on a missing model; it records the blocker instead.

Usage:
  python -m vrl.scripts.eval.cosmos3_nano_generator_probe \
    --out outputs/cosmos3_nano_probe/generator_decision.json \
    [--model nvidia/Cosmos3-Nano] [--prompt "..."] [--frames 17] [--steps 4]
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.models.support_matrix import get_support_entry

_ENTRY_KEY = "cosmos3_nano"


def _new_report(entry: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Seed the decision note with the matrix row + the open contract questions."""
    return {
        "probe": "cosmos3_nano_generator",
        "model": args.model or entry.path,
        "matrix_entry": asdict(entry),
        # The four contract questions, answered as we learn them.
        "questions": {
            "generator_exposes_trainable_denoiser": "unknown",
            "replayable_denoise_logprob": "unknown",
            "action_output_kind": "unknown",  # diffusion_token | json_action | server_api
            "diffusers_vllm_omni_layout_match": "unknown",
        },
        "component_graph": {},   # reasoner / generator / tokenizer / scheduler / media_tokenizer
        "denoise_tensor": {},    # shape / dtype / timestep / conditioning / output
        "artifact": None,        # path to a generated clip, if any
        "blockers": [],          # human-readable blockers (deps / weights / VRAM / permission)
        "decision": "undecided",  # positive | negative | undecided
        "env": {"python": platform.python_version()},
    }


def _probe_torch(report: dict[str, Any]) -> Any:
    """Record torch/CUDA availability; returns the torch module or None."""
    try:
        import torch
    except ImportError as err:
        report["blockers"].append(f"torch not importable: {err}")
        return None
    report["env"]["torch"] = torch.__version__
    report["env"]["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        report["env"]["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        report["env"]["gpu_free_gb"] = round(free / 1e9, 1)
        report["env"]["gpu_total_gb"] = round(total / 1e9, 1)
    else:
        report["blockers"].append("no CUDA device visible; 16B generator needs a GPU")
    return torch


def _probe_diffusers_pipeline(report: dict[str, Any], model: str, torch: Any) -> Any:
    """Try to load the Cosmos3-Nano Diffusers pipeline; record the component graph.

    Returns the loaded pipeline or None. A missing class / weights / auth is a
    recorded blocker, not a crash.
    """
    try:
        import diffusers
    except ImportError as err:
        report["blockers"].append(f"diffusers not importable: {err}")
        return None
    report["env"]["diffusers"] = diffusers.__version__

    # The omnimodal Cosmos 3 pipeline class name is not yet pinned in diffusers;
    # discover it instead of hard-assuming SD3/Wan equivalence (sprint §3.1). We
    # must distinguish a genuine Cosmos *3* / omnimodal pipeline from the Cosmos
    # 2.x classes already shipped — loading Nano weights into a Cosmos2 class
    # would be a false positive.
    cosmos_pipelines = [
        name for name in dir(diffusers) if "Cosmos" in name and name.endswith("Pipeline")
    ]
    report["component_graph"]["diffusers_cosmos_pipeline_candidates"] = cosmos_pipelines
    cosmos3 = [
        name for name in cosmos_pipelines if "Cosmos3" in name or "Omni" in name
    ]
    report["component_graph"]["diffusers_cosmos3_pipeline_candidates"] = cosmos3
    if not cosmos3:
        report["blockers"].append(
            "no Cosmos3/Omni *Pipeline class in this diffusers build "
            f"(found only Cosmos 2.x: {cosmos_pipelines}); the omnimodal Cosmos 3 "
            "generator is not exposed via the Diffusers path yet",
        )
        report["questions"]["generator_exposes_trainable_denoiser"] = "no_diffusers_path"
        return None

    pipeline_cls = getattr(diffusers, cosmos3[0])
    try:
        pipe = pipeline_cls.from_pretrained(
            model, torch_dtype=torch.bfloat16, local_files_only=True,
        )
    except Exception as err:
        report["blockers"].append(
            f"{cosmos3[0]}.from_pretrained({model!r}) failed: "
            f"{type(err).__name__}: {err}",
        )
        return None

    # Record the component graph the sprint asks for.
    components = getattr(pipe, "components", {}) or {}
    report["component_graph"]["components"] = sorted(components)
    report["component_graph"]["has_transformer"] = hasattr(pipe, "transformer")
    report["component_graph"]["has_scheduler"] = hasattr(pipe, "scheduler")
    report["component_graph"]["scheduler_cls"] = (
        type(pipe.scheduler).__name__ if getattr(pipe, "scheduler", None) else None
    )
    return pipe


def _probe_denoise_contract(report: dict[str, Any], pipe: Any) -> None:
    """Inspect whether one denoise step is replayable (trainable + logprob)."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        report["blockers"].append("pipeline exposes no .transformer (no denoiser handle)")
        report["questions"]["generator_exposes_trainable_denoiser"] = "no"
        return

    n_params = sum(p.numel() for p in transformer.parameters())
    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    report["denoise_tensor"]["transformer_params"] = int(n_params)
    report["denoise_tensor"]["transformer_trainable_params"] = int(n_trainable)
    report["questions"]["generator_exposes_trainable_denoiser"] = (
        "yes" if n_params > 0 else "no"
    )

    cfg = getattr(transformer, "config", None)
    if cfg is not None:
        report["denoise_tensor"]["transformer_config"] = {
            k: v
            for k, v in vars(cfg).items()
            if isinstance(v, (int, float, str, bool, list, tuple)) and not k.startswith("_")
        }
    # A flow-matching / DDPM scheduler with a known sigma schedule is the
    # precondition for a replayable logprob; record it, do not assert it here.
    sched = getattr(pipe, "scheduler", None)
    report["questions"]["replayable_denoise_logprob"] = (
        "candidate_flow_or_ddpm"
        if sched is not None and hasattr(sched, "set_timesteps")
        else "unknown"
    )


def _decide(report: dict[str, Any]) -> None:
    """Set the positive/negative/undecided decision from what was learned."""
    q = report["questions"]
    if report["artifact"] is not None and q["generator_exposes_trainable_denoiser"] == "yes":
        report["decision"] = "positive"
    elif q["generator_exposes_trainable_denoiser"] in ("no", "no_diffusers_path"):
        report["decision"] = "negative"
    else:
        report["decision"] = "undecided"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    entry = get_support_entry(_ENTRY_KEY)
    parser.add_argument("--model", default=None, help=f"override (default: {entry.path})")
    parser.add_argument(
        "--out",
        default="outputs/cosmos3_nano_probe/generator_decision.json",
        help="where to write the decision note JSON",
    )
    parser.add_argument("--prompt", default="A robot arm places a red block into a bin.")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    report = _new_report(entry, args)
    model = args.model or entry.path

    torch = _probe_torch(report)
    if torch is not None:
        pipe = _probe_diffusers_pipeline(report, model, torch)
        if pipe is not None:
            _probe_denoise_contract(report, pipe)
            # Generation is only attempted when a GPU is actually present; a 16B
            # generator on CPU is not a meaningful probe.
            if report["env"].get("cuda_available"):
                try:
                    _attempt_generation(report, pipe, args)
                except Exception as err:
                    report["blockers"].append(
                        f"generation failed: {type(err).__name__}: {err}",
                    )

    _decide(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[decision] {report['decision']}")
    for blocker in report["blockers"]:
        print(f"  blocker: {blocker}")
    print(f"[note] {out}")


def _attempt_generation(report: dict[str, Any], pipe: Any, args: argparse.Namespace) -> None:
    """Best-effort minimal clip; records the artifact path or a blocker."""
    import torch
    from diffusers.utils import export_to_video

    pipe.to("cuda")
    result = pipe(
        prompt=args.prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    frames = result.frames[0]
    clip = Path(args.out).with_suffix(".mp4")
    export_to_video(frames, str(clip), fps=8)
    report["artifact"] = str(clip)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

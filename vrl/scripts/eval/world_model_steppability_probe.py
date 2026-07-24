"""Probe: can this repo's diffusion world models be driven as a STEPPABLE,
action-conditioned environment? (world-model-as-env, Phase 0 KILL-RISK gate)

One-shot *contract* probe, NOT a training entrypoint. It answers the Phase-0
question from ``docs/sprints/parked/SPRINT_world_model_as_env.md``: before
building a ``WorldModelEnv`` whose ``step()`` runs Cosmos/Wan, confirm the models
even admit the two ingredients a steppable env needs:

  (A) Is a canonical interactive/omni generator reachable in the installed
      diffusers, or are we limited to the Cosmos 2.x backbones we already have?
  (B) Does a frame-PREFIX conditioning slot exist (obs_t -> next chunk) — can the
      denoiser be conditioned on previously *generated* frames, not just a fixed
      reference still?
  (C) Is there an extensible conditioning channel to thread an ACTION through?

Acceptance (sprint rule): a positive result is a PASS, but cleanly recording a
blocker is *also* a PASS — the only failure is leaving the question unanswered.
The probe NEVER crashes on a missing dep/weight; it records the blocker instead.
The static checks (A, B-signature, C) need NO GPU/weights and already decide the
gate; the live frame-prefix test (B-live) is gated behind ``--load-weights``.

Usage:
  python -m vrl.scripts.eval.world_model_steppability_probe \
    --out outputs/wm_as_env_probe/steppability_decision.json [--load-weights] \
    [--model nvidia/Cosmos-Predict2.5-2B] [--prefix-frames 2]
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import traceback
from pathlib import Path
from typing import Any


def _new_report() -> dict[str, Any]:
    return {
        "probe": "world_model_steppability",
        # The three Phase-0 questions, answered as we learn them.
        "questions": {
            "interactive_generator_available": "unknown",  # A: yes | no_only_2x
            "frame_prefix_conditioning_slot": "unknown",  # B: exists_unused | absent
            "action_conditioning_seam": "unknown",  # C: extensible_extra_dict | absent
        },
        "checks": {},
        "blockers": [],
        "decision": "undecided",  # go_prototype | go_prototype_partial | not_yet
        "next_step": "",
        "env": {"python": platform.python_version()},
    }


def _record_torch(report: dict[str, Any]) -> None:
    try:
        import torch

        report["env"]["torch"] = torch.__version__
        report["env"]["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:  # torch absent is itself a finding, not a crash
        report["env"]["torch"] = None
        report["blockers"].append(f"torch import failed: {exc!r}")


def _check_a_interactive_generator(report: dict[str, Any]) -> None:
    """A: is a Cosmos3/omni interactive generator pipeline present in diffusers?"""
    try:
        import diffusers
    except Exception as exc:
        report["checks"]["A_interactive_generator"] = {"status": "blocked", "error": repr(exc)}
        report["blockers"].append(f"diffusers import failed: {exc!r}")
        return
    names = dir(diffusers)
    cosmos = sorted(n for n in names if "Cosmos" in n and "Pipeline" in n)
    # A Cosmos3 / Cosmos-omni generator would surface as such a pipeline class.
    cosmos3 = sorted(n for n in names if "Cosmos3" in n or ("Cosmos" in n and "Omni" in n))
    report["checks"]["A_interactive_generator"] = {
        "diffusers_version": getattr(diffusers, "__version__", "?"),
        "cosmos_pipelines": cosmos,
        "cosmos3_or_omni_pipelines": cosmos3,
    }
    if cosmos3:
        report["questions"]["interactive_generator_available"] = "yes"
    else:
        report["questions"]["interactive_generator_available"] = "no_only_2x"
        report["blockers"].append(
            "No Cosmos3*/omni interactive generator pipeline in installed diffusers; only "
            f"2.x pipelines exist ({cosmos}). A prototype must graft onto a 2.x backbone — "
            "this is the same blocker cosmos3_nano_generator_probe recorded."
        )


def _check_b_frame_prefix(
    report: dict[str, Any], *, load_weights: bool, model: str | None, prefix_frames: int
) -> None:
    """B: does a frame-prefix conditioning slot exist (and is it currently unused)?"""
    has_slot = False
    # B-static-1: the diffusers Cosmos 2.5 pipeline's prepare_latents accepts a
    # video prefix + num_frames_in — introspectable with NO weights.
    try:
        from diffusers import Cosmos2_5_PredictBasePipeline as _Pipe

        params = list(inspect.signature(_Pipe.prepare_latents).parameters)
        has_slot = "num_frames_in" in params and "video" in params
        report["checks"]["B_prepare_latents_signature"] = {
            "params": params,
            "frame_prefix_slot": has_slot,
        }
    except Exception as exc:
        report["checks"]["B_prepare_latents_signature"] = {"status": "blocked", "error": repr(exc)}
        report["blockers"].append(
            f"could not introspect Cosmos2_5_PredictBasePipeline.prepare_latents: {exc!r}"
        )
    # B-static-2: confirm our wrapper currently leaves the slot UNUSED (num_frames_in=0),
    # so enabling a prefix is a wrapper change, not an upstream one.
    try:
        from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model

        src = inspect.getsource(CosmosPredict25Model.prepare_sampling).replace(" ", "")
        report["checks"]["B_wrapper_uses_slot"] = {
            "wrapper_passes_num_frames_in_0": "num_frames_in=0" in src,
        }
    except Exception as exc:
        report["checks"]["B_wrapper_uses_slot"] = {"status": "blocked", "error": repr(exc)}
    # B-live (gated): actually allocate a prefix-conditioned latent. Needs weights/GPU.
    if load_weights:
        _check_b_live(report, model=model, prefix_frames=prefix_frames)
    report["questions"]["frame_prefix_conditioning_slot"] = (
        "exists_unused" if has_slot else "absent"
    )


def _check_b_live(report: dict[str, Any], *, model: str | None, prefix_frames: int) -> None:
    """Best-effort live test: prepare_latents on a real frame prefix (num_frames_in>0)."""
    if not model:
        report["checks"]["B_live"] = {"status": "skipped", "reason": "no --model checkpoint given"}
        return
    try:
        import torch
        from diffusers import Cosmos2_5_PredictBasePipeline

        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        # A dummy [B, C, T_in, H, W] frame prefix standing in for obs_t.
        h, w, t_in = 480, 832, max(1, prefix_frames)
        video = torch.zeros(1, 3, t_in, h, w, device=device, dtype=torch.bfloat16)
        num_channels = pipe.transformer.config.in_channels - 1
        out = pipe.prepare_latents(
            video=video,
            batch_size=1,
            num_channels_latents=num_channels,
            height=h,
            width=w,
            num_frames_in=t_in,
            num_frames_out=t_in + 4,
            do_classifier_free_guidance=False,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(0),
            latents=None,
        )
        latents, _cond_latent, cond_mask, _cond_indicator = out
        report["checks"]["B_live"] = {
            "status": "ok",
            "accepts_frame_prefix": True,
            "latents_shape": list(latents.shape),
            "cond_mask_nonzero": bool(cond_mask.abs().sum().item() > 0),
        }
    except Exception as exc:
        report["checks"]["B_live"] = {
            "status": "blocked",
            "error": repr(exc),
            "trace": traceback.format_exc(limit=3),
        }
        report["blockers"].append(f"live frame-prefix prepare_latents failed: {exc!r}")


def _check_c_action_seam(report: dict[str, Any]) -> None:
    """C: is there an extensible conditioning channel to thread an ACTION through?"""
    import dataclasses

    has_extra = False
    try:
        from vrl.models.steps.denoise.common.backbone import DiffusionBackboneInput

        fields = {f.name: str(f.type) for f in dataclasses.fields(DiffusionBackboneInput)}
        has_extra = "extra" in fields
        report["checks"]["C_backbone_input"] = {
            "fields": list(fields),
            "extra_is_freeform_dict": has_extra,
        }
    except Exception as exc:
        report["checks"]["C_backbone_input"] = {"status": "blocked", "error": repr(exc)}
        report["blockers"].append(f"could not introspect DiffusionBackboneInput: {exc!r}")
    # Wan-I2V already threads `condition`/`image_embeds` through extra=; an `action`
    # key would be additive. The runner still has to be taught to CONSUME it.
    try:
        import vrl.models.families.wan_2_1.model as _wan

        src = inspect.getsource(_wan).replace(" ", "")
        report["checks"]["C_wan_extra_usage"] = {
            "wan_threads_condition_via_extra": 'extra={"condition"' in src
            or '"condition":' in src,
            "note": "adding extra['action'] is non-breaking, but WanI2V*BackboneRunner must be "
            "extended to consume it before it conditions generation.",
        }
    except Exception as exc:
        report["checks"]["C_wan_extra_usage"] = {"status": "blocked", "error": repr(exc)}
    report["questions"]["action_conditioning_seam"] = (
        "extensible_extra_dict" if has_extra else "absent"
    )


def _decide(report: dict[str, Any]) -> None:
    q = report["questions"]
    frame_slot = q["frame_prefix_conditioning_slot"] == "exists_unused"
    action_seam = q["action_conditioning_seam"] == "extensible_extra_dict"
    # The interactive omni generator is a BONUS, not required — a 2.x graft suffices.
    report["checks"]["interactive_generator_required"] = False
    if frame_slot and action_seam:
        report["decision"] = "go_prototype"
        report["next_step"] = (
            "Phase 1 (needs weights): drive Cosmos-Predict2.5 prepare_latents(num_frames_in=K>0) "
            "on a generated-frame prefix and graft a dummy action through DiffusionBackboneInput."
            "extra; confirm the next chunk VISIBLY responds to the action (action-fidelity gate)."
        )
    elif frame_slot or action_seam:
        report["decision"] = "go_prototype_partial"
        report["next_step"] = (
            "One of {frame-prefix slot, action seam} is reachable; the other is the blocker — "
            "see blockers[]. Resolve before Phase 1."
        )
    else:
        report["decision"] = "not_yet"
        report["next_step"] = (
            "Neither a frame-prefix slot nor an action seam is reachable on the installed models; "
            "world-model-as-env is not yet feasible here — wait for an upstream interactive "
            "(Genie-/Vid2World-style) checkpoint or a Cosmos3 diffusers pipeline."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/wm_as_env_probe/steppability_decision.json")
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="run the live frame-prefix test (needs a checkpoint + GPU)",
    )
    parser.add_argument("--model", default=None, help="checkpoint for the live B test")
    parser.add_argument("--prefix-frames", type=int, default=2)
    args = parser.parse_args(argv)

    report = _new_report()
    _record_torch(report)
    # Each check is isolated so one blocker never hides the others.
    for check in (
        lambda: _check_a_interactive_generator(report),
        lambda: _check_b_frame_prefix(
            report,
            load_weights=args.load_weights,
            model=args.model,
            prefix_frames=args.prefix_frames,
        ),
        lambda: _check_c_action_seam(report),
    ):
        try:
            check()
        except Exception as exc:  # a probe never crashes; it records.
            report["blockers"].append(f"unexpected check error: {exc!r}")
    _decide(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[world_model_steppability_probe] decision={report['decision']} -> {out}")


if __name__ == "__main__":
    main()

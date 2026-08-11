"""Shared sampling-config resolver for the video checkpoint-eval scripts.

``cosmos_predict25_kling_eval`` and ``wan_robotics_checkpoint_eval`` both projected the
same ~10-key sampling dict out of the same ``sampling.*`` / ``rollout.*`` paths of a
resolved training config. This is the single owner of that projection.

The dict is intentionally untyped: it is a per-request runtime payload handed straight to
the generation request, not a launch-time config layer object.

Note: ``sana_aesthetic_report.resolve_sampling`` is deliberately NOT routed here — it
returns the frozen ``OFFICIAL_SAMPLING_PROTOCOL`` to keep reproducibility independent
of the training SDE, and must stay separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


def resolve_eval_sampling(
    cfg: DictConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the sampling/rollout config into the runtime sampling dict.

    ``overrides`` supplies optional CLI-flag values keyed by the output field name
    (``width``, ``height``, ``num_frames``, ``num_steps``, ``fps``,
    ``max_sequence_length``, ``guidance_scale``). A falsy numeric override falls back to
    the config value; ``guidance_scale`` falls back only when the override is ``None`` so
    an explicit ``0`` (CFG disabled) is preserved. Callers without CLI overrides (wan)
    pass nothing and read the config directly.
    """
    ov = overrides or {}
    return {
        "width": int(ov.get("width") or OmegaConf.select(cfg, "sampling.width", default=512)),
        "height": int(ov.get("height") or OmegaConf.select(cfg, "sampling.height", default=512)),
        "num_frames": int(
            ov.get("num_frames") or OmegaConf.select(cfg, "sampling.num_frames", default=93),
        ),
        "num_steps": int(
            ov.get("num_steps") or OmegaConf.select(cfg, "sampling.num_steps", default=20),
        ),
        "fps": int(ov.get("fps") or OmegaConf.select(cfg, "sampling.fps", default=16)),
        "max_sequence_length": int(
            ov.get("max_sequence_length")
            or OmegaConf.select(cfg, "sampling.max_sequence_length", default=512),
        ),
        "guidance_scale": float(
            OmegaConf.select(cfg, "sampling.guidance_scale", default=1.0)
            if ov.get("guidance_scale") is None
            else ov["guidance_scale"],
        ),
        "denoise_mode": str(OmegaConf.select(cfg, "rollout.denoise_mode", default="sde")),
        "noise_level": float(OmegaConf.select(cfg, "rollout.noise_level", default=1.0)),
        "sde_type": str(OmegaConf.select(cfg, "rollout.sde.type", default="flow_grpo")),
    }

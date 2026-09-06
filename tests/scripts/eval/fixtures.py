"""Real objects shared by the SANA eval-script tests."""

from __future__ import annotations

from typing import Any

from vrl.scripts.eval.sana_inference import SCHEDULER_PROTOCOL


def build_official_sana_scheduler(**overrides: Any) -> Any:
    """The real ``DPMSolverMultistepScheduler`` at SANA's official protocol.

    Config-init, no download. An override is how the drift tests produce a
    scheduler that must be REJECTED, and because the object is genuine an
    upstream rename of any protocol key turns the accept case red instead of
    letting a double echo the table back.
    """

    from diffusers import DPMSolverMultistepScheduler

    kwargs = {key: value for key, value in SCHEDULER_PROTOCOL.items() if key != "class_name"}
    kwargs.update(overrides)
    return DPMSolverMultistepScheduler(**kwargs)


__all__ = ["build_official_sana_scheduler"]

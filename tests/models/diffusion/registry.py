"""Tiny HF diffusion pipelines (~1 MB) for real ``from_pretrained`` wiring tests.

``hf-internal-testing/tiny-*`` repos carry the full pipeline layout
(model_index.json + transformer/vae/text_encoder/scheduler) with random weights,
so loading one via the real ``from_pretrained`` path pins pipeline *assembly*
wiring cheaply enough for per-PR CI (numerics stay in the gated real-checkpoint
e2e suite). Cosmos / Wan-I2V have no such repo on the Hub, so they are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TinyPipelineSpec:
    """Pinned tiny HF pipeline fixture."""

    repo: str
    revision: str


_TINY_PIPELINES = {
    "wan-t2v": TinyPipelineSpec(
        repo="hf-internal-testing/tiny-wan-pipe",
        revision="8a97b5eb008dd10f4ac0d3f55d41f4acb3d65021",
    ),
    "sd3": TinyPipelineSpec(
        repo="hf-internal-testing/tiny-sd3-pipe",
        revision="309fbef85960712c84ff0793d3594951235fff9b",
    ),
}


class TinyPipelineUnavailable(RuntimeError):
    """Raised when a tiny pipeline cannot be fetched (offline / Hub down)."""


def archs_with_tiny_pipe() -> tuple[str, ...]:
    """Architecture keys that have a ~1 MB tiny HF pipeline (for parametrize)."""

    return tuple(_TINY_PIPELINES)


def load_tiny_pipeline(name: str, **from_pretrained: Any) -> Any:
    """Load a tiny ``hf-internal-testing`` pipeline via the real ``from_pretrained``.

    Uses ``DiffusionPipeline.from_pretrained`` (diffusers' by-name auto-loader,
    which reads model_index.json and assembles the right pipeline class), so this
    exercises the production loading/assembly path on a ~1 MB random-weight repo.
    Raises :class:`TinyPipelineUnavailable` on network failure so callers can skip
    cleanly in offline CI.
    """

    spec = _TINY_PIPELINES[name]
    from_pretrained.setdefault("torch_dtype", torch.float32)
    from_pretrained.setdefault("revision", spec.revision)

    import requests.exceptions
    from diffusers import DiffusionPipeline

    try:
        return DiffusionPipeline.from_pretrained(spec.repo, **from_pretrained)
    except (requests.exceptions.RequestException, OSError) as exc:
        raise TinyPipelineUnavailable(f"{spec.repo}: {exc}") from exc

"""Shared helpers for the config tests."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from vrl.config.schema import parse_config


def unknown_keys(cfg: Any) -> list[str]:
    """Dotted paths ``parse_config`` rejects as unknown keys (sorted), or ``[]``.

    Every other parse failure is re-raised: this helper only answers the
    unknown-key question, the way the old tree walker did.
    """

    try:
        parse_config(cfg)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("unknown ") and "; expected" not in message:
            return sorted(message[len("unknown ") :].split(", "))
        raise
    return []


def minimal_grpo_cfg(**overrides):
    """The smallest GRPO root the schema parses; ``overrides`` replace top-level sections."""

    base = {
        "algorithm": {"kind": "grpo"},
        "data": {
            "loader": "prompt_manifest",
            "manifest": "manifests/ocr/train.txt",
            "preprocessing": {"format": "text"},
            "sampler": {"type": "random_without_replacement"},
        },
        "rollout": {"sde": {"type": "flow_grpo"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)

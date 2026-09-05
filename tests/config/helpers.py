"""Shared helpers for the config tests."""

from __future__ import annotations

from typing import Any

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

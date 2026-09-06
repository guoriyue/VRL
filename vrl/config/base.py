"""Shared Pydantic contract for public configuration sections."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigBase(BaseModel):
    """Typed public section: every key is a declared field.

    ``extra="forbid"`` is the whole unknown-key mechanism — a typo, a removed
    key, and a never-seen key all fail at ``parse_config`` with the same
    ``unknown <dotted.path>`` message (see ``_extract_error_message``).
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def revalidate[SectionT: ConfigBase](
        cls: type[SectionT],
        payload: Any,
        *,
        section: str,
    ) -> SectionT:
        """Validate a bare section payload and re-anchor errors to its YAML path.

        A family-selected ``model``/``sampling`` class validates its bare payload;
        on failure the error is re-prefixed so callers still receive the public
        ``<section>.<field>`` location instead of the bare field name.
        """

        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(_extract_error_message(exc, section=section)) from exc


_UNKNOWN_KEY_ERRORS = ("extra_forbidden", "unexpected_keyword_argument")


def _extract_error_message(exc: ValidationError, *, section: str = "") -> str:
    """Extract a clean ValueError message from a Pydantic ValidationError.

    ``section`` re-anchors a bare section payload (a family-selected ``model``
    or ``sampling`` class validated on its own) to its public YAML path, so the
    location always reads ``<section>.<field>``.

    Unknown keys take precedence and are reported all at once, sorted — a typo,
    a removed key, and a never-seen key get the same ``unknown a.b, c.d`` line
    whether pydantic saw them as ``extra_forbidden`` on a model, as an
    unexpected keyword on a stdlib dataclass field, or already folded into a
    nested section's own message. Otherwise the first error is reported.
    """
    errors = exc.errors(include_url=False)

    def location(error: Any) -> str:
        # A union field (``teacache: bool | TeaCacheSection``) reports the
        # variant's class name as a loc segment; YAML keys are snake_case, so a
        # CapWords segment is never a key and only hides the real path.
        parts = [str(p) for p in error["loc"] if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", str(p))]
        if section:
            parts.insert(0, section)
        return ".".join(parts)

    unknown: set[str] = set()
    for error in errors:
        if error["type"] in _UNKNOWN_KEY_ERRORS:
            unknown.add(location(error))
        elif error["msg"].startswith("Value error, unknown ") and "; expected" not in error["msg"]:
            unknown.update(error["msg"][len("Value error, unknown ") :].split(", "))
    if unknown:
        return "unknown " + ", ".join(sorted(unknown))

    first = errors[0]
    error_type = first["type"]
    msg = first["msg"]
    loc = location(first)
    # Missing required field — remap to repo-standard message format
    if error_type == "missing":
        return f"config missing required field: {loc}"
    # Literal enum mismatch — reformat to "unknown {loc}={input!r}; expected ..."
    if error_type == "literal_error":
        input_val = first.get("input", "")
        expected = msg.replace("Input should be", "expected")
        return f"unknown {loc}={input_val!r}; {expected}"
    # ValueError raised inside a validator — strip Pydantic's "Value error, " prefix
    # (validators name the offending path themselves).
    if msg.startswith("Value error, "):
        return msg[len("Value error, ") :]
    # Type errors: pydantic's sentence plus the path it lost.
    return f"{loc}: {msg}" if loc else msg


__all__ = ["ConfigBase", "_extract_error_message"]

"""Whole-tree unknown-key reporting for resolved training configs.

One walker, one entry point, every depth: each mapping level of the config is
compared against that level's known-key set, which is DERIVED from the type
that actually consumes it (pydantic model fields or dataclass fields) instead
of a hand-maintained list. An old removed key, a typo, and a never-seen key
all get the same treatment — one warning naming the full dotted path.

Every block in the tree follows one rule, in priority order:
  1. a typed class consumes it        -> point at the class (keys derived)
  2. no class, but the key set is
     closed (code reads fixed keys)   -> hand list + comment naming the reader
  3. key NAMES are arbitrary by design
     (user-chosen names)              -> OPEN — the only case OPEN is allowed

All three rules are expressed AS FIELD DECLARATIONS in vrl.config.schema —
a field typed as a model/dataclass nests automatically (rule 1); a closed
dict field carries ``Annotated[..., ConfigBlock(("k1", "k2"))]`` naming its
reader (rule 2); an open dict carries ``Annotated[..., OPEN]`` (rule 3).
This module is pure mechanism with zero repo-specific names. Hand key lists
cannot rot silently: the AST sweep in tests/config/test_unknown_keys.py
cross-checks code reads against the derived tree on every CI run.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Callable, Mapping
from typing import Any

from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# Sentinel: this block accepts arbitrary keys by design — do not descend.
OPEN = object()


def _block_from(candidate: Any) -> Any:
    """Map an annotation/metadata object to a child block, or None."""

    if candidate is OPEN or isinstance(candidate, ConfigBlock):
        return candidate
    if hasattr(candidate, "model_fields") or dataclasses.is_dataclass(candidate):
        return ConfigBlock(candidate)
    return None


def _model_children(source: Any) -> dict[str, Any]:
    """Auto-derive child blocks from field declarations.

    pydantic models: a field typed as a model/dataclass nests automatically;
    ``Annotated[..., ConfigBlock(...)]`` / ``Annotated[..., OPEN]`` metadata
    declares the block spec next to the field it describes.
    dataclasses: fields typed as dataclasses/models nest automatically
    (e.g. RolloutOrchestrationConfig.continuous -> ContinuousRolloutConfig)."""

    children: dict[str, Any] = {}
    if hasattr(source, "model_fields"):
        for name, field in source.model_fields.items():
            for meta in getattr(field, "metadata", ()):
                block = _block_from(meta)
                if block is not None:
                    children[name] = block
                    break
            else:
                for candidate in typing.get_args(field.annotation) or (field.annotation,):
                    block = _block_from(candidate)
                    if block is not None:
                        children[name] = block
                        break
    elif dataclasses.is_dataclass(source):
        try:
            hints = typing.get_type_hints(source)
        except Exception:  # unresolvable forward refs — skip nesting
            hints = {}
        for field in dataclasses.fields(source):
            annotation = hints.get(field.name, field.type)
            for candidate in typing.get_args(annotation) or (annotation,):
                block = _block_from(candidate)
                if block is not None:
                    children[field.name] = block
                    break
    return children


class ConfigBlock:
    """One YAML block: its known keys plus the blocks nested inside it.

    Children are auto-derived for pydantic-model-typed fields; an explicit
    ``children`` entry overrides the auto one. ``open_keys=True`` accepts
    arbitrary keys at this level while still descending into known children."""

    def __init__(
        self,
        source: Any,
        children: dict[str, Any] | None = None,
        *,
        open_keys: bool = False,
        select: Callable[[Mapping[str, Any]], Any] | None = None,
        variants: tuple[Any, ...] = (),
    ) -> None:
        # Known keys derive from the type that consumes the block (pydantic
        # model or dataclass); an iterable is the hand-list escape hatch.
        if hasattr(source, "model_fields"):
            self.known = frozenset(source.model_fields)
        elif dataclasses.is_dataclass(source):
            self.known = frozenset(f.name for f in dataclasses.fields(source))
        else:
            self.known = frozenset(source)
        self.children = {**_model_children(source), **(children or {})}
        self.open_keys = open_keys
        self.select = select
        self.variants = tuple(
            variant if isinstance(variant, ConfigBlock) else ConfigBlock(variant)
            for variant in variants
        )


@functools.lru_cache(maxsize=1)
def _root_block() -> ConfigBlock:
    """The whole config surface, derived from RootConfig's field declarations.

    No repo-specific names live in this module: section models, nested blocks,
    hand key lists, and OPEN boundaries are all declared as field types /
    ``Annotated`` metadata in vrl.config.schema, next to the fields they
    describe. (Local import avoids a cycle: schema imports ConfigBlock/OPEN.)
    """

    from vrl.config.schema import RootConfig

    return ConfigBlock(RootConfig)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    try:  # OmegaConf DictConfig
        from omegaconf import DictConfig

        if isinstance(value, DictConfig):
            return value
    except Exception:  # pragma: no cover
        pass
    return None


def _collect_unknown(value: Any, block: Any, path: str, out: list[str]) -> None:
    from omegaconf import DictConfig, OmegaConf

    if block is OPEN:
        return
    mapping = _as_mapping(value)
    if mapping is None:  # scalar where a block may also be legal (precision)
        return
    if not isinstance(block, ConfigBlock):
        return
    if block.select is not None:
        selected = block.select(mapping)
        if selected is not None and selected is not block:
            _collect_unknown(value, selected, path, out)
            return
    for key in mapping:
        key = str(key)
        child_path = f"{path}.{key}" if path else key
        if not block.open_keys and key not in block.known and key not in block.children:
            out.append(child_path)
            continue
        child_block = block.children.get(key)
        if child_block is not None:
            # Structural lint checks names before runtime composition supplies
            # mandatory blocks. Reading their values here would resolve ???.
            if isinstance(mapping, DictConfig) and OmegaConf.is_missing(mapping, key):
                continue
            _collect_unknown(mapping[key], child_block, child_path, out)


def find_unknown_keys(cfg: Any, *, section: str | None = None) -> list[str]:
    """Return dotted paths of unknown keys anywhere in ``cfg``.

    ``section`` checks one top-level section subtree (e.g. ``"reward"``).
    """

    root = _root_block()
    out: list[str] = []
    if section is not None:
        _collect_unknown(cfg, root.children[section], section, out)
        return sorted(out)
    _collect_unknown(cfg, root, "", out)
    return sorted(out)


def require_no_unknown_keys(cfg: Any, *, section: str | None = None) -> None:
    """Fail loud on every unknown key path in closed blocks.

    One mechanism covers typos, deleted keys, and historical renames alike: a
    typed closed block accepts exactly its fields, an OPEN block accepts
    anything. A warning here would let a stale key silently fall back to the
    default behavior, which is worse than failing.
    """

    unknown = find_unknown_keys(cfg, section=section)
    if unknown:
        raise ValueError("unknown " + ", ".join(unknown))


__all__ = ["OPEN", "find_unknown_keys", "require_no_unknown_keys"]

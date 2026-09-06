"""Immutable model identity used by strict training-checkpoint restore.

Public model schemas classify their fields with ``checkpoint_identity_metadata``.
This module interprets that metadata; it deliberately contains no family-name
table, so the selected typed schema remains the single source of truth.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic.fields import PydanticUndefined

if TYPE_CHECKING:
    from vrl.models.interfaces.runtime import ModelBuild

MODEL_IDENTITY_SCHEMA = "vrl.model-identity/v1"
CHECKPOINT_IDENTITY_METADATA_KEY = "checkpoint_identity"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MISSING = object()

IdentityKind = Literal[
    "exclude",
    "lora",
    "member",
    "source",
    "source_revision",
    "value",
]
_IDENTITY_KINDS = frozenset(get_args(IdentityKind))


@dataclass(frozen=True, slots=True)
class LocalCheckpointContent:
    """Path-free content facts for one local checkpoint source."""

    kind: Literal["file", "tree"]
    sha256: str
    bytes: int
    files: int


def checkpoint_identity_metadata(
    kind: IdentityKind,
    *,
    source: str | None = None,
    revision_field: str | None = None,
    overridden_by: str | None = None,
    enabled_by: str | None = None,
    members_only: bool = False,
    omit_for_source_file: bool = False,
    required: bool = False,
    canonicalize: Literal["sorted_unique"] | None = None,
    default: Any = _MISSING,
) -> dict[str, Any]:
    """Declare one typed model field's role in checkpoint identity."""

    if kind not in _IDENTITY_KINDS:
        raise ValueError(f"unknown checkpoint identity metadata kind: {kind!r}")
    payload: dict[str, Any] = {"kind": kind}
    if source is not None:
        payload["source"] = source
    if revision_field is not None:
        payload["revision_field"] = revision_field
    if overridden_by is not None:
        payload["overridden_by"] = overridden_by
    if enabled_by is not None:
        payload["enabled_by"] = enabled_by
    if members_only:
        payload["members_only"] = True
    if omit_for_source_file:
        payload["omit_for_source_file"] = True
    if required:
        payload["required"] = True
    if canonicalize is not None:
        payload["canonicalize"] = canonicalize
    if default is not _MISSING:
        payload["default"] = default
    return {CHECKPOINT_IDENTITY_METADATA_KEY: payload}


def _field_metadata(field: Any, *, schema_name: str, field_name: str) -> dict[str, Any]:
    extra = field.json_schema_extra
    if not isinstance(extra, Mapping):
        raise TypeError(
            f"{schema_name}.{field_name} must declare checkpoint identity metadata",
        )
    metadata = extra.get(CHECKPOINT_IDENTITY_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"{schema_name}.{field_name} must declare checkpoint identity metadata",
        )
    result = dict(metadata)
    kind = result.get("kind")
    if kind not in _IDENTITY_KINDS:
        raise TypeError(
            f"{schema_name}.{field_name} has invalid checkpoint identity kind {kind!r}",
        )
    return result


def validate_checkpoint_identity_schema(schema_cls: type[Any]) -> None:
    """Fail closed when a concrete public model schema has an unclassified field."""

    fields = getattr(schema_cls, "model_fields", None)
    if not isinstance(fields, Mapping):
        raise TypeError(f"{schema_cls!r} is not a Pydantic model schema")

    metadata_by_field = {
        name: _field_metadata(
            field,
            schema_name=schema_cls.__name__,
            field_name=name,
        )
        for name, field in fields.items()
    }
    sources: dict[str, str] = {}
    revision_sources: dict[str, str] = {}
    for field_name, metadata in metadata_by_field.items():
        kind = metadata["kind"]
        allowed = {
            "exclude": {"kind"},
            "lora": {"enabled_by", "kind"},
            "member": {
                "kind",
                "omit_for_source_file",
                "overridden_by",
                "source",
            },
            "source": {"kind", "members_only", "revision_field", "source"},
            "source_revision": {"kind", "source"},
            "value": {"canonicalize", "default", "kind", "required"},
        }[kind]
        unknown = sorted(set(metadata) - allowed)
        if unknown:
            raise TypeError(
                f"{schema_cls.__name__}.{field_name} has unsupported checkpoint "
                f"identity metadata key(s): {', '.join(unknown)}",
            )

        if kind in {"member", "source", "source_revision"}:
            source = metadata.get("source")
            if not isinstance(source, str) or not source:
                raise TypeError(
                    f"{schema_cls.__name__}.{field_name} checkpoint identity "
                    f"{kind} requires a non-empty source",
                )
        if kind == "source":
            source = metadata["source"]
            if source in sources:
                raise TypeError(
                    f"{schema_cls.__name__} declares checkpoint source {source!r} "
                    f"on both {sources[source]!r} and {field_name!r}",
                )
            sources[source] = field_name
            revision_field = metadata.get("revision_field")
            if revision_field is not None and revision_field not in fields:
                raise TypeError(
                    f"{schema_cls.__name__}.{field_name} references missing revision "
                    f"field {revision_field!r}",
                )
        elif kind == "source_revision":
            revision_sources[field_name] = metadata["source"]
        elif kind == "lora":
            enabled_by = metadata.get("enabled_by")
            enabled_metadata = metadata_by_field.get(enabled_by)
            if (
                not isinstance(enabled_by, str)
                or enabled_metadata is None
                or enabled_metadata["kind"] != "value"
            ):
                raise TypeError(
                    f"{schema_cls.__name__}.{field_name} checkpoint identity lora "
                    "requires enabled_by naming a value field",
                )

    referenced_revisions: set[str] = set()
    for source, field_name in sources.items():
        metadata = metadata_by_field[field_name]
        revision_field = metadata.get("revision_field")
        if revision_field is not None:
            revision_metadata = metadata_by_field[revision_field]
            if (
                revision_metadata["kind"] != "source_revision"
                or revision_metadata.get("source") != source
            ):
                raise TypeError(
                    f"{schema_cls.__name__}.{revision_field} must be the "
                    f"source_revision for {source!r}",
                )
            referenced_revisions.add(revision_field)

    unreferenced = sorted(set(revision_sources) - referenced_revisions)
    if unreferenced:
        raise TypeError(
            f"{schema_cls.__name__} has unreferenced checkpoint source revision "
            f"field(s): {', '.join(unreferenced)}",
        )

    for field_name, metadata in metadata_by_field.items():
        if metadata["kind"] != "member":
            continue
        source = metadata["source"]
        if source not in sources:
            raise TypeError(
                f"{schema_cls.__name__}.{field_name} references unknown checkpoint "
                f"source {source!r}",
            )
        overridden_by = metadata.get("overridden_by")
        if overridden_by is not None:
            override_metadata = metadata_by_field.get(overridden_by)
            if override_metadata is None or override_metadata["kind"] != "source":
                raise TypeError(
                    f"{schema_cls.__name__}.{field_name} override {overridden_by!r} "
                    "must be a checkpoint source field",
                )
    member_sources = {
        metadata["source"]
        for metadata in metadata_by_field.values()
        if metadata["kind"] == "member"
    }
    invalid_members_only = sorted(
        source
        for source, field_name in sources.items()
        if metadata_by_field[field_name].get("members_only") and source not in member_sources
    )
    if invalid_members_only:
        raise TypeError(
            f"{schema_cls.__name__} checkpoint source(s) marked members_only have "
            f"no member fields: {', '.join(invalid_members_only)}",
        )


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _raise_changed(path: Path) -> None:
    raise RuntimeError(f"local checkpoint source changed while hashing: {path}")


def _hash_regular_file(digest: Any, path: Path, relative: str) -> int:
    before_path = path.stat()
    total_bytes = 0
    try:
        with path.open("rb") as handle:
            before_file = os.fstat(handle.fileno())
            if _stat_signature(before_path) != _stat_signature(before_file):
                _raise_changed(path)
            digest.update(b"file\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(before_file.st_size).encode("ascii"))
            digest.update(b"\0")
            while payload := handle.read(1024 * 1024):
                digest.update(payload)
                total_bytes += len(payload)
            digest.update(b"\0")
            after_file = os.fstat(handle.fileno())
    except OSError as exc:
        raise RuntimeError(f"cannot read local checkpoint source file {path}: {exc}") from exc
    try:
        after_path = path.stat()
    except OSError:
        _raise_changed(path)
    if (
        _stat_signature(before_file) != _stat_signature(after_file)
        or _stat_signature(after_file) != _stat_signature(after_path)
        or total_bytes != after_file.st_size
    ):
        _raise_changed(path)
    return total_bytes


def _hash_local_node(
    digest: Any,
    path: Path,
    relative: str,
    directory_stack: set[tuple[int, int]],
) -> tuple[int, int]:
    try:
        link_before = path.lstat()
        target_before = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot resolve local checkpoint source {path}: {exc}") from exc

    mode = target_before.st_mode
    if stat.S_ISREG(mode):
        total_bytes = _hash_regular_file(digest, path, relative)
        total_files = 1
    elif stat.S_ISDIR(mode):
        identity = (target_before.st_dev, target_before.st_ino)
        if identity in directory_stack:
            raise RuntimeError(f"local checkpoint source contains a symlink cycle at {path}")
        directory_stack.add(identity)
        digest.update(b"dir\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            children = sorted(path.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise RuntimeError(
                f"cannot enumerate local checkpoint source directory {path}: {exc}",
            ) from exc
        total_bytes = 0
        total_files = 0
        for child in children:
            child_relative = child.name if not relative else f"{relative}/{child.name}"
            child_bytes, child_files = _hash_local_node(
                digest,
                child,
                child_relative,
                directory_stack,
            )
            total_bytes += child_bytes
            total_files += child_files
        directory_stack.remove(identity)
    else:
        raise RuntimeError(
            f"local checkpoint source contains unsupported special file: {path}",
        )

    try:
        link_after = path.lstat()
        target_after = path.stat()
    except OSError:
        _raise_changed(path)
    if _stat_signature(link_before) != _stat_signature(link_after) or _stat_signature(
        target_before
    ) != _stat_signature(target_after):
        _raise_changed(path)
    return total_bytes, total_files


def local_checkpoint_content(path: str | Path) -> LocalCheckpointContent:
    """Describe one local file/tree without embedding its absolute root path.

    Symlinks are followed as content aliases. Broken links, cycles, special
    files, and sources that mutate during the read fail closed.
    """

    source = Path(path).expanduser()
    try:
        source.lstat()
    except OSError as exc:
        raise RuntimeError(f"local checkpoint source does not exist: {source}") from exc
    try:
        mode = source.stat().st_mode
    except OSError as exc:
        raise RuntimeError(f"cannot resolve local checkpoint source {source}: {exc}") from exc
    if stat.S_ISREG(mode):
        kind: Literal["file", "tree"] = "file"
    elif stat.S_ISDIR(mode):
        kind = "tree"
    else:
        raise RuntimeError(f"local checkpoint source is an unsupported special file: {source}")
    digest = hashlib.sha256()
    total_bytes, total_files = _hash_local_node(digest, source, "", set())
    return LocalCheckpointContent(
        kind=kind,
        sha256=digest.hexdigest(),
        bytes=total_bytes,
        files=total_files,
    )


def _normalize_identity_value(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"checkpoint identity field {field_name} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item, field_name=field_name) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(
                f"checkpoint identity field {field_name} mapping keys must be strings",
            )
        return {
            key: _normalize_identity_value(item, field_name=field_name)
            for key, item in sorted(value.items())
        }
    raise TypeError(
        f"checkpoint identity field {field_name} has unsupported "
        f"value type {type(value).__name__}",
    )


def _normalize_identity_field(
    value: Any,
    *,
    field_name: str,
    metadata: Mapping[str, Any],
) -> Any:
    normalized = _normalize_identity_value(value, field_name=field_name)
    canonicalize = metadata.get("canonicalize")
    if canonicalize is None:
        return normalized
    if canonicalize != "sorted_unique":
        raise TypeError(
            f"checkpoint identity field {field_name} has unknown canonicalizer {canonicalize!r}",
        )
    if not isinstance(normalized, list) or not all(isinstance(item, str) for item in normalized):
        raise TypeError(
            f"checkpoint identity field {field_name} sorted_unique "
            "canonicalization requires a string sequence",
        )
    return sorted(set(normalized))


def _is_configured(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def require_checkpoint_source_member(value: Any, *, field_name: str) -> str:
    """Return one safe POSIX-relative member of a checkpoint source.

    Local directories and Hugging Face repositories share this contract. An
    artifact outside a source root is a separate source and must use its own
    explicit path field instead of escaping through ``..`` or an absolute path.
    """

    member = str(value)
    if not member or member != member.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path")
    if "\\" in member:
        raise ValueError(f"{field_name} must use POSIX '/' path separators")
    parts = member.split("/")
    if (
        member.startswith("/")
        or PureWindowsPath(member).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(
            f"{field_name} must stay within its checkpoint source "
            "(absolute paths and '.', '..', or empty segments are not allowed)",
        )
    return member


def require_remote_checkpoint_source_pin(
    reference: Any,
    revision: Any,
    *,
    field_name: str,
) -> tuple[str, str] | None:
    """Return a pinned Hugging Face source, or ``None`` for an existing local path."""

    text = str(reference).strip()
    if not text:
        raise ValueError(f"checkpoint source field {field_name} must be non-empty")
    candidate = Path(text).expanduser()
    if candidate.exists() or candidate.is_symlink():
        return None

    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(text)
    except HFValidationError as exc:
        raise ValueError(
            f"checkpoint source {field_name} is neither an existing local path "
            f"nor a valid Hugging Face repo id: {text!r}",
        ) from exc
    revision_text = "" if revision is None else str(revision).strip()
    if _COMMIT_RE.fullmatch(revision_text) is None:
        raise ValueError(
            f"remote checkpoint source {field_name} requires a full lowercase "
            "40-character commit revision",
        )
    return text, revision_text


def _resolve_source(
    reference: Any,
    revision: Any,
    *,
    field_name: str,
    local_resolver: Callable[[Path], LocalCheckpointContent],
    content_cache: dict[Path, LocalCheckpointContent],
) -> tuple[dict[str, Any], bool]:
    text = str(reference).strip()
    if not text:
        raise ValueError(f"checkpoint source field {field_name} must be non-empty")
    candidate = Path(text).expanduser()
    if candidate.exists() or candidate.is_symlink():
        try:
            alias_before = candidate.lstat()
            target_before = candidate.stat()
            canonical = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"cannot resolve local checkpoint source {field_name}={text!r}: {exc}",
            ) from exc
        content = content_cache.get(canonical)
        if content is None:
            # Hash through the configured alias, not only its resolved target:
            # the hasher then observes a root symlink throughout the read.
            content = local_resolver(candidate)
            if not isinstance(content, LocalCheckpointContent):
                raise TypeError(
                    "local checkpoint resolver must return LocalCheckpointContent",
                )
            checksum = content.sha256.lower()
            if _SHA256_RE.fullmatch(checksum) is None:
                raise ValueError(
                    f"local digest resolver returned invalid SHA256 for {field_name}: "
                    f"{checksum!r}",
                )
            if content.kind not in ("file", "tree"):
                raise ValueError(
                    f"local checkpoint resolver returned invalid kind: {content.kind!r}",
                )
            if content.bytes < 0 or content.files < 0:
                raise ValueError("local checkpoint resolver returned negative content counts")
            if content.kind == "file" and content.files != 1:
                raise ValueError("local file checkpoint identity must report exactly one file")
            content = LocalCheckpointContent(
                kind=content.kind,
                sha256=checksum,
                bytes=content.bytes,
                files=content.files,
            )
        expected_kind = (
            "file"
            if stat.S_ISREG(target_before.st_mode)
            else "tree"
            if stat.S_ISDIR(target_before.st_mode)
            else None
        )
        if content.kind != expected_kind:
            raise ValueError(
                "local checkpoint resolver content kind does not match source "
                f"{field_name}: expected {expected_kind!r}, got {content.kind!r}",
            )
        try:
            alias_after = candidate.lstat()
            target_after = candidate.stat()
            canonical_after = candidate.resolve(strict=True)
        except OSError:
            _raise_changed(candidate)
        if (
            _stat_signature(alias_before) != _stat_signature(alias_after)
            or _stat_signature(target_before) != _stat_signature(target_after)
            or canonical != canonical_after
        ):
            _raise_changed(candidate)
        content_cache.setdefault(canonical, content)
        return {
            "kind": f"local-{content.kind}",
            "sha256": content.sha256,
            "bytes": content.bytes,
            "files": content.files,
        }, content.kind == "file"

    remote = require_remote_checkpoint_source_pin(
        text,
        revision,
        field_name=field_name,
    )
    if remote is None:
        raise RuntimeError(
            f"checkpoint source {field_name} became local while resolving identity",
        )
    repo_id, revision_text = remote
    return {
        "kind": "huggingface",
        "repo_id": repo_id,
        "revision": revision_text,
    }, False


def _value_for(
    field_name: str,
    field: Any,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Any:
    if field_name in values:
        return values[field_name]
    if "default" in metadata:
        return metadata["default"]
    default = field.default
    if default is not PydanticUndefined:
        return default
    return _MISSING


def _resolve_token_lora_values(
    resolved_config: Any,
    *,
    lora_schema: type[Any],
) -> dict[str, Any]:
    """Resolve partial public LoRA overrides through the token family's config.

    Token-family config dataclasses own their LoRA defaults. Calling the same
    registry-backed projection used by model construction keeps checkpoint
    identity equal for an omitted default and the same value written explicitly.
    """

    values: dict[str, Any] = {}
    for field_name, field in lora_schema.model_fields.items():
        metadata = _field_metadata(
            field,
            schema_name=lora_schema.__name__,
            field_name=field_name,
        )
        if metadata["kind"] != "value":
            continue
        attribute = f"lora_{field_name}"
        if not hasattr(resolved_config, attribute):
            raise TypeError(
                f"{type(resolved_config).__name__} must expose {attribute!r} for "
                "checkpoint identity",
            )
        values[field_name] = getattr(resolved_config, attribute)
    return values


def resolve_checkpoint_model_identity(
    build: ModelBuild,
    *,
    local_resolver: Callable[[Path], LocalCheckpointContent] = local_checkpoint_content,
) -> dict[str, Any]:
    """Resolve one path-independent identity from a validated ``ModelBuild``."""

    from vrl.models.families.registry import TokenFamilyBuild, get_model_family_entry
    from vrl.utils.config import import_from_path

    entry = get_model_family_entry(str(build.family))
    schema_cls = import_from_path(entry.model_section_cls)
    validate_checkpoint_identity_schema(schema_cls)
    from vrl.config.model_schema import LoraSection

    validate_checkpoint_identity_schema(LoraSection)

    model_config = dict(getattr(build, "model_config", None) or {})
    values: dict[str, Any] = {
        **model_config,
        "family": str(build.family),
        "path": build.model_name_or_path,
        "revision": getattr(build, "revision", None),
    }
    metadata_by_field = {
        name: _field_metadata(
            field,
            schema_name=schema_cls.__name__,
            field_name=name,
        )
        for name, field in schema_cls.model_fields.items()
    }
    recipe = entry.family_build
    resolved_token_config = None
    if isinstance(recipe, TokenFamilyBuild):
        # Project through the registry-declared builder model construction uses,
        # so identity sees the family's effective config, not the raw build.
        projected = import_from_path(recipe.config_builder)(build)
        resolved_token_config = import_from_path(recipe.config_cls)(**projected)
        # Family config dataclasses own effective defaults for their public
        # source/member/value fields. The public field name is deliberately the
        # same attribute name, so this remains registry-driven and table-free.
        for field_name in schema_cls.model_fields:
            if hasattr(resolved_token_config, field_name):
                values[field_name] = getattr(resolved_token_config, field_name)

    active_members: dict[str, list[tuple[str, Any, dict[str, Any]]]] = {}
    for field_name, field in schema_cls.model_fields.items():
        metadata = metadata_by_field[field_name]
        if metadata["kind"] != "member":
            continue
        value = _value_for(field_name, field, values, metadata)
        overridden_by = metadata.get("overridden_by")
        if value is _MISSING or not _is_configured(value):
            continue
        if overridden_by is not None and _is_configured(values.get(overridden_by)):
            continue
        active_members.setdefault(metadata["source"], []).append(
            (field_name, value, metadata),
        )

    sources: dict[str, dict[str, Any]] = {}
    source_is_file: dict[str, bool] = {}
    content_cache: dict[Path, LocalCheckpointContent] = {}
    for field_name, field in schema_cls.model_fields.items():
        metadata = metadata_by_field[field_name]
        if metadata["kind"] != "source":
            continue
        source_name = metadata["source"]
        value = _value_for(field_name, field, values, metadata)
        if value is _MISSING or not _is_configured(value):
            continue
        if metadata.get("members_only") and not active_members.get(source_name):
            continue
        revision_field = metadata.get("revision_field")
        revision = values.get(revision_field) if revision_field is not None else None
        source, is_file = _resolve_source(
            value,
            revision,
            field_name=f"model.{field_name}",
            local_resolver=local_resolver,
            content_cache=content_cache,
        )
        sources[source_name] = source
        source_is_file[source_name] = is_file

    build_values: dict[str, Any] = {}
    for source_name, members in active_members.items():
        if source_name not in sources:
            continue
        for field_name, value, metadata in members:
            if metadata.get("omit_for_source_file") and source_is_file[source_name]:
                continue
            value = require_checkpoint_source_member(
                value,
                field_name=f"model.{field_name}",
            )
            build_values[field_name] = _normalize_identity_field(
                value,
                field_name=f"model.{field_name}",
                metadata=metadata,
            )

    for field_name, field in schema_cls.model_fields.items():
        metadata = metadata_by_field[field_name]
        kind = metadata["kind"]
        if kind == "value":
            value = _value_for(field_name, field, values, metadata)
            if value is _MISSING or value is None:
                if metadata.get("required"):
                    raise ValueError(
                        f"checkpoint identity field model.{field_name} is required",
                    )
                continue
            build_values[field_name] = _normalize_identity_field(
                value,
                field_name=f"model.{field_name}",
                metadata=metadata,
            )
        elif kind == "lora":
            enabled_by = metadata["enabled_by"]
            enabled_field = schema_cls.model_fields[enabled_by]
            enabled_metadata = metadata_by_field[enabled_by]
            enabled = _value_for(
                enabled_by,
                enabled_field,
                values,
                enabled_metadata,
            )
            if enabled is _MISSING or not bool(enabled):
                continue
            if isinstance(recipe, TokenFamilyBuild):
                assert resolved_token_config is not None
                raw_lora = _resolve_token_lora_values(
                    resolved_token_config,
                    lora_schema=LoraSection,
                )
            else:
                raw_lora = values.get(field_name)
                if not isinstance(raw_lora, Mapping):
                    raise ValueError("model.use_lora=true requires a model.lora mapping")
            lora_values: dict[str, Any] = {}
            for lora_name, lora_field in LoraSection.model_fields.items():
                lora_metadata = _field_metadata(
                    lora_field,
                    schema_name=LoraSection.__name__,
                    field_name=lora_name,
                )
                if lora_metadata["kind"] != "value":
                    continue
                value = _value_for(
                    lora_name,
                    lora_field,
                    raw_lora,
                    lora_metadata,
                )
                if value is _MISSING or value is None:
                    if lora_metadata.get("required"):
                        raise ValueError(
                            f"checkpoint identity field model.lora.{lora_name} "
                            "is required when LoRA is enabled",
                        )
                    continue
                lora_values[lora_name] = _normalize_identity_field(
                    value,
                    field_name=f"model.lora.{lora_name}",
                    metadata=lora_metadata,
                )
            build_values["lora"] = lora_values

    return {
        "schema": MODEL_IDENTITY_SCHEMA,
        "sources": dict(sorted(sources.items())),
        "build": dict(sorted(build_values.items())),
    }


__all__ = [
    "CHECKPOINT_IDENTITY_METADATA_KEY",
    "MODEL_IDENTITY_SCHEMA",
    "LocalCheckpointContent",
    "checkpoint_identity_metadata",
    "local_checkpoint_content",
    "require_checkpoint_source_member",
    "require_remote_checkpoint_source_pin",
    "resolve_checkpoint_model_identity",
    "validate_checkpoint_identity_schema",
]

"""verl/Hydra-style YAML loader with ``defaults:`` overlay."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

from omegaconf import DictConfig, ListConfig, OmegaConf

type ConfigSource = Path | Traversable

_BUNDLED_CONFIGS = files("vrl.config.presets")

_SELF_ = "_self_"


def _normalize_config_name(path: str | Path) -> str:
    text = str(path).strip().replace("\\", "/").lstrip("/")
    if not text.endswith((".yaml", ".yml")):
        text = f"{text}.yaml"
    parts = PurePosixPath(text).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid bundled config name: {path!r}")
    return PurePosixPath(*parts).as_posix()


def _join_config(root: ConfigSource, logical_name: str) -> ConfigSource:
    target = root
    for part in PurePosixPath(logical_name).parts:
        target = target.joinpath(part)
    return target


def bundled_config_resource(path: str | Path) -> Traversable:
    """Return one bundled config resource by its logical name."""

    return _join_config(_BUNDLED_CONFIGS, _normalize_config_name(path))


def list_bundled_configs(prefix: str = "") -> tuple[str, ...]:
    """List bundled YAML configs below an optional logical-name prefix."""

    prefix_parts = PurePosixPath(prefix.strip().strip("/")).parts if prefix.strip("/") else ()
    if any(part in ("", ".", "..") for part in prefix_parts):
        raise ValueError(f"invalid bundled config prefix: {prefix!r}")
    start = (
        _join_config(_BUNDLED_CONFIGS, PurePosixPath(*prefix_parts).as_posix())
        if prefix_parts
        else _BUNDLED_CONFIGS
    )
    if not start.is_dir():
        return ()

    names: list[str] = []

    def visit(node: Traversable, parts: tuple[str, ...]) -> None:
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            child_parts = (*parts, child.name)
            if child.is_dir():
                visit(child, child_parts)
            elif child.name.endswith((".yaml", ".yml")):
                names.append(PurePosixPath(*child_parts).as_posix())

    visit(start, prefix_parts)
    return tuple(names)


def _apply_default_override(entry: str | dict, default_overrides: dict[str, str]) -> str | dict:
    """Swap a defaults entry's option when a ``/group=option`` override names its group."""

    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"defaults dict must have exactly one key: {entry}")
        group = str(next(iter(entry.keys()))).strip().lstrip("/") or None
    else:
        text = str(entry).strip().lstrip("/")
        if not text or text == _SELF_:
            group = None
        else:
            if text.endswith((".yaml", ".yml")):
                text = text.rsplit(".", 1)[0]
            parts = text.split("/")
            group = "/".join(parts[:-1]) if len(parts) >= 2 else None

    if group is None or group not in default_overrides:
        return entry
    option = default_overrides[group].strip().lstrip("/")
    if not option:
        raise ValueError(f"defaults override for {group!r} cannot be empty")
    return f"{group}/{option}"


def _load_one(
    path: ConfigSource,
    root: ConfigSource,
    _seen: set[str] | None = None,
    default_overrides: dict[str, str] | None = None,
) -> DictConfig:
    """Load a single YAML file and recursively merge its ``defaults:`` list."""

    default_overrides = default_overrides or {}
    if _seen is None:
        _seen = set()
    source_key = str(path.resolve()) if isinstance(path, Path) else str(path)
    if source_key in _seen:
        raise RuntimeError(f"Cyclic defaults: {path}")
    _seen = _seen | {source_key}

    with path.open("r", encoding="utf-8") as stream:
        raw = OmegaConf.load(stream)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"{path}: top-level must be a mapping")

    defaults = raw.pop("defaults", None) if "defaults" in raw else None
    merged: DictConfig = OmegaConf.create({})

    if defaults is not None:
        if not isinstance(defaults, (list, ListConfig)):
            raise TypeError(f"{path}: 'defaults' must be a list")
        self_seen = False
        for entry in defaults:
            entry_val = entry if not hasattr(entry, "_content") else OmegaConf.to_container(entry)
            if entry_val == _SELF_:
                merged = OmegaConf.merge(merged, raw)
                self_seen = True
                continue
            entry_val = _apply_default_override(entry_val, default_overrides)
            if isinstance(entry_val, dict):
                if len(entry_val) != 1:
                    raise ValueError(f"defaults dict must have exactly one key: {entry_val}")
                key, value = next(iter(entry_val.items()))
                entry_val = f"{key}/{value}"
            sub_path = _join_config(root, _normalize_config_name(entry_val))
            merged = OmegaConf.merge(
                merged,
                _load_one(sub_path, root, _seen, default_overrides),
            )
        if not self_seen:
            merged = OmegaConf.merge(merged, raw)
    else:
        merged = raw

    assert isinstance(merged, DictConfig)
    return merged


def compose_config(
    path: str | Path,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    """Compose defaults and overrides without resolving mandatory values.

    This inspection boundary exists for structural tooling such as unknown-key
    linting. Runtime entrypoints must use :func:`load_config`, which resolves
    interpolations and rejects every mandatory ``???`` value.

    ``+group=option`` appends an independent preset after the base config, in
    command-line order (for example, ``+reward=ocr +dataset=drawbench_train_192``).
    These layers merge, rather than replace, existing sections; start with a
    reward/data-neutral recipe when selecting those roles at launch. All scalar
    dotlist overrides apply last, regardless of their command-line position.
    """

    config_root: ConfigSource = root.resolve() if root is not None else _BUNDLED_CONFIGS

    config_path = Path(path)
    source: ConfigSource
    if config_path.is_absolute():
        source = config_path.resolve()
    else:
        source = _join_config(config_root, _normalize_config_name(path))

    # ``/group=option`` swaps defaults; ``+group=option`` appends presets.
    # Dotlist values merge after both kinds of composition.
    default_overrides: dict[str, str] = {}
    preset_overlays: list[str] = []
    value_overrides: list[str] = []
    for override in overrides or []:
        if isinstance(override, str) and override.startswith("+"):
            if "=" not in override:
                raise ValueError(f"invalid additive preset override: {override!r}")
            group, option = (part.strip() for part in override[1:].split("=", 1))
            name = f"{group}/{option}"
            if (
                not group
                or not option
                or group.startswith("+")
                or any(part in ("", ".", "..") for part in name.replace("\\", "/").split("/"))
            ):
                raise ValueError(f"invalid additive preset override: {override!r}")
            preset_overlays.append(_normalize_config_name(name))
        elif isinstance(override, str) and override.startswith("/") and "=" in override:
            key, value = override.split("=", 1)
            group = key.strip().lstrip("/")
            if not group:
                raise ValueError(f"invalid defaults override: {override!r}")
            default_overrides[group] = value
        else:
            value_overrides.append(override)

    cfg = _load_one(source, config_root, default_overrides=default_overrides)

    for name in preset_overlays:
        # A selected policy may itself inherit another preset in the same group.
        # Resolve it independently so base-default replacements cannot rewrite
        # that inheritance into a cycle or change the explicitly selected layer.
        overlay = _load_one(_join_config(config_root, name), config_root)
        cfg = OmegaConf.merge(cfg, overlay)

    if value_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(value_overrides))
        assert isinstance(cfg, DictConfig)

    return cfg


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    """Load a YAML config with defaults overlay and required-value checks."""

    cfg = compose_config(path, overrides=overrides, root=root)

    OmegaConf.resolve(cfg)

    # Hard-required keys are declared as ``???`` in YAML (OmegaConf's mandatory
    # marker, e.g. trainer.entrypoint in the bundled base/trainer.yaml). Enforce
    # them here so every entrypoint fails at load time with the full list,
    # instead of one MissingMandatoryValue at first access deep into a run.
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(
            "config is missing required key(s) declared as '???': "
            + ", ".join(missing)
            + ". Set them in the experiment YAML or via dotlist overrides.",
        )
    return cfg


__all__ = [
    "bundled_config_resource",
    "compose_config",
    "list_bundled_configs",
    "load_config",
]

"""verl/Hydra-style YAML loader with ``defaults:`` overlay."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"

_SELF_ = "_self_"


def _resolve_default(entry: str | dict, root: Path) -> Path:
    """Resolve a ``defaults:`` entry to a YAML path under ``root``."""

    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"defaults dict must have exactly one key: {entry}")
        key, value = next(iter(entry.items()))
        entry = f"{key}/{value}"
    text = entry.lstrip("/")
    if not text.endswith((".yaml", ".yml")):
        text = f"{text}.yaml"
    return root / text


def _default_group(entry: str | dict) -> str | None:
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"defaults dict must have exactly one key: {entry}")
        key = str(next(iter(entry.keys()))).strip().lstrip("/")
        return key or None

    text = str(entry).strip().lstrip("/")
    if not text or text == _SELF_:
        return None
    if text.endswith((".yaml", ".yml")):
        text = text.rsplit(".", 1)[0]
    parts = text.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:-1])


def _apply_default_override(entry: str | dict, default_overrides: dict[str, str]) -> str | dict:
    group = _default_group(entry)
    if group is None or group not in default_overrides:
        return entry
    option = default_overrides[group].strip().lstrip("/")
    if not option:
        raise ValueError(f"defaults override for {group!r} cannot be empty")
    return f"{group}/{option}"


def _split_defaults_overrides(overrides: list[str] | None) -> tuple[dict[str, str], list[str]]:
    default_overrides: dict[str, str] = {}
    value_overrides: list[str] = []
    for override in overrides or []:
        if isinstance(override, str) and override.startswith("/") and "=" in override:
            key, value = override.split("=", 1)
            group = key.strip().lstrip("/")
            if not group:
                raise ValueError(f"invalid defaults override: {override!r}")
            default_overrides[group] = value
            continue
        value_overrides.append(override)
    return default_overrides, value_overrides


def _load_one(
    path: Path,
    root: Path,
    _seen: set[Path] | None = None,
    default_overrides: dict[str, str] | None = None,
) -> DictConfig:
    """Load a single YAML file and recursively merge its ``defaults:`` list."""

    path = path.resolve()
    default_overrides = default_overrides or {}
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise RuntimeError(f"Cyclic defaults: {path}")
    _seen = _seen | {path}

    raw = OmegaConf.load(path)
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
            sub_path = _resolve_default(entry_val, root)
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


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    """Load a YAML config with defaults overlay and dotlist overrides."""

    root = (root or CONFIGS_ROOT).resolve()

    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.exists():
        rel = path if isinstance(path, str) else str(path)
        rel = rel.lstrip("/")
        if not rel.endswith((".yaml", ".yml")):
            rel = f"{rel}.yaml"
        config_path = root / rel

    default_overrides, value_overrides = _split_defaults_overrides(overrides)
    cfg = _load_one(config_path, root, default_overrides=default_overrides)

    if value_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(value_overrides))
        assert isinstance(cfg, DictConfig)

    OmegaConf.resolve(cfg)
    return cfg


__all__ = ["CONFIGS_ROOT", "load_config"]

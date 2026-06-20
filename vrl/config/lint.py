"""Config lint: run `python -m vrl.config.lint` from the repo root.

Two machine sweeps, no LLM involved:
  1. code sweep   — AST-scan vrl/ for config reads (cfg_get / require /
                    OmegaConf.select); every dotted path must be registered
                    in the known-key tree.
  2. config sweep — load every experiment YAML; any unknown key is named.

Exit code 0 = clean; 1 = findings (suitable for CI / pre-commit).
The tests in tests/config/test_unknown_keys.py run the same sweeps.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from vrl.config.unknown_keys import OPEN, ConfigBlock, _root_block

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_reads_in_code(root: Path | None = None) -> tuple[set[str], set[str]]:
    """Extract (bare_keys, dotted_paths) read via the config-access helpers."""

    bare: set[str] = set()
    dotted: set[str] = set()
    for f in (root or REPO_ROOT / "vrl").rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            arg = node.args[1]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            value = arg.value
            if name == "cfg_get" or (
                name in ("require", "select")
                and value.replace(".", "").replace("_", "").isalnum()
            ):
                (dotted if "." in value else bare).add(value)
    return bare, dotted


def _path_is_known(block: object, segments: tuple[str, ...]) -> bool:
    if not segments:
        return True
    if block is OPEN:
        return True
    if not isinstance(block, ConfigBlock):
        return True  # descended past spec depth through a known key

    segment = segments[0]
    if segment in block.known or segment in block.children:
        child = block.children.get(segment, None) or (OPEN if block.open_keys else None)
        return _path_is_known(child, segments[1:])
    if any(_path_is_known(variant, segments) for variant in block.variants):
        return True
    return bool(block.open_keys)


def unregistered_code_paths() -> list[str]:
    """Code sweep: dotted config paths read by code but absent from the tree."""

    _, dotted = config_reads_in_code()
    root = _root_block()
    return sorted(p for p in dotted if not _path_is_known(root, tuple(p.split("."))))


def unknown_yaml_keys() -> dict[str, list[str]]:
    """Config sweep: unknown keys per experiment, for every shipped experiment."""

    from vrl.config.loading import load_config
    from vrl.config.unknown_keys import find_unknown_keys

    offenders: dict[str, list[str]] = {}
    experiment_dir = REPO_ROOT / "configs" / "experiment"
    for p in sorted(experiment_dir.rglob("*.yaml")):
        name = p.relative_to(experiment_dir).with_suffix("").as_posix()
        unknown = find_unknown_keys(load_config(f"experiment/{name}"))
        if unknown:
            offenders[name] = unknown
    return offenders


def main() -> int:
    findings = 0

    code_paths = unregistered_code_paths()
    if code_paths:
        findings += len(code_paths)
        print("✗ code reads config paths that are not registered:")
        for path in code_paths:
            print(f"    {path}")
        print("  fix: declare the key on its section model in vrl/config/schema.py")
    else:
        print("✓ code sweep: every config path read by vrl/ is registered")

    yaml_offenders = unknown_yaml_keys()
    if yaml_offenders:
        findings += sum(len(v) for v in yaml_offenders.values())
        print("✗ experiment configs contain unknown keys:")
        for name, keys in yaml_offenders.items():
            for key in keys:
                print(f"    {name}: {key}")
        print("  fix: typo/dead key in YAML, or a missing schema declaration")
    else:
        print("✓ config sweep: all experiment configs load with zero unknown keys")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

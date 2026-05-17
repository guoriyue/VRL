"""Architecture checks for generation, rollout, and Ray package boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"


def test_generation_layer_does_not_import_rollout_or_training_layers() -> None:
    violations = _forbidden_imports(
        VRL_ROOT / "generation",
        forbidden=(
            "vrl.algorithms",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_trajectory_layer_stays_family_neutral() -> None:
    violations = _forbidden_imports(
        VRL_ROOT / "trajectory",
        forbidden=(
            "vrl.algorithms",
            "vrl.generation.ar",
            "vrl.generation.diffusion",
            "vrl.generation.ray",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_removed_boundary_packages_stay_removed() -> None:
    assert not (VRL_ROOT / "distributed").exists()
    assert not (VRL_ROOT / "runtime").exists()


def test_new_runtime_code_does_not_import_engine_compat_paths() -> None:
    violations = []
    for path in _python_files(VRL_ROOT):
        rel = path.relative_to(ROOT)
        for module in _imports(path):
            if module == "vrl.engine" or module.startswith("vrl.engine."):
                violations.append((rel, module))
    assert not violations, _format_violations(violations)


def test_legacy_engine_package_is_removed() -> None:
    assert not (VRL_ROOT / "engine").exists()


def _forbidden_imports(
    root: Path,
    *,
    forbidden: tuple[str, ...],
    allow_path_prefixes: tuple[Path, ...] = (),
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(root):
        rel = path.relative_to(ROOT)
        if any(_is_relative_to(rel, prefix) for prefix in allow_path_prefixes):
            continue
        for module in _imports(path):
            if any(module == item or module.startswith(f"{item}.") for item in forbidden):
                violations.append((rel, module))
    return violations


def _imports(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _python_files(root: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)

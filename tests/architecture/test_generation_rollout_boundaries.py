"""Architecture checks for generation, rollout, and Ray package boundaries."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"
_GENERATION_MODEL_IMPORT_FLOOR = (
    "vrl.models.checkpoint_identity",
    "vrl.models.dtypes",
    "vrl.models.families.registry",
    "vrl.models.interfaces",
    "vrl.models.loader",
    # Residency/parking primitives are a shared runtime contract (rewards and
    # trainers use them too), not a family implementation.
    "vrl.models.parking",
)

# The ``vrl`` packages each top-level package imports, lazy imports included.
# The rows are the graph as it stands, cycles included (config <-> trainers,
# rollouts <-> trainers, models <-> generation, generation <-> trajectory are
# existing debt): a new edge is a layering decision made in this table, and a
# removed one shrinks its row. ``vrl.scripts`` is the composition root and may
# import anything.
_PACKAGE_IMPORTS: dict[str, frozenset[str]] = {
    "algorithms": frozenset({"math", "models", "rollouts", "trajectory"}),
    "config": frozenset(
        {
            "algorithms",
            "generation",
            "models",
            "ray",
            "rewards",
            "rollouts",
            "trainers",
            "trajectory",
            "utils",
        }
    ),
    "generation": frozenset(
        {"config", "math", "models", "ray", "runtime_errors", "trajectory", "utils"}
    ),
    "math": frozenset(),
    "models": frozenset({"config", "generation", "nn", "trajectory", "utils"}),
    "nn": frozenset({"utils"}),
    "ray": frozenset({"config", "models", "runtime_errors", "utils"}),
    "rewards": frozenset({"config", "models", "ray", "runtime_errors", "utils"}),
    "rollouts": frozenset(
        {
            "algorithms",
            "config",
            "generation",
            "math",
            "models",
            "ray",
            "rewards",
            "runtime_errors",
            "trainers",
            "trajectory",
            "utils",
        }
    ),
    "run": frozenset({"config", "utils"}),
    "runtime_errors": frozenset(),
    "trainers": frozenset(
        {
            "algorithms",
            "config",
            "generation",
            "math",
            "models",
            "nn",
            "rollouts",
            "trajectory",
            "utils",
        }
    ),
    "training_run_result": frozenset({"config", "runtime_errors", "utils"}),
    "trajectory": frozenset({"generation", "models", "utils"}),
    "utils": frozenset({"runtime_errors"}),
}


def test_packages_import_exactly_their_declared_dependencies() -> None:
    """Every ``vrl`` package imports the packages its ``_PACKAGE_IMPORTS`` row names, no
    more (a new edge needs a row change) and no fewer (a dropped edge shrinks the row).
    """
    actual: dict[str, set[str]] = {}
    for path in _python_files(VRL_ROOT):
        parts = path.relative_to(VRL_ROOT).parts
        package = parts[0].removesuffix(".py")
        if package in ("__init__", "scripts"):
            continue
        edges = actual.setdefault(package, set())
        for target in _imports(path):
            names = target.split(".")
            if names[0] == "vrl" and len(names) > 1 and names[1] != package:
                edges.add(names[1])

    declared = {package: set(row) for package, row in _PACKAGE_IMPORTS.items()}
    drift = {
        package: {
            "new": sorted(actual.get(package, set()) - declared.get(package, set())),
            "gone": sorted(declared.get(package, set()) - actual.get(package, set())),
        }
        for package in sorted(actual.keys() | declared.keys())
        if actual.get(package) != declared.get(package)
    }
    assert not drift, drift


def test_importing_rewards_loads_no_generation_rollout_or_training_layer() -> None:
    """The package table scans import statements and cannot see a leak through
    ``vrl.config``; import every reward module and inspect what actually loaded.
    """
    code = (
        "import importlib, pkgutil, sys, vrl.rewards\n"
        "for info in pkgutil.walk_packages(vrl.rewards.__path__, 'vrl.rewards.'):\n"
        "    importlib.import_module(info.name)\n"
        "print('\\n'.join(sys.modules))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split()
    forbidden = ("vrl.algorithms", "vrl.generation", "vrl.rollouts", "vrl.scripts", "vrl.trainers")
    leaked = sorted(
        module for module in loaded if any(_is_module_or_child(module, item) for item in forbidden)
    )
    assert not leaked, leaked


def test_generation_model_imports_stay_on_public_floor() -> None:
    """Generation may use model contracts and the registry, not family implementations."""
    violations: list[tuple[Path, str]] = []
    for path in _python_files(VRL_ROOT / "generation"):
        for target in _imports(path):
            if _is_generation_model_import_violation(target):
                violations.append((path.relative_to(ROOT), target))
    assert not violations, _format_violations(violations)


def test_import_scanner_preserves_from_import_targets(tmp_path: Path) -> None:
    """Imported aliases must remain visible to architecture boundary checks."""
    path = tmp_path / "vrl" / "generation" / "execution" / "probe.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
import vrl.models.loader as model_loader
from vrl import models
from vrl.models import checkpoint_identity, families
from vrl.models.families import registry as family_registry
from vrl.models.interfaces import RuntimeModel
from vrl.models.interfaces_bad import RuntimeModel as BadRuntimeModel
from ...models import steps

def lazy_import():
    from vrl.models.interfaces.runtime import ModelBuild
""",
        encoding="utf-8",
    )

    targets = set(_imports(path, root=tmp_path))
    assert targets == {
        "vrl.models",
        "vrl.models.checkpoint_identity",
        "vrl.models.families",
        "vrl.models.families.registry",
        "vrl.models.interfaces.RuntimeModel",
        "vrl.models.interfaces_bad.RuntimeModel",
        "vrl.models.interfaces.runtime.ModelBuild",
        "vrl.models.loader",
        "vrl.models.steps",
    }
    assert {target for target in targets if _is_generation_model_import_violation(target)} == {
        "vrl.models",
        "vrl.models.families",
        "vrl.models.interfaces_bad.RuntimeModel",
        "vrl.models.steps",
    }


def test_ray_working_dir_keeps_vendored_runtime_inputs() -> None:
    """Exercise Ray's real ignore traversal for required vendored runtime files."""

    if not (ROOT / "third_party" / "vdn-minimax-h3" / "src").is_dir():
        # The traversal is over the source checkout Ray would package; a Bazel
        # runfiles tree carries only the vendored subset, not the submodule tree.
        pytest.skip("needs a source checkout with the vdn-minimax-h3 submodule")

    import logging

    from ray._private import ray_constants
    from ray._private.runtime_env import packaging

    required = {
        "third_party/vdn-minimax-h3/src/__init__.py",
        "third_party/videophy/videocon/training/pipeline_video/mplug_owl_video/__init__.py",
    }
    excluded = {
        "third_party/vdn-minimax-h3/.git/HEAD",
        "third_party/videophy/.git/HEAD",
        "third_party/DynamicEval/docs/static/videos/prompt_id_024_compressed.mp4",
        "third_party/VMBench/Grounded-SAM-2/assets/tracking_car.mp4",
    }
    targets = required | excluded
    visited: set[str] = set()

    def record(path: Path) -> None:
        relative = path.relative_to(ROOT).as_posix()
        if relative in targets:
            visited.add(relative)

    default_excludes = packaging._get_excludes(
        ROOT,
        ray_constants.get_runtime_env_default_excludes(),
    )
    packaging._dir_travel(
        ROOT,
        [default_excludes],
        record,
        include_gitignore=True,
        logger=logging.getLogger("test-ray-package-contents"),
    )

    assert required <= visited
    assert not excluded & visited


def test_ray_ignore_excludes_submodule_git_pointer_files(tmp_path: Path) -> None:
    """A normal submodule's .git is a file, unlike local standalone clones."""

    import logging

    from ray._private.runtime_env import packaging

    (tmp_path / ".rayignore").write_text(
        (ROOT / ".rayignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_root = tmp_path / "third_party" / "vdn-minimax-h3"
    source_root.mkdir(parents=True)
    (source_root / ".git").write_text(
        "gitdir: ../../../.git/modules/third_party/vdn-minimax-h3\n",
        encoding="utf-8",
    )
    runtime_file = source_root / "src" / "models" / "runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("# runtime\n", encoding="utf-8")
    visited: set[Path] = set()

    packaging._dir_travel(
        tmp_path,
        [],
        visited.add,
        include_gitignore=True,
        logger=logging.getLogger("test-ray-submodule-pointer"),
    )

    assert source_root / ".git" not in visited
    assert runtime_file in visited


def test_trajectory_layer_stays_family_neutral() -> None:
    """``vrl.trajectory`` stays family- and runtime-neutral: no imports from the generation
    bindings, the Ray runtime, rewards, rollouts, algorithms or trainers.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "trajectory",
        forbidden=(
            "vrl.algorithms",
            "vrl.generation.bindings.full_sequence",
            "vrl.generation.ray",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_generation_execution_core_stays_ray_neutral() -> None:
    """``vrl.generation.execution`` is the Ray-free core: neither ``vrl.generation.ray`` nor
    ``vrl.ray`` may be imported there.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "generation" / "execution",
        forbidden=("vrl.generation.ray", "vrl.ray"),
    )
    assert not violations, _format_violations(violations)


def test_chunk_executor_base_stays_family_registry_neutral() -> None:
    """The composition root injects gatherers; the shared base never re-resolves them."""
    path = VRL_ROOT / "generation" / "execution" / "executor_base.py"
    violations = [
        (path.relative_to(ROOT), target)
        for target in _imports(path)
        if _is_module_or_child(target, "vrl.models.families.registry")
    ]
    assert not violations, _format_violations(violations)


def _forbidden_imports(
    root: Path,
    *,
    forbidden: tuple[str, ...],
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(root):
        rel = path.relative_to(ROOT)
        for module in _imports(path):
            if any(_is_module_or_child(module, item) for item in forbidden):
                violations.append((rel, module))
    return violations


def _imports(path: Path, *, root: Path = ROOT) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = path.relative_to(root).with_suffix("").parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent_count = len(package_parts) - node.level + 1
                if parent_count < 0:
                    raise ValueError(f"{path}: relative import escapes its package")
                base_parts = package_parts[:parent_count]
                if node.module:
                    base_parts += tuple(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""

            # Preserve the imported name so ``from vrl import models`` cannot
            # bypass a boundary that watches ``vrl.models``.
            for alias in node.names:
                if alias.name == "*":
                    if base:
                        yield base
                elif base:
                    yield f"{base}.{alias.name}"
                else:
                    yield alias.name


def _python_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _is_module_or_child(module: str, parent: str) -> bool:
    return module == parent or module.startswith(f"{parent}.")


def _is_generation_model_import_violation(target: str) -> bool:
    return _is_module_or_child(target, "vrl.models") and not any(
        _is_module_or_child(target, allowed) for allowed in _GENERATION_MODEL_IMPORT_FLOOR
    )


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)

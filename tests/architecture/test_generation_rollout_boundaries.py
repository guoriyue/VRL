"""Architecture checks for generation, rollout, and Ray package boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"
_GENERATION_MODEL_IMPORT_FLOOR = (
    "vrl.models.checkpoint_identity",
    "vrl.models.dtypes",
    "vrl.models.interfaces",
    "vrl.models.loader",
)


def test_generation_layer_does_not_import_rollout_or_training_layers() -> None:
    """Checks generation layer does not import rollout or training layers."""
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


def test_generation_model_imports_stay_on_public_floor() -> None:
    """Generation may use model contracts, not family or step implementations."""
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


def test_ray_working_dir_keeps_pinned_chunk_runtime_inputs() -> None:
    """Exercise Ray's real ignore traversal for required vendored runtime files."""

    import logging

    from ray._private import ray_constants
    from ray._private.runtime_env import packaging

    required = {
        "third_party/CausVid/causvid/models/wan/causal_model.py",
        "third_party/MAGI-1/example/4.5B/4.5B_base_config.json",
        "third_party/MAGI-1/example/assets/special_tokens.npz",
        "third_party/MAGI-1/inference/pipeline/entry.py",
    }
    excluded = {
        "third_party/CausVid/.git/HEAD",
        "third_party/MAGI-1/.git/HEAD",
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
    source_root = tmp_path / "third_party" / "CausVid"
    source_root.mkdir(parents=True)
    (source_root / ".git").write_text(
        "gitdir: ../../../.git/modules/third_party/CausVid\n",
        encoding="utf-8",
    )
    runtime_file = source_root / "causvid" / "models" / "runtime.py"
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
    """Checks trajectory layer stays family neutral."""
    violations = _forbidden_imports(
        VRL_ROOT / "trajectory",
        forbidden=(
            "vrl.algorithms",
            "vrl.generation.bindings.chunk_autoregressive_denoise",
            "vrl.generation.bindings.token_autoregressive",
            "vrl.generation.bindings.full_sequence_denoise",
            "vrl.generation.ray",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_families_registry_stays_import_light() -> None:
    """vrl/families is a neutral registry that must stay importable during config
    parse without paying torch: every MODULE-LEVEL import must be stdlib, a sibling
    ``vrl.families.*`` module, or the torch-free ``vrl.config`` schema layer
    (``registry.py`` reads ``MODEL_MEMORY_SECTIONS`` from it as the capability
    source of truth). Edges into vrl.models / vrl.trainers / vrl.generation /
    vrl.utils are deliberately function-level lazy and must stay that way.

    Walk ``tree.body`` only — NOT ``ast.walk`` — so the intentional function-level
    lazy imports (e.g. registry.py's gradient-checkpointing resolver) are not swept
    in and false-failed. This turns the lazy-import convention into a mechanical gate.
    """
    violations: list[tuple[Path, str]] = []
    for path in _python_files(VRL_ROOT / "families"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if not module.startswith("vrl."):
                    continue  # stdlib / third-party import-light deps are unrestricted
                if module == "vrl.families" or module.startswith("vrl.families."):
                    continue
                if module == "vrl.config" or module.startswith("vrl.config."):
                    continue  # torch-free config-schema layer (capability SoT)
                violations.append((path.relative_to(ROOT), module))
    assert not violations, _format_violations(violations)


def test_removed_boundary_packages_stay_removed() -> None:
    """Checks removed boundary packages stay removed."""
    assert not (VRL_ROOT / "distributed").exists()
    assert not (VRL_ROOT / "runtime").exists()
    assert not (VRL_ROOT / "generation" / "runtime").exists()
    assert not (VRL_ROOT / "rollouts" / "families").exists()
    assert not (VRL_ROOT / "rollouts" / "family_names.py").exists()

    retired_regime_paths = (
        VRL_ROOT / "generation" / "bindings" / ("joint" + "_denoise"),
        VRL_ROOT / "generation" / "bindings" / ("causal" + "_token"),
        VRL_ROOT / "generation" / "composition" / ("caus" + "al"),
        VRL_ROOT / "scripts" / "generation" / ("joint" + "_denoise.py"),
    )
    assert not [path for path in retired_regime_paths if path.exists()]


def test_retired_routing_paths_have_no_python_source() -> None:
    """The family-first layout forbids restoring routing packages under the four
    retired paths. Assert no ``*.py``/``*.pyi`` source lives there — NOT
    ``.exists()``, which would false-fail on a correct checkout that merely
    retains stale ``__pycache__`` bytecode from an older commit.
    """
    retired = (
        VRL_ROOT / "models" / "ar",
        VRL_ROOT / "models" / "diffusion",
        VRL_ROOT / "generation" / "ar",
        VRL_ROOT / "generation" / "diffusion",
    )
    offenders = [
        source
        for path in retired
        for pattern in ("*.py", "*.pyi")
        for source in path.rglob(pattern)
        if "__pycache__" not in source.parts
    ]
    assert not offenders, "retired routing paths must contain no Python source:\n" + "\n".join(
        str(source.relative_to(ROOT)) for source in offenders
    )


def test_shared_ray_substrate_stays_domain_neutral() -> None:
    """Checks shared Ray substrate stays domain neutral."""
    assert (VRL_ROOT / "ray" / "resources.py").exists()
    assert not (VRL_ROOT / "generation" / "resources.py").exists()
    violations = _forbidden_imports(
        VRL_ROOT / "ray",
        forbidden=(
            "vrl.generation",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_reward_scoring_is_in_process() -> None:
    """Rewards score in-process; the removed Ray pool transport must stay gone.

    The pool (actor pool + release_after_call kill/reload + resident parking)
    was replaced by InProcessRewardRuntime sleep/wake offload. Guard against it
    creeping back as a directory, and keep the in-process runtime generic (no
    model-specific code in the shared transport).
    """
    assert not (VRL_ROOT / "rewards" / "ray").exists()
    assert not (VRL_ROOT / "rewards" / "ray.py").exists()
    runtime_text = (VRL_ROOT / "rewards" / "runtime.py").read_text(encoding="utf-8")
    for snippet in ("KlingTeam", "VideoVLMRewardInference", "huggingface_hub"):
        assert snippet not in runtime_text, "runtime.py leaks a specific model"
    assert not (VRL_ROOT / "rewards" / "inference").exists()
    assert not (VRL_ROOT / "rewards" / "video_inference").exists()
    assert not list((VRL_ROOT / "rewards").rglob("spec.py"))


def test_reward_models_live_under_models() -> None:
    """Checks reward models live under models."""
    models_root = VRL_ROOT / "rewards" / "models"
    present = _module_filenames(models_root)
    # Every registered reward has a model module here (registry is the source).
    assert _registered_reward_modules() <= present
    # Only scaffolding may live alongside the per-reward modules.
    scaffolding = {"__init__.py", "base.py", "hub.py"}
    extras = present - _registered_reward_modules() - scaffolding
    assert not extras, f"unexpected modules under rewards/models/: {extras}"
    assert not (VRL_ROOT / "rewards" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "ray" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "scorers").exists()


def test_reward_inference_is_a_single_domain_module() -> None:
    """Checks reward inference is a single domain module."""
    inference_path = VRL_ROOT / "rewards" / "inference.py"
    assert inference_path.exists()
    inference_text = inference_path.read_text(encoding="utf-8")
    assert "build_reward_inference_runtime" not in inference_text
    assert "vrl.ray" not in inference_text
    assert "vrl.rewards.ray" not in inference_text
    assert not (VRL_ROOT / "rewards" / "inference_runtime.py").exists()
    assert not (VRL_ROOT / "rewards" / "inference_worker.py").exists()
    assert not (VRL_ROOT / "rewards" / "inference_scheduler.py").exists()
    assert not (VRL_ROOT / "rewards" / "scoring_worker.py").exists()


def test_reward_function_implementations_live_under_functions() -> None:
    """Checks reward function implementations live under functions."""
    rewards_root = VRL_ROOT / "rewards"
    required_root = {
        "__init__.py",
        "artifacts.py",
        "base.py",
        "inference.py",
        "runtime.py",
        "types.py",
    }
    assert required_root <= _module_filenames(rewards_root)

    functions = _module_filenames(rewards_root / "functions")
    assert _registered_reward_modules() <= functions
    scaffolding = {"__init__.py", "base.py", "registry.py"}
    extras = functions - _registered_reward_modules() - scaffolding
    assert not extras, f"unexpected modules under rewards/functions/: {extras}"


def test_generation_ray_adapter_stays_lean() -> None:
    """Checks generation Ray adapter stays lean."""
    ray_root = VRL_ROOT / "generation" / "ray"
    required = {
        "__init__.py",
        "config.py",
        "executor.py",
        "launcher.py",
        "runtime.py",
        "weight_sync.py",
        "worker.py",
    }
    assert required <= _module_filenames(ray_root)
    for speculative_stage_adapter in ("pipeline_runner.py", "stage_worker.py"):
        assert not (ray_root / speculative_stage_adapter).exists()
    assert not (VRL_ROOT / "generation" / "pipeline").exists()
    ray_adapter_files = (
        ray_root / "config.py",
        ray_root / "executor.py",
        ray_root / "launcher.py",
        ray_root / "runtime.py",
        ray_root / "worker.py",
        ray_root / "weight_sync.py",
    )
    for path in ray_adapter_files:
        text = path.read_text(encoding="utf-8")
        assert "vrl.generation.execution.planner import build_engine_plan" not in text
        assert "vrl.generation.execution.chunks import" not in text


def test_generation_execution_core_stays_flat_and_ray_neutral() -> None:
    """Checks generation execution core stays flat and Ray neutral."""
    execution_root = VRL_ROOT / "generation" / "execution"
    for expected in (
        "__init__.py",
        "chunk_placement.py",
        "types.py",
        "worker.py",
    ):
        assert (execution_root / expected).exists()
    for ray_specific in ("executor.py", "placement.py"):
        assert not (execution_root / ray_specific).exists()
    assert not (execution_root / "distributed").exists()
    for obsolete in (
        "distributed_executor.py",
        "distributed_planner.py",
        "distributed_types.py",
        "placement.py",
        "stage_plan.py",
        "worker_core.py",
    ):
        assert not (execution_root / obsolete).exists()


def test_chunk_executor_base_stays_family_registry_neutral() -> None:
    """The composition root injects gatherers; the shared base never re-resolves them."""
    path = VRL_ROOT / "generation" / "execution" / "executor_base.py"
    violations = [
        (path.relative_to(ROOT), target)
        for target in _imports(path)
        if _is_module_or_child(target, "vrl.families")
    ]
    assert not violations, _format_violations(violations)


def test_new_runtime_code_does_not_import_engine_compat_paths() -> None:
    """Checks new runtime code does not import engine compat paths."""
    violations = []
    for path in _python_files(VRL_ROOT):
        rel = path.relative_to(ROOT)
        for module in _imports(path):
            if module == "vrl.engine" or module.startswith("vrl.engine."):
                violations.append((rel, module))
    assert not violations, _format_violations(violations)


def test_removed_engine_packages_stay_removed() -> None:
    """Checks removed engine packages stay removed."""
    assert not (VRL_ROOT / "engine").exists()
    assert not (ROOT / "tests" / "engine").exists()


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


def _module_filenames(root: Path) -> set[str]:
    return {path.name for path in root.glob("*.py") if "__pycache__" not in path.parts}


def _registered_reward_modules() -> set[str]:
    """Reward-impl filenames derived from the registry, the single source of truth.

    Each registered reward ``<name>`` owns a ``<name>.py`` module, so the
    expected module set is the registry keys — never a hand-typed ``ls``.
    Registration is lazy (``_register_builtins`` runs inside ``from_dict``),
    so trigger it once with an empty score dict before reading the keys.
    """
    from vrl.rewards.functions.registry import _REWARD_REGISTRY, MultiReward

    MultiReward.from_dict({}, device="cpu")  # populate _REWARD_REGISTRY
    return {f"{name}.py" for name in _REWARD_REGISTRY}


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _is_module_or_child(module: str, parent: str) -> bool:
    return module == parent or module.startswith(f"{parent}.")


def _is_generation_model_import_violation(target: str) -> bool:
    return _is_module_or_child(target, "vrl.models") and not any(
        _is_module_or_child(target, allowed) for allowed in _GENERATION_MODEL_IMPORT_FLOOR
    )


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)

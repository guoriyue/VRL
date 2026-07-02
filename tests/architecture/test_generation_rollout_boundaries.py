"""Architecture checks for generation, rollout, and Ray package boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"


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


def test_trajectory_layer_stays_family_neutral() -> None:
    """Checks trajectory layer stays family neutral."""
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
    """Checks removed boundary packages stay removed."""
    assert not (VRL_ROOT / "distributed").exists()
    assert not (VRL_ROOT / "runtime").exists()
    assert not (VRL_ROOT / "generation" / "runtime").exists()


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
    was replaced by LocalRewardRuntime sleep/wake offload. Guard against it
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
    required_root = {"__init__.py", "artifacts.py", "base.py", "inference.py", "runtime.py", "types.py"}
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
        "pipeline_runner.py",
        "stage_worker.py",
        "weight_sync.py",
        "worker.py",
    }
    assert required <= _module_filenames(ray_root)
    ray_adapter_files = (
        ray_root / "config.py",
        ray_root / "executor.py",
        ray_root / "launcher.py",
        ray_root / "runtime.py",
        ray_root / "pipeline_runner.py",
        ray_root / "stage_worker.py",
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


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)

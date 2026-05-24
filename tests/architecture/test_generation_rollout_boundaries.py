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
    assert not (VRL_ROOT / "generation" / "runtime").exists()


def test_shared_ray_substrate_stays_domain_neutral() -> None:
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


def test_reward_ray_adapter_stays_lean() -> None:
    ray_root = VRL_ROOT / "rewards" / "ray"
    assert _module_filenames(ray_root) == {
        "__init__.py",
        "model.py",
        "runtime.py",
        "worker.py",
    }
    forbidden_text = (
        "class RewardInferenceArtifact",
        "class RewardInferenceRequest",
        "class RewardInferenceResult",
        "class VideoRewardArtifactStore",
    )
    model_specific = ("KlingTeam", "VideoVLMRewardInference", "huggingface_hub")
    for path in _python_files(ray_root):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_text:
            assert snippet not in text
        for snippet in model_specific:
            assert snippet not in text, f"{path} leaks a specific model into generic ray/"
    assert not (VRL_ROOT / "rewards" / "ray.py").exists()
    assert not (VRL_ROOT / "rewards" / "inference").exists()
    assert not (VRL_ROOT / "rewards" / "video_inference").exists()
    assert not list((VRL_ROOT / "rewards").rglob("spec.py"))


def test_reward_models_live_under_models() -> None:
    models_root = VRL_ROOT / "rewards" / "models"
    assert _module_filenames(models_root) == {
        "__init__.py",
        "kling_video_reward.py",
    }
    assert not (VRL_ROOT / "rewards" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "ray" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "scorers").exists()


def test_reward_inference_is_a_single_domain_module() -> None:
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
    rewards_root = VRL_ROOT / "rewards"
    assert _module_filenames(rewards_root) == {
        "__init__.py",
        "artifacts.py",
        "base.py",
        "inference.py",
        "types.py",
    }
    assert _module_filenames(rewards_root / "functions") == {
        "__init__.py",
        "aesthetic.py",
        "anime_anatomy.py",
        "clip.py",
        "codex_image_qa.py",
        "geneval.py",
        "nsfw_safety.py",
        "ocr.py",
        "pickscore.py",
        "registry.py",
        "video_reward.py",
    }


def test_generation_ray_adapter_stays_lean() -> None:
    ray_root = VRL_ROOT / "generation" / "ray"
    assert _module_filenames(ray_root) == {
        "__init__.py",
        "config.py",
        "executor.py",
        "launcher.py",
        "placement.py",
        "runtime.py",
        "weight_sync.py",
        "worker.py",
    }
    ray_adapter_files = (
        ray_root / "config.py",
        ray_root / "executor.py",
        ray_root / "launcher.py",
        ray_root / "placement.py",
        ray_root / "runtime.py",
        ray_root / "worker.py",
        ray_root / "weight_sync.py",
    )
    for path in ray_adapter_files:
        text = path.read_text(encoding="utf-8")
        assert "vrl.generation.execution.planner import build_engine_plan" not in text
        assert "vrl.generation.execution.chunks import" not in text


def test_generation_execution_core_stays_flat_and_ray_neutral() -> None:
    execution_root = VRL_ROOT / "generation" / "execution"
    for expected in (
        "__init__.py",
        "scheduler.py",
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


def _module_filenames(root: Path) -> set[str]:
    return {
        path.name
        for path in root.glob("*.py")
        if "__pycache__" not in path.parts
    }


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)

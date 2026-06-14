"""Architecture checks for runtime memory policy boundaries."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"


def test_diffusion_model_loaders_use_vae_decode_memory_boundary() -> None:
    """VAE tiling/slicing must go through vrl.models.diffusion.common.vae_decode_memory."""

    # Only VAE tiling/slicing has a policy boundary (vae_decode_memory). Diffusers
    # pipeline-level offload (enable_model_cpu_offload / enable_sequential_cpu_offload)
    # is a legitimate single-GPU inference strategy with no boundary abstraction —
    # e.g. Wan I2V 14B on a 32 GB card — so it is intentionally not forbidden here.
    violations = _forbidden_text(
        VRL_ROOT / "models" / "diffusion",
        filename="model.py",
        forbidden=(
            "enable_tiling(",
            "enable_slicing(",
        ),
    )
    assert not violations, _format_violations(violations)


def test_diffusion_train_scripts_do_not_inline_cpu_offload_policy() -> None:
    """Train scripts must not inline driver CPU offload.

    The trainer never loads the generation-only modules (text encoders, VAE):
    each family builds a minimal ReplayModel that omits them, so there is
    nothing to offload. A train.py reaching for ``.to("cpu")`` or
    ``enable_model_cpu_offload`` is reintroducing a problem the ReplayModel
    boundary already removed.
    """

    violations = _forbidden_text(
        VRL_ROOT / "scripts" / "diffusion",
        filename="train.py",
        forbidden=(
            "enable_model_cpu_offload(",
            '.to("cpu")',
            ".to('cpu')",
        ),
    )
    assert not violations, _format_violations(violations)


def test_runtime_interface_does_not_parse_model_memory_sections() -> None:
    """RuntimeBuildSpec is a data contract, not a model.memory parser."""

    text = (VRL_ROOT / "models" / "interfaces" / "runtime.py").read_text(
        encoding="utf-8",
    )

    assert "model_memory_config_from_cfg" not in text
    assert "memory_policy_config_from_cfg" not in text


def _forbidden_text(
    root: Path,
    *,
    filename: str,
    forbidden: tuple[str, ...],
) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob(filename)):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for snippet in forbidden:
                if snippet in line:
                    violations.append((path.relative_to(ROOT), lineno, snippet))
    return violations


def _format_violations(violations: list[tuple[Path, int, str]]) -> str:
    return "\n".join(
        f"{path}:{lineno}: inline memory policy call {snippet!r}"
        for path, lineno, snippet in violations
    )

"""Every family's policy-core declaration must be reachable by the pass layer.

This is the automated form of the bug that motivated the declaration: Wan
overrode ``torch_compile_transformer`` to walk both experts but inherited a
quantization path that walked only ``self.transformer``, so its second expert
silently ran at the base dtype. Nothing caught it -- the runtime guard only
checked that the swap count was non-zero.

A model class cannot be instantiated here without checkpoints, so these tests
assert the CLASS-LEVEL contract: that each family's model class resolves
``policy_cores`` and ``quantization_exclude``, and that a family owning more than
one rollout module declares all of them rather than relying on a per-pass
override.
"""

from __future__ import annotations

import pytest

from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    DenoiseFamilyBuild,
    TokenFamilyBuild,
)
from vrl.utils.config import import_from_path


def _model_classes() -> list[tuple[str, type]]:
    """(family, rollout model class) for every registry entry that has one."""

    resolved: list[tuple[str, type]] = []
    for family, entry in sorted(FAMILY_REGISTRY.items()):
        recipe = entry.family_build
        if not isinstance(recipe, DenoiseFamilyBuild | TokenFamilyBuild):
            continue
        try:
            resolved.append((family, import_from_path(recipe.model_cls)))
        except Exception as exc:  # optional per-family deps are not this test's subject
            pytest.skip(f"{family}: {exc}")
    return resolved


FAMILY_MODEL_CLASSES = _model_classes()


@pytest.mark.parametrize(
    ("family", "model_cls"),
    FAMILY_MODEL_CLASSES,
    ids=[family for family, _ in FAMILY_MODEL_CLASSES],
)
def test_every_family_declares_the_optimization_contract(family: str, model_cls: type) -> None:
    """Both halves of the declaration must resolve on every rollout policy."""

    assert hasattr(model_cls, "policy_cores"), (
        f"{family} declares no policy_cores, so the rollout optimization passes "
        "have no roots to walk"
    )
    exclude = getattr(model_cls, "quantization_exclude", None)
    assert isinstance(exclude, tuple), (
        f"{family}.quantization_exclude must be a tuple of path substrings; got {exclude!r}"
    )


@pytest.mark.parametrize(
    ("family", "model_cls"),
    FAMILY_MODEL_CLASSES,
    ids=[family for family, _ in FAMILY_MODEL_CLASSES],
)
def test_no_family_reintroduces_a_per_scheme_quantization_override(
    family: str,
    model_cls: type,
) -> None:
    """Coverage comes from the declaration, never from a per-scheme method.

    A family that re-adds ``quantize_rollout_fp8``/``_nvfp4`` would be covered
    for those two schemes and silently uncovered for the next one -- exactly the
    duplication ``policy_cores`` removed.
    """

    for scheme_method in ("quantize_rollout_fp8", "quantize_rollout_nvfp4"):
        assert not hasattr(model_cls, scheme_method), (
            f"{family} defines {scheme_method}; declare policy_cores instead so "
            "every present and future pass covers the same roots"
        )


def test_multi_expert_family_declares_all_its_rollout_modules() -> None:
    """Wan is the multi-root case: its declaration, not a pass, is the coverage.

    Asserted on the real class rather than a fake so a future edit that narrows
    ``policy_cores`` back to ``self.transformer`` fails here.
    """

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    source = WanT2VDiffusersModel.policy_cores.fget
    assert source is not None
    # The declaration must delegate to the both-experts accessor, not to the
    # single ``transformer`` attribute the base class uses.
    assert "_wan_transformers" in source.__code__.co_names, (
        "Wan's policy_cores no longer returns both experts; a dual-stage rollout "
        "would run its two halves at different precisions"
    )

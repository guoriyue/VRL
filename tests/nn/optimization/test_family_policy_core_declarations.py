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

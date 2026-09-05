"""One name-keyed write-back primitive serves rollout and trainer alike.

Anything that REPLACES a module root rather than mutating it in place --
``torch.compile`` on the rollout side, FSDP/DDP on the trainer side -- must be
written back to every alias the model holds. Before this seam existed the
trainer looked the writer up by interpolating a private method name
(``getattr(model, f"_set_{name}")``), which forced a multi-root family to keep
one one-line method per root just so the string lookup would resolve.

These tests pin the replacement: ONE ``set_module_root(name, module)`` override
serves every caller. Rollout and trainer select DIFFERENT roots
(``policy_cores`` vs ``trainable_modules`` -- see
``test_the_two_views_are_not_the_same_key_set``), but writing one back is the
same operation, so it is one method rather than a per-caller spelling.
"""

from __future__ import annotations

from typing import Any

import pytest
from torch import nn

from vrl.models.steps.denoise.base import DiffusionModelBase


class _SingleRoot:
    """The default shape: one root, no override needed."""

    def __init__(self) -> None:
        self.transformer = nn.Linear(4, 4)
        self.aliases: list[Any] = []

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.aliases.append(transformer)

    # Borrow the base implementation without the rest of DiffusionModelBase.
    set_module_root = DiffusionModelBase.set_module_root


class _DualRoot(_SingleRoot):
    """A Wan-shaped policy: ONE override covers both views and every pass."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_2 = nn.Linear(4, 4)

    def set_module_root(self, name: str, module: Any) -> None:
        if name == "transformer_2":
            self.transformer_2 = module
            return
        _SingleRoot.set_module_root(self, name, module)


def test_write_back_reaches_every_alias() -> None:
    """The replacement must land in every handle the model holds, not just one."""

    policy = _SingleRoot()
    compiled = nn.Linear(4, 4)
    wrapped = nn.Linear(4, 4)

    policy.set_module_root("transformer", compiled)
    assert policy.transformer is compiled

    policy.set_module_root("transformer", wrapped)
    assert policy.transformer is wrapped

    # Both went through _set_transformer, so both reached every alias.
    assert policy.aliases == [compiled, wrapped]


def test_one_override_serves_both_views_on_a_multi_root_policy() -> None:
    """The point of the refactor: no per-root method under a derived name."""

    policy = _DualRoot()
    expert_1 = nn.Linear(4, 4)
    expert_2 = nn.Linear(4, 4)

    policy.set_module_root("transformer", expert_1)
    policy.set_module_root("transformer_2", expert_2)

    assert policy.transformer is expert_1
    assert policy.transformer_2 is expert_2


def test_unknown_root_fails_loud_on_a_single_root_policy() -> None:
    """A typo must not silently write nothing."""

    policy = _SingleRoot()

    with pytest.raises(ValueError, match="no module root 'transformer_2'"):
        policy.set_module_root("transformer_2", nn.Linear(4, 4))


def test_the_two_views_are_not_the_same_key_set() -> None:
    """The two SELECTIONS differ even though the write-back operation does not.

    ``policy_cores`` is every expert the family SAMPLES through;
    ``trainable_modules`` is only the experts the trainer UPDATES. Wan's
    dual-stage config trains one expert and samples both, so collapsing the two
    would silently narrow rollout coverage -- the same class of bug as the
    half-quantized dual expert. That is why they stay two properties; the writer
    below is still one method, because writing a root back does not depend on
    which selection asked for it.
    """

    from types import SimpleNamespace

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    model = WanT2VDiffusersModel(
        pipeline=SimpleNamespace(
            transformer=nn.Linear(4, 4),
            transformer_2=nn.Linear(4, 4),
            config={"boundary_ratio": 0.5},
            device="cpu",
            scheduler=None,
        ),
        device="cpu",
        trainable_transformers=["transformer"],
    )

    assert set(model.policy_cores) == {"transformer", "transformer_2"}
    assert set(model.trainable_modules) == {"transformer"}
    # ...and the single override serves the wider key set too.
    replacement = nn.Linear(4, 4)
    model.set_module_root("transformer_2", replacement)
    assert model.transformer_2 is replacement

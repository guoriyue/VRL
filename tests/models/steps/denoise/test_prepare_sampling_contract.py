"""Every registered family honors ``initial_latents`` explicitly.

The executor passes the group's shared start as a keyword on every call. A
family whose ``prepare_sampling`` only absorbed it through ``**kwargs`` would
draw its own noise and silently break group-shared noise, so the parameter must
be declared by name. Stubs that refuse rollout sampling altogether are exempt.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from vrl.models.families.registry import FAMILY_REGISTRY
from vrl.utils.config import import_from_path


def _prepare_sampling_raises_unconditionally(function: object) -> bool:
    """True when the method's first real statement is ``raise`` (a rollout stub)."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    return bool(body) and isinstance(body[0], ast.Raise)


@pytest.mark.parametrize("key", sorted(FAMILY_REGISTRY))
def test_registered_family_declares_initial_latents(key: str) -> None:
    entry = FAMILY_REGISTRY[key]
    model_cls = import_from_path(entry.family_build.model_cls)
    prepare_sampling = getattr(model_cls, "prepare_sampling", None)
    if prepare_sampling is None:
        pytest.skip(f"{model_cls.__name__} has no in-process sampling")
    if _prepare_sampling_raises_unconditionally(prepare_sampling):
        pytest.skip(f"{model_cls.__name__} refuses rollout sampling")
    parameters = inspect.signature(prepare_sampling).parameters
    assert "initial_latents" in parameters, (
        f"{model_cls.__name__}.prepare_sampling must declare initial_latents; "
        "absorbing it through **kwargs would silently ignore a group-shared start"
    )
    assert parameters["initial_latents"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["initial_latents"].default is None

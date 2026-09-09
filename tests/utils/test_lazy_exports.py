"""The lazy public boundary the trajectory and trainer packages install.

These packages exist so `vrl.config.schema` can reach a torch-free type without
charging every config parse for the tensor submodules next to it, so the
contract under test is: nothing loads until it is asked for, the surface still
looks complete, and the packages stay importable without torch.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from vrl.utils.config import install_lazy_exports

LAZY_PACKAGES = ("vrl.trajectory", "vrl.trainers.data", "vrl.trainers.online")
# The two a torch-free config parse reaches; vrl.trainers.online is the trainer
# itself and is only imported once torch is already loaded.
TORCH_FREE_PACKAGES = ("vrl.trajectory", "vrl.trainers.data")


def _namespace() -> dict[str, object]:
    return {"__name__": "tests.fake_package"}


def test_installer_defers_the_import_until_the_symbol_is_requested() -> None:
    namespace = _namespace()
    install_lazy_exports(namespace, {"Mapping": "collections.abc"})

    assert namespace["__all__"] == ["Mapping"]
    assert "Mapping" not in namespace

    from collections.abc import Mapping

    assert namespace["__getattr__"]("Mapping") is Mapping
    # Resolved once, then served from the package namespace.
    assert namespace["Mapping"] is Mapping


def test_installer_reports_an_unknown_name_as_a_module_attribute_error() -> None:
    namespace = _namespace()
    install_lazy_exports(namespace, {"Mapping": "collections.abc"})

    with pytest.raises(
        AttributeError, match=r"module 'tests.fake_package' has no attribute 'nope'"
    ):
        namespace["__getattr__"]("nope")


def test_installer_dir_covers_declared_and_already_resolved_names() -> None:
    namespace = _namespace()
    install_lazy_exports(namespace, {"Mapping": "collections.abc"})
    namespace["local"] = object()

    listing = namespace["__dir__"]()

    assert "Mapping" in listing and "local" in listing
    assert listing == sorted(listing)


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_every_declared_export_resolves(package: str) -> None:
    """A stale table entry would only fail on the first access in production."""
    module = importlib.import_module(package)

    for name in module.__all__:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_type_checking_block_and_export_table_agree(package: str) -> None:
    """The TYPE_CHECKING re-exports are what a type checker sees; keep them in sync."""
    import ast
    from pathlib import Path

    source = Path(str(sys.modules[package].__file__)).read_text()
    declared = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom):
                    declared.update(alias.asname or alias.name for alias in stmt.names)

    assert declared == set(importlib.import_module(package).__all__)


@pytest.mark.parametrize("package", TORCH_FREE_PACKAGES)
def test_package_imports_without_torch(package: str) -> None:
    """Subprocess: this session has already imported torch, and evicting it from
    ``sys.modules`` in-process would break every later test that holds a torch
    reference."""

    probe = f"import sys, importlib; importlib.import_module({package!r}); raise SystemExit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0

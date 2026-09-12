"""Storage roots constrain resolved references across artifact consumers."""

from pathlib import Path

import pytest

from vrl.utils.artifacts import PathOutsideRootsError, RootedPaths


def test_relative_absolute_and_internal_symlink_references(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    target = root / "image.png"
    target.touch()
    (root / "alias.png").symlink_to(target)
    paths = RootedPaths(root)

    assert paths.resolve("image.png", strict=True) == target
    assert paths.resolve(target, strict=True) == target
    assert paths.resolve("alias.png", strict=True) == target
    assert paths.resolve("new/output.json") == root / "new/output.json"
    with pytest.raises(FileNotFoundError):
        paths.resolve("missing.png", strict=True)


def test_parent_absolute_and_symlink_escapes_are_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.touch()
    (root / "alias.png").symlink_to(outside)
    paths = RootedPaths(root)

    for reference in ("../outside.png", outside, "alias.png"):
        with pytest.raises(PathOutsideRootsError) as error:
            paths.resolve(reference)
        assert error.value.path == outside


def test_multiple_roots_accept_absolute_references_without_guessing_relative_root(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    paths = RootedPaths(first, second)

    assert paths.resolve(first / "a") == first / "a"
    assert paths.resolve(second / "b") == second / "b"
    with pytest.raises(ValueError, match="exactly one storage root"):
        paths.resolve("a")
    with pytest.raises(PathOutsideRootsError):
        paths.resolve(tmp_path / "first-lookalike" / "a")


def test_relative_root_is_normalized_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = RootedPaths(Path("data"))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert paths.resolve("a") == tmp_path / "data/a"

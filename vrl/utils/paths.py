"""Where paths are relative to: the user, the data root, the repository.

``user_path`` reads a path the user wrote, ``RootedPaths`` judges artifact
references against storage roots with a policy fixed at construction, and
``repo_root`` is the checkout's root for local defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from vrl.utils.artifacts import ArtifactManifestError

DATA_ROOT_ENV = "VRL_DATA_ROOT"


class PathOutsideRootsError(ArtifactManifestError):
    """A resolved path escaped its configured storage roots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"path is outside configured roots: {path}")


class RootedPaths:
    """Where artifact references may point: one or more roots, and one ``resolve``.

    The policy is fixed when the object is built, so every reference a
    consumer resolves is judged the same way:

    - a relative reference lands under the single root (several roots cannot
      say which one, so relative references are refused there);
    - an absolute reference is, per ``absolute``, kept ``"inside"`` a root
      after symlinks resolve (the reward service's sandbox), refused outright
      (``"reject"``, the default for manifests under the data root), or taken
      from ``"anywhere"`` (manifests allowed to point outside the data root).

    Symlink escapes are caught after resolution. Protocol-specific errors and
    file writers remain with the caller.
    """

    def __init__(
        self,
        root: str | Path,
        *additional_roots: str | Path,
        absolute: Literal["inside", "reject", "anywhere"] = "inside",
    ) -> None:
        self.roots = tuple(user_path(value) for value in (root, *additional_roots))
        self.absolute = absolute

    @classmethod
    def data_root(
        cls, value: str | Path | None = None, *, allow_absolute: bool = False
    ) -> RootedPaths:
        """The manifest data root: ``value``, else ``VRL_DATA_ROOT``, else the local default."""

        if value is None:
            value = os.environ.get(DATA_ROOT_ENV, "").strip() or repo_root() / "data" / "external"
        return cls(value, absolute="anywhere" if allow_absolute else "reject")

    @property
    def root(self) -> Path:
        """The single root, for callers that write under it."""

        if len(self.roots) != 1:
            raise ValueError("root is defined only for a single storage root")
        return self.roots[0]

    def resolve(self, raw_path: str | Path, *, strict: bool = False) -> Path:
        """One reference to an absolute path under the policy; ``strict`` requires it to exist."""

        text = str(raw_path).strip()
        if not text:
            raise ArtifactManifestError("artifact path is empty")
        path = Path(text).expanduser()
        if path.is_absolute():
            if self.absolute == "reject":
                raise ArtifactManifestError(
                    f"absolute artifact paths are not allowed by default: {text}",
                )
            if self.absolute == "anywhere":
                return path.resolve(strict=strict)
        else:
            if len(self.roots) != 1:
                raise ValueError("relative paths require exactly one storage root")
            path = self.roots[0] / path
        resolved = path.resolve(strict=strict)
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise PathOutsideRootsError(resolved)
        return resolved


def repo_root() -> Path:
    """Return the repository root for local ignored artifact defaults."""

    return Path(__file__).resolve().parents[2]


def user_path(value: str | Path) -> Path:
    """A path the user wrote (CLI argument or config value): ``~`` expanded, made absolute."""

    return Path(value).expanduser().resolve()


__all__ = [
    "PathOutsideRootsError",
    "RootedPaths",
    "repo_root",
    "user_path",
]

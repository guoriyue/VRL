"""Assemble the qualified CountGD runtime tree from Bazel-fetched inputs.

Inputs are the patched upstream source (``@countgd_src``) and the pinned
Hugging Face Space assets. The output layout is the one the reward model
verifies: ``<root>/countgd/source/...`` plus ``<root>/countgd/install_manifest.json``.
The tree digest must equal ``COUNTGD_RUNTIME_TREE_SHA256``; a different
digest means the inputs no longer match the qualified protocol, so this fails.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from vrl.rewards.models.countgd import (
    _RUNTIME_TREE_ALGORITHM,
    _RUNTIME_TREE_SCHEMA,
    COUNTGD_CHECKPOINT_SHA256,
    COUNTGD_INSTALL_SCHEMA,
    COUNTGD_MODEL_VERSION,
    COUNTGD_RUNTIME_TREE_SHA256,
    COUNTGD_SOURCE_REVISION,
    COUNTGD_SPACE_REVISION,
    _runtime_tree_digest,
    countgd_model_protocol,
)
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import write_json

# The qualified tree carries an upstream Python 3.10 build directory whose
# only non-binary members are these copies of their sources. Recreating them
# keeps the audited 133-file layout without trusting a foreign binary.
_QUALIFIED_BUILD_LIB = Path("models/GroundingDINO/ops/build/lib.linux-x86_64-cpython-310")
_QUALIFIED_BUILD_PYTHON_FILES = (
    "functions/__init__.py",
    "functions/ms_deform_attn_func.py",
    "modules/__init__.py",
    "modules/ms_deform_attn.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset", action="append", default=[], metavar="RELATIVE=FILE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.output / "countgd" / "source"
    # Bazel adds repository boilerplate at the archive root; it is not upstream.
    bazel_boilerplate = {
        "BUILD.bazel",
        "WORKSPACE",
        "WORKSPACE.bazel",
        "MODULE.bazel",
        "REPO.bazel",
    }
    shutil.copytree(
        args.source_root,
        source_dir,
        symlinks=False,
        ignore=lambda directory, names: (
            bazel_boilerplate & set(names) if Path(directory) == Path(args.source_root) else set()
        ),
    )
    for spec in args.asset:
        relative, _, file = spec.partition("=")
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    ops_root = source_dir / "models/GroundingDINO/ops"
    for relative in _QUALIFIED_BUILD_PYTHON_FILES:
        destination = source_dir / _QUALIFIED_BUILD_LIB / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ops_root / relative, destination)

    checkpoint = sha256_file(source_dir / "checkpoint_best_regular.pth")
    if checkpoint != COUNTGD_CHECKPOINT_SHA256:
        raise SystemExit(f"CountGD checkpoint hash mismatch: {checkpoint}")
    tree = _runtime_tree_digest(source_dir)
    if tree.sha256 != COUNTGD_RUNTIME_TREE_SHA256:
        raise SystemExit(
            "CountGD runtime tree does not match the qualified digest: "
            f"expected={COUNTGD_RUNTIME_TREE_SHA256} actual={tree.sha256} "
            f"files={tree.file_count}"
        )
    write_json(
        args.output / "countgd" / "install_manifest.json",
        {
            "schema": COUNTGD_INSTALL_SCHEMA,
            "assembled_by": "bazel //third_party/countgd:runtime",
            "source_revision": COUNTGD_SOURCE_REVISION,
            "space_revision": COUNTGD_SPACE_REVISION,
            "checkpoint_file": "source/checkpoint_best_regular.pth",
            "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
            "model_version": COUNTGD_MODEL_VERSION,
            "inference_protocol": countgd_model_protocol(),
            "runtime_tree": {
                "schema": _RUNTIME_TREE_SCHEMA,
                "algorithm": _RUNTIME_TREE_ALGORITHM,
                "file_count": tree.file_count,
                "sha256": tree.sha256,
            },
            "qualified_device": "cpu",
        },
    )


if __name__ == "__main__":
    main()

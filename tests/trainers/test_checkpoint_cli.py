from __future__ import annotations

from pathlib import Path

import pytest

from vrl.trainers.checkpointing import (
    CheckpointTarget,
)


def test_checkpoint_cli_values_become_distinct_labelled_arms(tmp_path: Path) -> None:
    """`--checkpoint [LABEL=]PATH` parsing, shared by the evaluation entrypoints.

    Two of them used to carry byte-identical copies of this, down to the error
    strings, differing only in whether a plain file is acceptable.
    """

    final = tmp_path / "epoch-3" / "checkpoint-final"
    final.mkdir(parents=True)
    other = tmp_path / "epoch-7"
    other.mkdir()

    # Whitespace around the label is trimmed; the path is taken as written,
    # because a padded path is a typo, not a label style.
    targets = CheckpointTarget.from_cli_values([str(final), f" late ={other}"])

    # An omitted label comes from the parent when the leaf is checkpoint-final.
    assert [target.label for target in targets] == ["epoch-3", "late"]
    assert [target.path for target in targets] == [final.resolve(), other.resolve()]


def test_checkpoint_cli_values_refuse_collisions_and_the_reserved_label(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a" / "epoch-1"
    second = tmp_path / "b" / "epoch-1"
    for path in (first, second):
        path.mkdir(parents=True)

    with pytest.raises(ValueError, match="labels must be unique"):
        CheckpointTarget.from_cli_values([str(first), str(second)])

    with pytest.raises(ValueError, match="reserved"):
        CheckpointTarget.from_cli_values([f"base={first}"], reserved_label="base")

    with pytest.raises(ValueError, match="must be non-empty"):
        CheckpointTarget.from_cli_values(["  "])

    with pytest.raises(ValueError, match="resolved empty"):
        CheckpointTarget.from_cli_values([f"!!!={first}"])


def test_require_directory_is_what_separates_the_two_entrypoints(tmp_path: Path) -> None:
    """One entrypoint loads a published directory; another may name a weights file."""

    weights = tmp_path / "adapter.safetensors"
    weights.touch()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        CheckpointTarget.from_cli_values([f"arm={weights}"])

    (target,) = CheckpointTarget.from_cli_values([f"arm={weights}"], require_directory=False)
    assert target.path == weights.resolve()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        CheckpointTarget.from_cli_values(
            [f"arm={tmp_path / 'absent'}"],
            require_directory=False,
        )

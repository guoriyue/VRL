"""The clean-latents shard contract (GRPO diffusion-loss regularizer data)."""

from __future__ import annotations

import pytest
import torch

from vrl.trainers.data.sft_latents import (
    CleanTargetRef,
    load_sft_latents,
    save_sft_latents,
)


def test_sft_latents_round_trip(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    latents = {
        "targets/fox.mp4": torch.randn(4, 2, 8, 8),
        "targets/car.mp4": torch.randn(4, 2, 8, 8),
    }
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="nvidia/x",
        model_revision="main",
        latents_by_target=latents,
    )

    loaded = load_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="nvidia/x",
        model_revision="main",
    )
    assert set(loaded) == set(latents)
    for target, value in latents.items():
        torch.testing.assert_close(loaded[target], value)
        assert loaded[target].device.type == "cpu"


@pytest.mark.parametrize("stage", ["serialize", "flush"])
def test_failed_shard_publication_preserves_existing_file(tmp_path, monkeypatch, stage):
    from pathlib import Path

    import vrl.utils.artifacts as artifacts

    path = tmp_path / "sft.pt"
    path.write_bytes(b"previous shard")

    def fail_save(payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial replacement")
        else:
            Path(destination).write_bytes(b"partial replacement")
        raise RuntimeError("serialization failed")

    def fail_flush(fd):
        raise OSError("flush failed")

    if stage == "serialize":
        monkeypatch.setattr(torch, "save", fail_save)
    else:
        monkeypatch.setattr(artifacts.os, "fsync", fail_flush)
    with pytest.raises((RuntimeError, OSError), match="failed"):
        save_sft_latents(
            path,
            family="f",
            model_path="m",
            model_revision="r",
            latents_by_target={"target": torch.zeros(1)},
        )
    assert path.read_bytes() == b"previous shard"
    assert list(tmp_path.iterdir()) == [path]


def test_shard_publication_preserves_destination_symlink(tmp_path):
    target = tmp_path / "target.pt"
    link = tmp_path / "linked.pt"
    link.symlink_to(target)
    save_sft_latents(
        link,
        family="f",
        model_path="m",
        model_revision="r",
        latents_by_target={"target": torch.ones(1)},
    )
    assert link.is_symlink()
    torch.testing.assert_close(load_sft_latents(target)["target"], torch.ones(1))


def test_sft_latents_rejects_family_mismatch(tmp_path) -> None:
    shard = tmp_path / "sft.pt"
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        model_path="x",
        model_revision="main",
        latents_by_target={"target.mp4": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match="not interchangeable"):
        load_sft_latents(shard, family="wan_2_1")


@pytest.mark.parametrize(
    ("saved", "loaded", "match"),
    [
        (
            {"model_path": "nvidia/source", "model_revision": "main"},
            {"model_path": "nvidia/training", "model_revision": "main"},
            r"model\.path",
        ),
        (
            {"model_path": "nvidia/model", "model_revision": "base"},
            {"model_path": "nvidia/model", "model_revision": "post-trained"},
            r"model\.revision",
        ),
    ],
    ids=["path", "revision"],
)
def test_sft_latents_rejects_producer_model_mismatch(tmp_path, saved, loaded, match) -> None:
    """A shard names the model that produced it; loading it for another
    path or revision is refused by the field that differs."""

    shard = tmp_path / "sft.pt"
    save_sft_latents(
        shard,
        family="cosmos-predict2",
        **saved,
        latents_by_target={"target.mp4": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match=match):
        load_sft_latents(shard, family="cosmos-predict2", **loaded)


def test_sft_latents_missing_file_names_the_producer(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="encode_targets"):
        load_sft_latents(tmp_path / "nope.pt")


def test_sft_latents_refuses_empty_shard(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_sft_latents(
            tmp_path / "s.pt",
            family="f",
            model_path="m",
            model_revision="",
            latents_by_target={},
        )


def test_sft_latents_rejects_foreign_payload(tmp_path) -> None:
    shard = tmp_path / "junk.pt"
    torch.save({"weights": torch.zeros(1)}, shard)
    with pytest.raises(ValueError, match="not an sft-latents shard"):
        load_sft_latents(shard)


@pytest.mark.parametrize("version", [2.9, "2", True, None])
def test_shard_schema_version_is_not_coerced(tmp_path, version):
    path = tmp_path / "shard.pt"
    torch.save({"schema_version": version, "latents": {"target": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="unsupported sft-latents schema_version"):
        load_sft_latents(path)


@pytest.mark.parametrize("key", [1, "", "   "])
def test_shard_preserves_target_key_type_instead_of_stringifying(tmp_path, key):
    path = tmp_path / "shard.pt"
    latents = {key: torch.zeros(1), "1": torch.ones(1)}
    with pytest.raises(ValueError, match="target keys must be non-empty strings"):
        save_sft_latents(
            path, family="f", model_path="m", model_revision="", latents_by_target=latents
        )
    assert not path.exists()
    torch.save({"schema_version": 2, "latents": latents}, path)
    with pytest.raises(ValueError, match="target keys must be non-empty strings"):
        load_sft_latents(path)


@pytest.mark.parametrize("value", [123, True, ["target.mp4"], {"path": "target.mp4"}])
@pytest.mark.parametrize("as_example", [False, True])
def test_clean_target_identity_rejects_non_string(value, as_example):
    from vrl.trainers.data.prompts import PromptExample

    source = (
        PromptExample(prompt="p", target_video=value) if as_example else {"target_video": value}
    )
    with pytest.raises(ValueError, match="target_video must be a string"):
        CleanTargetRef.from_source(source)


def test_clean_target_identity_keeps_existing_empty_and_whitespace_semantics():
    assert CleanTargetRef.from_source(
        {"target_image": " ", "target_video": " targets/a.mp4 "}
    ) == CleanTargetRef(field="target_video", key="targets/a.mp4")
    with pytest.raises(ValueError, match="exactly one clean target"):
        CleanTargetRef.from_source({"target_image": "a.png", "target_video": "b.mp4"})

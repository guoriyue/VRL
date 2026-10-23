from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.rewards.types import RewardOutput
from vrl.scripts.eval import image_checkpoint_eval as checkpoint_eval
from vrl.trainers.data.prompts import PromptExample


def _checkpoint(path, epoch):
    path.mkdir(parents=True)
    (path / "checkpoint.pt").write_bytes(b"checkpoint")
    (path / "checkpoint_meta.json").write_text(
        json.dumps({"completed_epoch": epoch, "checkpoint_file_bytes": 10}),
        encoding="utf-8",
    )
    return path


@dataclass
class _FakePrecision:
    dtype: str = "float32"


@pytest.fixture
def evaluation(tmp_path):
    _checkpoint(tmp_path / "checkpoint-3", 3)
    _checkpoint(tmp_path / "checkpoint-8", 8)
    return checkpoint_eval.CheckpointEvaluation(
        entry=SimpleNamespace(family="test"),
        model_build=SimpleNamespace(
            device="cpu",
            parameter_dtype="float32",
            precision=_FakePrecision(),
            checkpoint_identity=lambda: {"family": "test"},
        ),
        model_identity={"family": "test"},
        targets=checkpoint_eval.discover_targets(tmp_path),
        prompts=(
            checkpoint_eval.EvalPrompt(
                9,
                PromptExample(
                    prompt="A shop sign.",
                    target_text="OPEN",
                    metadata={"scene": "street"},
                ),
            ),
        ),
        sampling=checkpoint_eval.ImageSampling(8, 8, 1, 0.0, 128),
        reward={"components": {"fake": 1.0}, "kwargs": {}, "inference": {}},
        samples_per_prompt=2,
        seed=17,
        negative_prompt="",
        config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        training_reward_components=("fake",),
    )


@pytest.fixture
def generation(tmp_path, evaluation):
    directory = tmp_path / "evaluation"
    rows = list(evaluation.cells())
    for row in rows:
        path = directory / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (row["epoch"], row["sample_index"], 0)).save(path)
        row["image_sha256"] = checkpoint_eval.sha256_file(path)
    archive = checkpoint_eval.EvaluationArchive(directory, evaluation)
    archive.publish_generation(rows)
    return archive, rows


def test_discovers_arbitrary_complete_epochs_and_hashes(tmp_path):
    for epoch in (3, 8):
        _checkpoint(tmp_path / f"checkpoint-{epoch}", epoch)
    _checkpoint(tmp_path / "checkpoint-final", 8)
    (tmp_path / "checkpoint-11").mkdir()
    targets = checkpoint_eval.discover_targets(tmp_path)
    assert [(target.label, target.epoch) for target in targets] == [
        ("base", 0),
        ("checkpoint-3", 3),
        ("checkpoint-final", 8),
    ]
    (tmp_path / "checkpoint-3" / "checkpoint.pt").write_bytes(b"checkpoinu")
    rediscovered = checkpoint_eval.discover_targets(tmp_path)
    assert rediscovered[1].checkpoint_sha256 != targets[1].checkpoint_sha256
    with pytest.raises(ValueError, match="no complete checkpoint"):
        checkpoint_eval.discover_targets(tmp_path / "checkpoint-11")


def test_explicit_checkpoints_become_distinct_labelled_arms(tmp_path):
    run = tmp_path / "run"
    own_final = _checkpoint(run / "checkpoint-final", 7)
    (own_final / "lora_weights").mkdir()
    other_final = _checkpoint(tmp_path / "epoch-3" / "checkpoint-final", 3)

    # The adapter weights mean their checkpoint; a padded label is trimmed. A
    # checkpoint-final leaf is named after its run unless it is the run evaluated.
    targets = checkpoint_eval.discover_targets(
        run,
        checkpoint_specs=(f" candidate ={own_final / 'lora_weights'}", str(other_final)),
    )
    assert [(t.label, t.epoch, t.path) for t in targets] == [
        ("base", 0, None),
        ("candidate", 7, own_final),
        ("epoch-3", 3, other_final),
    ]
    assert targets[1].checkpoint_sha256 == checkpoint_eval.sha256_file(own_final / "checkpoint.pt")
    assert checkpoint_eval.discover_targets(run, checkpoint_specs=(str(own_final),))[1].label == (
        "checkpoint-final"
    )
    for specs, message in (
        ((str(own_final), f"checkpoint-final={other_final}"), "unique"),
        ((f"base={own_final}",), "not 'base'"),
        ((f"!!!={own_final}",), "resolved empty"),
        ((str(tmp_path / "missing"),), "incomplete checkpoint"),
    ):
        with pytest.raises(ValueError, match=message):
            checkpoint_eval.discover_targets(run, checkpoint_specs=specs)


def test_seed_grid_is_dense_and_shared_by_every_arm(evaluation):
    prompts = tuple(
        checkpoint_eval.EvalPrompt(index, PromptExample(f"prompt {index}")) for index in range(3)
    )
    cells = list(replace(evaluation, prompts=prompts).cells())
    for label in ("base", "checkpoint-3", "checkpoint-8"):
        seeds = [cell["seed"] for cell in cells if cell["checkpoint_label"] == label]
        assert seeds == list(range(17, 23))


def test_prompt_selection_limits_rows_and_preserves_reward_metadata(evaluation):
    examples = [
        PromptExample(
            prompt=f"scene {index}",
            target_text="OPEN",
            metadata={"scene": "street", "object_class": "car", "expected_count": 2},
        )
        for index in range(3)
    ]
    selected = checkpoint_eval.select_prompts(examples, limit=2)
    assert [row.manifest_index for row in selected] == [0, 1]
    assert selected[0].example.reward_metadata() == {
        "scene": "street",
        "object_class": "car",
        "expected_count": 2,
        "target_text": "OPEN",
    }
    with pytest.raises(ValueError, match="reference_video"):
        checkpoint_eval.select_prompts([PromptExample("scene", reference_video="input.mp4")])
    # Reference-image edits are selected; generation routes them through the family
    # executor, and every grid cell carries the references for the edit reward.
    edit = checkpoint_eval.select_prompts([PromptExample("scene", reference_images=["input.png"])])
    edit_cell = next(replace(evaluation, prompts=tuple(edit)).cells())
    assert edit_cell["reward_metadata"]["reference_images"] == ["input.png"]
    for example in (
        PromptExample("scene", request_overrides={"width": 64}),
        PromptExample("scene", task_type="text_to_video"),
    ):
        with pytest.raises(ValueError):
            checkpoint_eval.select_prompts([example])
    assert checkpoint_eval.select_prompts([PromptExample("scene", task_type="text_to_image")])


def test_overrides_compose_the_evaluation_on_the_run_config(tmp_path, evaluation, monkeypatch):
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        '{"prompt":"A sign.","target_text":"OPEN","metadata":{"scene":"shop"}}\n', encoding="utf-8"
    )
    run_config = tmp_path / "resolved_config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"family": "sd3_5", "use_lora": False},
                "precision": {"training": {"dtype": "fp32"}, "float32_precision": "ieee"},
                "sampling": {
                    "width": 8,
                    "height": 8,
                    "num_steps": 7,
                    "guidance_scale": 2.0,
                    "max_sequence_length": 8,
                    "negative_prompt": "blurry",
                },
                "data": {
                    "manifest": str(manifest),
                    "eval_manifest": str(manifest),
                    "preprocessing": {},
                    "sampler": {"type": "sequential_window"},
                },
                "reward": {"components": {"ocr": 1.0}},
            }
        ),
        run_config,
    )
    monkeypatch.setattr(
        "vrl.models.families.registry.ModelFamilyEntry.resolve_model_build",
        lambda *_args, **_kwargs: evaluation.model_build,
    )

    def resolve(*overrides):
        args = checkpoint_eval.build_parser().parse_args(["--run-dir", str(tmp_path), *overrides])
        return checkpoint_eval.CheckpointEvaluation.from_args(args)

    plain = resolve()
    assert [target.label for target in plain.targets] == ["base", "checkpoint-3", "checkpoint-8"]
    assert (plain.sampling.num_steps, plain.negative_prompt) == (7, "blurry")
    assert plain.training_reward_components == ("ocr",)
    assert next(plain.cells())["reward_metadata"] == {"scene": "shop", "target_text": "OPEN"}

    resolved = resolve(
        "sampling.width=16",
        "sampling.num_steps=40",
        "reward.components.fake=1.0",
        "reward.kwargs.fake.rubric=lighting",
        "reward.kwargs.fake.ocr.debug_dir=/training/debug",
        "reward.kwargs.fake.guard.scored_rollout_dir=/training/scored",
    )
    assert (resolved.sampling.width, resolved.sampling.num_steps) == (16, 40)
    assert resolved.sampling.guidance_scale == 2.0
    assert resolved.reward["components"] == {"ocr": 1.0, "fake": 1.0}
    assert resolved.reward["kwargs"]["fake"] == {"rubric": "lighting", "ocr": {}, "guard": {}}
    assert resolved.training_reward_components == ("ocr",)
    assert resolved.record() != plain.record()
    assert resolved.config_sha256 != plain.config_sha256
    for key, value in (("images_per_call", 2), ("expected_group_size", 8)):
        with pytest.raises(ValueError, match=key):
            resolve(f"reward.kwargs.fake.guard.{key}={value}")
    assert OmegaConf.load(run_config).sampling.num_steps == 7

    # A reward preset replaces the run's reward rather than adding to it.
    judged = resolve("+reward=aesthetic")
    assert list(judged.reward["components"]) == ["aesthetic"]
    assert judged.training_reward_components == ("ocr",)

    # --config is the base model alone, with no training reward to overlap.
    args = checkpoint_eval.build_parser().parse_args(["--config", str(run_config)])
    base = checkpoint_eval.CheckpointEvaluation.from_args(args)
    assert [target.label for target in base.targets] == ["base"]
    assert base.training_reward_components == ()


def test_base_disables_adapter_before_checkpoint_restores(tmp_path, evaluation, monkeypatch):
    events = []

    class FakeModel:
        adapter_enabled = True

        def eval(self):
            return self

        @contextlib.contextmanager
        def disable_adapter(self):
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

    model = FakeModel()
    evaluation.entry.build_rollout = lambda *_args, **_kwargs: SimpleNamespace(model=model)

    def generate(_model, **kwargs):
        events.append(("generate", model.adapter_enabled, kwargs["seed"]))
        return Image.new("RGB", (8, 8))

    def load(path):
        checkpoint = SimpleNamespace(next_epoch=int(path.name.rsplit("-", 1)[1]))

        def restore(**kwargs):
            assert kwargs["expected_model_identity"] == evaluation.model_identity
            events.append(("restore", checkpoint.next_epoch))

        checkpoint.restore_model = restore
        return checkpoint

    monkeypatch.setattr(checkpoint_eval, "generate_image", generate)
    monkeypatch.setattr("vrl.trainers.checkpointing.TrainingCheckpoint.load", load)
    monkeypatch.setattr("vrl.utils.cuda_memory.release_cuda_memory", lambda: None)
    rows = evaluation.generate(tmp_path / "generated")
    assert events == [
        ("generate", False, 17),
        ("generate", False, 18),
        ("restore", 3),
        ("generate", True, 17),
        ("generate", True, 18),
        ("restore", 8),
        ("generate", True, 17),
        ("generate", True, 18),
    ]
    assert len(rows) == 6 and model.adapter_enabled is True


def test_reward_batches_are_blinded_consistently_across_seeds(generation, monkeypatch):
    archive, rows = generation
    batches, shutdowns = [], []

    class FakeReward:
        async def score_batch(self, samples):
            batches.append(samples)
            values = tuple(index / 10 for index in range(len(samples)))
            return RewardOutput(tuple(value * 2 for value in values), {"fake": values})

        async def shutdown(self):
            shutdowns.append(True)

    def build_reward(_weights, **kwargs):
        # Offline scoring must not inherit training's unowned Ray deployment.
        assert kwargs["inference_configs"]["fake"].kind == "in_process"
        assert "artifact_dir" not in kwargs["reward_kwargs"]["fake"]
        return FakeReward()

    monkeypatch.setattr(
        "vrl.rewards.functions.registry.MultiReward.from_mapping",
        build_reward,
    )
    scored = asyncio.run(checkpoint_eval.score_images(archive.evaluation, rows, archive.directory))
    order = archive.evaluation.blind_orders()[0]
    for sample_index, batch in enumerate(batches):
        scored_sample = [row for row in scored if row["sample_index"] == sample_index]
        assert [row["checkpoint_label"] for row in scored_sample] == order
        assert [row["r_fake"] for row in scored_sample] == pytest.approx([0, 0.1, 0.2])
        assert [row["r_total"] for row in scored_sample] == pytest.approx([0, 0.2, 0.4])
        assert all(sample.metadata["target_text"] == "OPEN" for sample in batch)
        assert all(
            "checkpoint" not in sample.sample_id and "base" not in sample.sample_id
            for sample in batch
        )
    assert len(batches) == 2 and shutdowns == [True]


def test_failed_scoring_reuses_verified_images_without_materialization(generation, monkeypatch):
    archive, rows = generation
    monkeypatch.setattr(
        checkpoint_eval.CheckpointEvaluation, "from_args", lambda _args: archive.evaluation
    )
    monkeypatch.setattr(
        checkpoint_eval.CheckpointEvaluation,
        "generate",
        lambda *_args: pytest.fail("must reuse generation"),
    )

    async def fail_scoring(_evaluation, saved, _directory):
        assert saved == rows
        raise RuntimeError("reward unavailable")

    monkeypatch.setattr(checkpoint_eval, "score_images", fail_scoring)
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="reward unavailable"):
            checkpoint_eval.main(
                [
                    "--run-dir",
                    str(archive.directory.parent),
                    "--output-dir",
                    str(archive.directory),
                ]
            )
    assert archive.load_generation() == rows
    assert not (archive.directory / "report").exists()


@pytest.mark.parametrize("change", ["protocol", "image", "extra"])
def test_generation_archive_rejects_changed_inputs_or_files(generation, change):
    archive, rows = generation
    if change == "protocol":
        archive = replace(archive, evaluation=replace(archive.evaluation, config_sha256="f" * 64))
    elif change == "image":
        Image.new("RGB", (8, 8), "white").save(archive.directory / rows[0]["image_path"])
    else:
        Image.new("RGB", (8, 8)).save(archive.directory / "images" / "extra.png")
    with pytest.raises(ValueError, match=r"protocol|integrity|not exact"):
        archive.load_generation()


def test_completed_report_is_immutable(generation):
    archive, rows = generation
    scored = [
        {
            **row,
            "image_path": str(archive.directory / row["image_path"]),
            "r_fake": row["epoch"] / 10,
        }
        for row in rows
    ]
    summary = archive.publish_report(scored)
    assert summary["statistical_unit"] == "prompt"
    report = archive.directory / "report"
    assert (report / "curve.png").is_file()
    assert (report / "contact_sheets" / "prompt0000.png").is_file()
    assert json.loads((report / "provenance.json").read_text())["training_reward_overlap"] == [
        "fake"
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        archive.reject_completed()

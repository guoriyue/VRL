from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY, RewardOutput
from vrl.scripts.eval import image_checkpoint_eval as checkpoint_eval
from vrl.trainers.data.prompts import PromptExample


def _checkpoint(path, epoch, *, uses_lora=False):
    path.mkdir()
    (path / "checkpoint.pt").write_bytes(b"checkpoint")
    (path / "checkpoint_meta.json").write_text(
        json.dumps(
            {"completed_epoch": epoch, "uses_lora": uses_lora, "checkpoint_file_bytes": 10}
        ),
        encoding="utf-8",
    )
    return path


@dataclass
class _FakePrecision:
    dtype: str = "float32"


@pytest.fixture
def plan(tmp_path):
    _checkpoint(tmp_path / "checkpoint-3", 3)
    _checkpoint(tmp_path / "checkpoint-8", 8)
    return checkpoint_eval.EvaluationPlan(
        resolved_model=SimpleNamespace(
            identity={"family": "test"},
            entry=SimpleNamespace(family="test"),
            build=SimpleNamespace(
                device="cpu", parameter_dtype="float32", precision=_FakePrecision()
            ),
        ),
        targets=checkpoint_eval.discover_targets(tmp_path, uses_lora=False),
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
        blind_seed=4,
        negative_prompt="",
        reward_device="cpu",
        tie_epsilon=0.02,
        bootstrap_resamples=20,
        config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        training_reward_components=("fake",),
    )


@pytest.fixture
def generation(tmp_path, plan):
    directory = tmp_path / "evaluation"
    rows = list(plan.cells())
    for row in rows:
        path = directory / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (row["epoch"], row["sample_index"], 0)).save(path)
        row["image_sha256"] = checkpoint_eval.sha256_file(path)
    archive = checkpoint_eval.EvaluationArchive(directory, plan)
    archive.publish_generation(rows)
    return archive, rows


def test_discovers_arbitrary_complete_epochs_and_hashes(tmp_path):
    for epoch in (3, 8):
        _checkpoint(tmp_path / f"checkpoint-{epoch}", epoch)
    _checkpoint(tmp_path / "checkpoint-final", 8)
    (tmp_path / "checkpoint-11").mkdir()
    targets = checkpoint_eval.discover_targets(tmp_path, uses_lora=False)
    assert [(target.label, target.epoch) for target in targets] == [
        ("base", 0),
        ("checkpoint-3", 3),
        ("checkpoint-final", 8),
    ]
    (tmp_path / "checkpoint-3" / "checkpoint.pt").write_bytes(b"checkpoinu")
    selected = checkpoint_eval.discover_targets(tmp_path, epochs=(3,), uses_lora=False)
    assert selected[1].checkpoint_sha256 != targets[1].checkpoint_sha256
    with pytest.raises(ValueError, match="missing complete checkpoint"):
        checkpoint_eval.discover_targets(tmp_path, epochs=(11,), uses_lora=False)


def test_explicit_lora_export_uses_parent_checkpoint_and_rejects_wrong_mode(tmp_path):
    path = _checkpoint(tmp_path / "checkpoint-final", 7, uses_lora=True)
    (path / "lora_weights").mkdir()
    targets = checkpoint_eval.discover_targets(
        tmp_path,
        checkpoint_specs=(f"candidate={path / 'lora_weights'}",),
        uses_lora=True,
    )
    assert (targets[1].label, targets[1].epoch, targets[1].path) == ("candidate", 7, path)
    assert targets[1].checkpoint_sha256 == checkpoint_eval.sha256_file(path / "checkpoint.pt")
    with pytest.raises(ValueError, match="uses_lora"):
        checkpoint_eval.load_target(str(path), uses_lora=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        checkpoint_eval.discover_targets(
            tmp_path, checkpoint_specs=(str(path),), epochs=(7,), uses_lora=True
        )


def test_prompt_selection_uses_user_strata_and_preserves_reward_metadata():
    examples = [
        PromptExample(
            prompt=f"{scene} {index}",
            target_text="OPEN",
            references=["reference.png"],
            metadata={"scene": scene, "object_class": "car", "expected_count": 2},
        )
        for scene in ("street", "shop")
        for index in range(3)
    ]
    selected = checkpoint_eval.select_prompts(examples, strata=("scene",), per_stratum=2)
    assert [row.manifest_index for row in selected] == [0, 1, 3, 4]
    assert selected[0].example.reward_metadata() == {
        "scene": "street",
        "object_class": "car",
        "expected_count": 2,
        "target_text": "OPEN",
        "references": ["reference.png"],
    }
    with pytest.raises(ValueError, match="fewer rows"):
        checkpoint_eval.select_prompts(examples, strata=("scene",), per_stratum=4)
    with pytest.raises(ValueError, match="text-conditioned"):
        checkpoint_eval.select_prompts([PromptExample("scene", reference_image="input.png")])
    for example in (
        PromptExample("scene", request_overrides={"width": 64}),
        PromptExample("scene", task_type="text_to_video"),
    ):
        with pytest.raises(ValueError):
            checkpoint_eval.select_prompts([example])
    assert checkpoint_eval.select_prompts([PromptExample("scene", task_type="text_to_image")])


def test_independent_policy_does_not_replace_run_sampling(tmp_path, plan, monkeypatch):
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
                },
                "reward": {"components": {"ocr": 1.0}},
            }
        ),
        run_config,
    )
    policy_path = tmp_path / "policy.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "sampling": {"num_steps": 99},
                "data": {"eval_manifest": str(manifest)},
                "reward": {
                    "components": {"fake": 1.0, "observation": 0.0},
                    "kwargs": {
                        "fake": {
                            "rubric": "lighting",
                            "ocr": {"debug_dir": "/training/debug"},
                            "guard": {"scored_rollout_dir": "/training/scored"},
                        }
                    },
                },
            }
        ),
        policy_path,
    )
    monkeypatch.setattr("vrl.run.resolve_model", lambda *_args, **_kwargs: plan.resolved_model)
    monkeypatch.setattr(
        "vrl.trainers.checkpointing.validate_checkpoint_meta_compatibility",
        lambda *_args, **_kwargs: None,
    )
    parser = checkpoint_eval.build_parser()
    args = parser.parse_args(
        ["--run-dir", str(tmp_path), "--eval-policy-config", str(policy_path)]
    )
    resolved = checkpoint_eval.resolve_plan(args)
    assert resolved.sampling.num_steps == 7
    assert resolved.reward["components"] == {"fake": 1.0, "observation": 0.0}
    assert resolved.reward["kwargs"]["fake"] == {"rubric": "lighting", "ocr": {}, "guard": {}}
    assert resolved.training_reward_components == ("ocr",)
    assert next(resolved.cells())["reward_metadata"] == {"scene": "shop", "target_text": "OPEN"}
    args.eval_policy_override = ["sampling.num_steps=40"]
    assert checkpoint_eval.resolve_plan(args).record() == resolved.record()
    args.eval_policy_override = ["reward.kwargs.fake.rubric=color"]
    assert checkpoint_eval.resolve_plan(args).record() != resolved.record()
    for key, value in (("images_per_call", 2), ("expected_group_size", 8)):
        args.eval_policy_override = [f"reward.kwargs.fake.guard.{key}={value}"]
        with pytest.raises(ValueError, match=key):
            checkpoint_eval.resolve_plan(args)
    assert OmegaConf.load(run_config).reward.components == {"ocr": 1.0}


def test_base_disables_adapter_before_checkpoint_restores(tmp_path, plan, monkeypatch):
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
    plan.resolved_model.materialize = lambda **_kwargs: SimpleNamespace(model=model)

    def generate(_model, **kwargs):
        events.append(("generate", model.adapter_enabled, kwargs["seed"]))
        return [Image.new("RGB", (8, 8))]

    def restore(checkpoint, **kwargs):
        assert kwargs["strict"] is True
        assert kwargs["expected_model_identity"] == plan.resolved_model.identity
        events.append(("restore", checkpoint.next_epoch))

    monkeypatch.setattr(checkpoint_eval, "generate_images", generate)
    monkeypatch.setattr(
        "vrl.trainers.checkpointing.TrainingCheckpoint.load",
        lambda path: SimpleNamespace(next_epoch=int(path.name.rsplit("-", 1)[1])),
    )
    monkeypatch.setattr("vrl.trainers.checkpointing.restore_model_checkpoint", restore)
    monkeypatch.setattr("vrl.utils.cuda_memory.release_cuda_memory", lambda: None)
    rows = plan.generate(tmp_path / "generated")
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
        "vrl.rewards.functions.registry.MultiReward.from_dict",
        build_reward,
    )
    scored = asyncio.run(checkpoint_eval.score_images(archive.plan, rows, archive.directory))
    order = archive.plan.blind_orders()[0]
    for sample_index, batch in enumerate(batches):
        scored_sample = [row for row in scored if row["sample_index"] == sample_index]
        assert [row["checkpoint_label"] for row in scored_sample] == order
        assert [row["r_fake"] for row in scored_sample] == pytest.approx([0, 0.1, 0.2])
        assert [row["r_total"] for row in scored_sample] == pytest.approx([0, 0.2, 0.4])
        assert all(sample.metadata["target_text"] == "OPEN" for sample in batch)
        assert len({sample.metadata[REWARD_GROUP_ID_METADATA_KEY] for sample in batch}) == 1
        assert all(
            "checkpoint" not in sample.sample_id and "base" not in sample.sample_id
            for sample in batch
        )
    assert len(batches) == 2 and shutdowns == [True]


def test_failed_scoring_reuses_verified_images_without_materialization(generation, monkeypatch):
    archive, rows = generation
    monkeypatch.setattr(checkpoint_eval, "resolve_plan", lambda _args: archive.plan)
    monkeypatch.setattr(
        checkpoint_eval.EvaluationPlan,
        "generate",
        lambda *_args: pytest.fail("must reuse generation"),
    )

    async def fail_scoring(_plan, saved, _directory):
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


@pytest.mark.parametrize("change", ["protocol", "image", "extra", "symlink"])
def test_generation_archive_rejects_changed_inputs_or_files(generation, tmp_path, change):
    archive, rows = generation
    if change == "protocol":
        archive = replace(archive, plan=replace(archive.plan, config_sha256="f" * 64))
    elif change == "image":
        Image.new("RGB", (8, 8), "white").save(archive.directory / rows[0]["image_path"])
    elif change == "extra":
        Image.new("RGB", (8, 8)).save(archive.directory / "images" / "extra.png")
    else:
        (archive.directory / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match=r"protocol|integrity|not exact|symlinks"):
        archive.load_generation()


def test_completed_report_is_immutable_and_integrity_checked(generation):
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
    (report / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        archive.reject_completed()


@pytest.fixture
def completed_training_evaluation(generation, tmp_path, monkeypatch):
    import shutil

    from vrl.trainers import trace
    from vrl.training_run_result import TrainingRunResultWriter

    archive, rows = generation
    archive.publish_report(
        [
            {
                **row,
                "image_path": str(archive.directory / row["image_path"]),
                "r_fake": row["epoch"] / 10,
            }
            for row in rows
        ]
    )
    training = tmp_path / "training"
    training.mkdir()
    shutil.copytree(tmp_path / "checkpoint-8", training / "checkpoint-final")
    (training / "metrics.csv").write_text("epoch,reward\n0,0.1\n")
    monkeypatch.setattr(
        trace.TrainingRunTrace,
        "_runtime_snapshot",
        lambda: {"environment": {"WORLD_SIZE": "1"}},
    )
    monkeypatch.setattr(
        trace.TrainingRunTrace, "_git_snapshot", lambda _path: {"available": False}
    )
    launch = trace.TrainingRunTrace.capture(
        OmegaConf.create({"seed": 17}),
        training,
        model_identity=archive.plan.resolved_model.identity,
        resumed=False,
    ).launch_path
    seal = trace.TrainingRunTrace.load(launch).seal_artifacts()
    TrainingRunResultWriter(str(training), environ={}).write()
    result_path = training / "training_run_result.json"
    result = json.loads(result_path.read_text())
    # This is fixture data for association validation, not a real trainer run.
    result["supervisor_exit_code"] = 0
    result_path.write_text(json.dumps(result))
    return seal, result_path, archive


def test_training_evaluation_association_uses_content_not_target_path(
    completed_training_evaluation,
):
    from vrl.trainers.trace import TrainingRunTrace

    seal, result, archive = completed_training_evaluation
    result = TrainingRunTrace.load(seal).verify_evaluation(result, archive)
    assert "checkpoint-8" in result["checkpoint_labels"]
    assert "attempt_id" not in result
    assert result["evaluation_content"]["files"] > len(list(archive.plan.cells()))
    assert len(result["evaluation_protocol_sha256"]) == 64


@pytest.mark.parametrize(
    "change", ["model", "checkpoint", "scores", "image", "protocol", "status"]
)
def test_training_evaluation_rejects_mismatched_evidence(completed_training_evaluation, change):
    from vrl.trainers import trace

    seal, result, archive = completed_training_evaluation
    if change in {"model", "checkpoint"}:
        # Publish an internally valid training receipt for a different run/state.
        # Association must fail even though each side independently verifies.
        launch_path = seal.with_name(seal.name.removesuffix(".artifacts.json") + ".json")
        if change == "model":
            launch = json.loads(launch_path.read_text())
            launch["model_identity"] = {"family": "different"}
            launch_path.write_text(json.dumps(launch))
        else:
            (seal.parent.parent / "checkpoint-final/checkpoint.pt").write_bytes(
                b"different weights"
            )
        seal.unlink()
        trace.TrainingRunTrace.load(launch_path).seal_artifacts()
    elif change == "scores":
        (archive.directory / "report/scores.jsonl").write_text("{}\n")
    elif change == "image":
        image = archive.directory / next(archive.plan.cells())["image_path"]
        Image.new("RGB", (8, 8), "red").save(image)
    elif change == "protocol":
        archive = checkpoint_eval.EvaluationArchive(
            archive.directory, replace(archive.plan, seed=999)
        )
    else:
        record = json.loads(result.read_text())
        record["status"] = "failed"
        result.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        trace.TrainingRunTrace.load(seal).verify_evaluation(result, archive)


def test_cli_verifies_existing_training_evaluation_without_generation_or_scoring(
    completed_training_evaluation, monkeypatch, capsys
):
    seal, _result, archive = completed_training_evaluation
    monkeypatch.setattr(checkpoint_eval, "resolve_plan", lambda _args: archive.plan)
    monkeypatch.setattr(
        checkpoint_eval.EvaluationPlan, "generate", lambda *_: pytest.fail("generation")
    )
    monkeypatch.setattr(checkpoint_eval, "score_images", lambda *_: pytest.fail("scoring"))
    checkpoint_eval.main(
        [
            "--run-dir",
            str(seal.parent.parent),
            "--output-dir",
            str(archive.directory),
            "--verify-training-evidence",
            str(seal),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert "attempt_id" not in result


def test_nested_completion_marker_is_not_exempt_from_report_integrity(
    completed_training_evaluation,
):
    _seal, _result, archive = completed_training_evaluation
    extra = archive.directory / "report/nested/evaluation_complete.json"
    extra.parent.mkdir()
    extra.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        archive.verify_report()

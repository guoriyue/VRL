from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.scripts.eval import anima_codex_quality_checkpoint_eval as checkpoint_eval
from vrl.scripts.families.cosmos.anima.generation_protocol import AnimaSampling

_TEST_PROMPT_STYLES = ("language", "tag")


def _write_complete_checkpoint(path, epoch: int, *, uses_lora: bool = False) -> None:
    path.mkdir()
    (path / "checkpoint.pt").write_bytes(b"checkpoint")
    (path / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "checkpoint_file_bytes": len(b"checkpoint"),
                "completed_epoch": epoch,
                "uses_lora": uses_lora,
            },
        ),
        encoding="utf-8",
    )


def _build_generation_stage(tmp_path):
    labels = ("base", "checkpoint-5", "checkpoint-10", "checkpoint-15", "checkpoint-final")
    targets = tuple(
        checkpoint_eval.CheckpointTarget(label, epoch, None, {})
        for label, epoch in zip(labels, (0, 5, 10, 15, 20), strict=True)
    )
    prompt = checkpoint_eval.EvalPrompt(0, 9, "single anime girl", "hands", "language")
    config_path = tmp_path / "resolved_config.yaml"
    manifest_path = tmp_path / "eval.jsonl"
    config_path.write_text("model: test\n", encoding="utf-8")
    manifest_path.write_text('{"prompt": "single anime girl"}\n', encoding="utf-8")
    output_dir = tmp_path / "evaluation"
    generated = []
    for target in targets:
        for sample_index in range(checkpoint_eval.SAMPLES_PER_PROMPT):
            path = output_dir / checkpoint_eval._image_relative_path(
                target.label,
                prompt.prompt_index,
                sample_index,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(target.epoch, sample_index, 0)).save(path)
            generated.append(
                checkpoint_eval.GeneratedImage(
                    checkpoint_label=target.label,
                    epoch=target.epoch,
                    prompt_index=prompt.prompt_index,
                    manifest_index=prompt.manifest_index,
                    sample_index=sample_index,
                    seed=checkpoint_eval._sample_seed(prompt.prompt_index, sample_index),
                    prompt=prompt.prompt,
                    bucket=prompt.bucket,
                    prompt_style=prompt.prompt_style,
                    path=path,
                    image_sha256=checkpoint_eval.sha256_file(path),
                ),
            )
    protocol = checkpoint_eval.EvaluationProtocol(
        run_dir=tmp_path,
        config_path=config_path,
        eval_manifest_path=manifest_path,
        config_sha256=checkpoint_eval.sha256_file(config_path),
        eval_manifest_sha256=checkpoint_eval.sha256_file(manifest_path),
        eval_policy_source=str(config_path),
        eval_policy_sha256="e" * 64,
        resolved_model=SimpleNamespace(identity={"family": "test"}),
        targets=targets,
        prompts=(prompt,),
        training_reward_components=("codex_image_qa",),
        sampling=AnimaSampling(
            width=8,
            height=8,
            num_steps=1,
            guidance_scale=0.0,
            max_sequence_length=1,
        ),
        negative_prompt="",
        luna_config={"command": ["unused"], "images_per_call": len(targets)},
    )
    checkpoint_eval._write_generation_stage(protocol, generated, output_dir)
    return protocol, generated, output_dir


def test_discovers_registered_full_parameter_curve_without_duplicate_epoch_20(tmp_path) -> None:
    for epoch in checkpoint_eval.CURVE_EPOCHS[:-1]:
        _write_complete_checkpoint(tmp_path / f"checkpoint-{epoch}", epoch)
    _write_complete_checkpoint(tmp_path / "checkpoint-final", 20)

    targets = checkpoint_eval._discover_targets(tmp_path)

    assert [(target.label, target.epoch) for target in targets] == [
        ("base", 0),
        ("checkpoint-5", 5),
        ("checkpoint-10", 10),
        ("checkpoint-15", 15),
        ("checkpoint-final", 20),
    ]
    checkpoint_five_sha256 = targets[1].checkpoint_sha256
    (tmp_path / "checkpoint-5" / "checkpoint.pt").write_bytes(b"checkpoinu")
    replaced_targets = checkpoint_eval._discover_targets(tmp_path)
    assert replaced_targets[1].checkpoint_sha256 != checkpoint_five_sha256


def test_early_curve_discovers_only_numbered_completed_checkpoints(tmp_path) -> None:
    for epoch in (5, 10):
        _write_complete_checkpoint(tmp_path / f"checkpoint-{epoch}", epoch)

    targets = checkpoint_eval._discover_targets(tmp_path, curve_epochs=(5, 10))

    assert [(target.label, target.epoch) for target in targets] == [
        ("base", 0),
        ("checkpoint-5", 5),
        ("checkpoint-10", 10),
    ]
    with pytest.raises(ValueError, match="epoch 15"):
        checkpoint_eval._discover_targets(tmp_path)


def test_epochs_cli_accepts_only_registered_curve_prefix() -> None:
    parser = checkpoint_eval.build_parser()

    assert parser.parse_args(["--run-dir", "run"]).epochs == checkpoint_eval.CURVE_EPOCHS
    policy_args = parser.parse_args(
        [
            "--run-dir",
            "run",
            "--eval-policy-config",
            "reward/codex_image_qa_anime_general_quality",
            "--eval-policy-override",
            "+reward=codex_image_qa_luna",
            "--eval-policy-override",
            "+dataset=anima_quality_ddrl",
        ],
    )
    assert policy_args.eval_policy_config == "reward/codex_image_qa_anime_general_quality"
    assert policy_args.eval_policy_override == [
        "+reward=codex_image_qa_luna",
        "+dataset=anima_quality_ddrl",
    ]
    assert parser.parse_args(["--run-dir", "run", "--epochs", "5,10"]).epochs == (5, 10)
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run", "--epochs", "5,15"])


def test_explicit_lora_export_resolves_to_checkpoint_source_of_truth(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-final"
    _write_complete_checkpoint(checkpoint, 1, uses_lora=True)
    (checkpoint / "lora_weights").mkdir()

    targets = checkpoint_eval._discover_explicit_targets(
        (f"ddrl-canary={checkpoint / 'lora_weights'}",),
        expected_uses_lora=True,
    )

    assert [(target.label, target.epoch, target.path) for target in targets] == [
        ("base", 0, None),
        ("ddrl-canary", 1, checkpoint),
    ]
    parser = checkpoint_eval.build_parser()
    args = parser.parse_args(
        ["--run-dir", str(tmp_path), "--checkpoint", str(checkpoint / "lora_weights")],
    )
    assert args.checkpoint == [str(checkpoint / "lora_weights")]


def test_explicit_checkpoint_rejects_adapter_mode_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-final"
    _write_complete_checkpoint(checkpoint, 1, uses_lora=True)

    with pytest.raises(ValueError, match=r"does not match resolved model\.use_lora=False"):
        checkpoint_eval._discover_explicit_targets(
            (str(checkpoint),),
            expected_uses_lora=False,
        )


def test_protocol_hash_binds_selected_epochs(tmp_path) -> None:
    protocol, _generated, _output_dir = _build_generation_stage(tmp_path)
    luna_config = checkpoint_eval._resolve_luna_config(
        OmegaConf.create(
            {
                "reward": {
                    "kwargs": {
                        "codex_image_qa": {
                            "command": ["codex", "--model", "gpt-5.6-luna"],
                            "images_per_call": 5,
                        },
                    },
                },
            },
        ),
        arm_count=3,
    )
    early_protocol = replace(
        protocol,
        targets=protocol.targets[:3],
        luna_config=luna_config,
    )

    assert early_protocol.luna_config["images_per_call"] == 3
    assert checkpoint_eval._protocol_sha256(early_protocol) != checkpoint_eval._protocol_sha256(
        protocol,
    )
    assert checkpoint_eval._evaluation_protocol_record(early_protocol)["curve_epochs"] == [5, 10]
    assert checkpoint_eval._evaluation_protocol_record(early_protocol)[
        "training_reward_components"
    ] == ["codex_image_qa"]
    assert checkpoint_eval._protocol_sha256(
        replace(early_protocol, eval_policy_sha256="a" * 64),
    ) != checkpoint_eval._protocol_sha256(early_protocol)
    assert checkpoint_eval._protocol_sha256(
        replace(early_protocol, training_reward_components=("ocr",)),
    ) != checkpoint_eval._protocol_sha256(early_protocol)


def test_evaluation_policy_uses_bundled_config_without_replacing_run_config(tmp_path) -> None:
    run_config_path = tmp_path / "resolved_config.yaml"
    run_config_path.write_text("reward: {components: {ocr: 1.0}}\n", encoding="utf-8")
    run_cfg = OmegaConf.create(
        {
            "sampling": {"num_steps": 20},
            "data": {"eval_manifest": "datasets/ocr/test.txt"},
            "reward": {"components": {"ocr": 1.0}},
        },
    )

    policy = checkpoint_eval._resolve_evaluation_policy(
        run_cfg,
        run_config_path=run_config_path,
        policy_config="reward/codex_image_qa_anime_general_quality",
        policy_overrides=("+reward=codex_image_qa_luna", "+dataset=anima_quality_ddrl"),
        arm_count=5,
    )

    assert json.loads(policy.source) == {
        "config": "reward/codex_image_qa_anime_general_quality",
        "overrides": ["+reward=codex_image_qa_luna", "+dataset=anima_quality_ddrl"],
    }
    assert policy.eval_manifest_path == (
        Path("datasets/anima/quality_v1/eval_prompts.jsonl").resolve()
    )
    assert len(policy.sha256) == 64
    assert policy.luna_config["images_per_call"] == 5
    assert "gpt-5.6-luna" in policy.luna_config["command"]
    assert checkpoint_eval._resolve_sampling(run_cfg).num_steps == 20
    assert dict(run_cfg.reward.components) == {"ocr": 1.0}


def test_evaluation_policy_overlays_saved_run_without_changing_generation(tmp_path) -> None:
    run_config_path = tmp_path / "resolved_config.yaml"
    run_cfg = OmegaConf.create(
        {"sampling": {"num_steps": 20}, "reward": {"components": {"ocr": 1.0}}},
    )
    OmegaConf.save(run_cfg, run_config_path)
    overrides = (
        "+reward=codex_image_qa_anime_general_quality",
        "+reward=codex_image_qa_luna",
        "+dataset=anima_quality_ddrl",
    )
    policy = checkpoint_eval._resolve_evaluation_policy(
        run_cfg,
        run_config_path=run_config_path,
        policy_config=None,
        policy_overrides=overrides,
        arm_count=2,
    )
    changed = checkpoint_eval._resolve_evaluation_policy(
        run_cfg,
        run_config_path=run_config_path,
        policy_config=None,
        policy_overrides=(*overrides, "sampling.num_steps=40"),
        arm_count=2,
    )
    changed_rubric = checkpoint_eval._resolve_evaluation_policy(
        run_cfg,
        run_config_path=run_config_path,
        policy_config=None,
        policy_overrides=(*overrides, "reward.kwargs.codex_image_qa.tile_size=256"),
        arm_count=2,
    )

    assert policy.sha256 == changed.sha256
    assert policy.sha256 != changed_rubric.sha256
    assert json.loads(policy.source)["config"] == str(run_config_path)
    assert run_cfg == OmegaConf.load(run_config_path)
    assert checkpoint_eval._resolve_sampling(run_cfg).num_steps == 20
    assert dict(run_cfg.reward.components) == {"ocr": 1.0}


def test_evaluation_policy_defaults_to_the_run_config(tmp_path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text('{"prompt": "A held-out prompt."}\n', encoding="utf-8")
    run_config_path = tmp_path / "resolved_config.yaml"
    run_config_path.write_text("model: test\n", encoding="utf-8")
    run_cfg = OmegaConf.create(
        {
            "data": {"eval_manifest": str(manifest_path)},
            "reward": {
                "kwargs": {
                    "codex_image_qa": {
                        "command": ["codex", "--model", "gpt-5.6-luna"],
                    },
                },
            },
        },
    )

    policy = checkpoint_eval._resolve_evaluation_policy(
        run_cfg,
        run_config_path=run_config_path,
        policy_config=None,
        arm_count=2,
    )

    assert policy.source == str(run_config_path)
    assert policy.eval_manifest_path == manifest_path
    assert policy.luna_config["images_per_call"] == 2


def test_balanced_prompt_selection_uses_manifest_taxonomy_and_training_sampling_defaults() -> None:
    examples = []
    for bucket in ("action", "hands"):
        for style in _TEST_PROMPT_STYLES:
            for index in range(3):
                examples.append(
                    SimpleNamespace(
                        prompt=f"{bucket}-{style}-{index}",
                        metadata={"bucket": bucket, "prompt_style": style},
                    ),
                )

    selected = checkpoint_eval._select_balanced_prompts(examples)
    sampling = checkpoint_eval._resolve_sampling(
        OmegaConf.create(
            {
                "sampling": {
                    "width": 512,
                    "height": 512,
                    "num_steps": 20,
                    "guidance_scale": 4.5,
                },
            },
        ),
    )

    assert len(selected) == 8
    assert {(prompt.bucket, prompt.prompt_style) for prompt in selected} == {
        (bucket, style) for bucket in ("action", "hands") for style in _TEST_PROMPT_STYLES
    }
    assert sampling.max_sequence_length == 128


def test_balanced_prompt_selection_accepts_and_records_one_natural_language_style(
    tmp_path,
) -> None:
    examples = [
        SimpleNamespace(
            prompt=f"{bucket} natural sentence {index}.",
            metadata={"bucket": bucket, "prompt_style": "natural_language"},
        )
        for bucket in ("action", "hands")
        for index in range(checkpoint_eval.PROMPTS_PER_BUCKET_STYLE)
    ]

    selected = checkpoint_eval._select_balanced_prompts(examples)
    protocol, _generated, _output_dir = _build_generation_stage(tmp_path)
    provenance = checkpoint_eval._provenance_record(
        replace(protocol, prompts=selected),
        [],
        {},
    )

    assert len(selected) == 4
    assert {prompt.bucket for prompt in selected} == {"action", "hands"}
    assert {prompt.prompt_style for prompt in selected} == {"natural_language"}
    assert provenance["heldout_manifest"]["selection"]["prompt_styles"] == [
        "natural_language",
    ]
    assert provenance["training_reward_components"] == ["codex_image_qa"]


def test_balanced_prompt_selection_can_consume_a_larger_formal_stratum(
    tmp_path,
) -> None:
    examples = [
        SimpleNamespace(
            prompt=f"{bucket} formal prompt {index}.",
            metadata={"bucket": bucket, "prompt_style": "natural_language"},
        )
        for bucket in ("palette", "lighting")
        for index in range(6)
    ]

    selected = checkpoint_eval._select_balanced_prompts(
        examples,
        prompts_per_bucket_style=6,
    )
    protocol, _generated, _output_dir = _build_generation_stage(tmp_path)
    provenance = checkpoint_eval._provenance_record(
        replace(protocol, prompts=selected),
        [],
        {},
    )

    assert len(selected) == 12
    assert provenance["heldout_manifest"]["selection"]["per_bucket_style"] == 6
    with pytest.raises(ValueError, match="must be >= 1"):
        checkpoint_eval._select_balanced_prompts(
            examples,
            prompts_per_bucket_style=0,
        )


def test_base_arm_disables_training_lora_before_paired_generation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_adapter_states = []

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

    def fake_generate(runtime, **_kwargs):
        seen_adapter_states.append(runtime.adapter_enabled)
        return [Image.new("RGB", (8, 8), color="white")]

    monkeypatch.setattr(
        "vrl.scripts.families.cosmos.anima.generate.generate_images",
        fake_generate,
    )
    monkeypatch.setattr("vrl.utils.cuda_memory.release_cuda_memory", lambda: None)
    protocol = SimpleNamespace(
        resolved_model=SimpleNamespace(
            identity={"family": "test"},
            materialize=lambda **_kwargs: SimpleNamespace(model=model),
        ),
        targets=(checkpoint_eval.CheckpointTarget("base", 0, None, {}),),
        prompts=(
            checkpoint_eval.EvalPrompt(
                0,
                0,
                "A natural anime portrait.",
                "portrait",
                "natural_language",
            ),
        ),
        sampling=AnimaSampling(
            width=8,
            height=8,
            num_steps=1,
            guidance_scale=0.0,
            max_sequence_length=1,
        ),
        negative_prompt="",
    )
    output_dir = tmp_path / "evaluation"

    generated = checkpoint_eval._generate_curve(protocol, output_dir)

    assert len(generated) == checkpoint_eval.SAMPLES_PER_PROMPT
    assert seen_adapter_states == [False, False]
    assert model.adapter_enabled is True


def test_luna_scores_map_back_through_one_blind_arm_order_for_both_seeds(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ("base", "checkpoint-5", "checkpoint-10", "checkpoint-15", "checkpoint-final")
    targets = tuple(
        checkpoint_eval.CheckpointTarget(label, epoch, None, {})
        for label, epoch in zip(labels, (0, 5, 10, 15, 20), strict=True)
    )
    prompt = checkpoint_eval.EvalPrompt(0, 9, "single anime girl", "hands", "language")
    generated = []
    for target in targets:
        for sample_index in range(checkpoint_eval.SAMPLES_PER_PROMPT):
            path = tmp_path / f"{target.label}-{sample_index}.png"
            Image.new("RGB", (8, 8), color=(target.epoch, sample_index, 0)).save(path)
            generated.append(
                checkpoint_eval.GeneratedImage(
                    checkpoint_label=target.label,
                    epoch=target.epoch,
                    prompt_index=0,
                    manifest_index=9,
                    sample_index=sample_index,
                    seed=checkpoint_eval._sample_seed(0, sample_index),
                    prompt=prompt.prompt,
                    bucket=prompt.bucket,
                    prompt_style=prompt.prompt_style,
                    path=path,
                    image_sha256="0" * 64,
                ),
            )

    def fake_score_batch(_self, artifacts):
        return [
            {"codex_image_qa": (ord(artifact.metadata["blind_cell"]) - ord("A")) / 10}
            for artifact in artifacts
        ]

    monkeypatch.setattr(
        "vrl.rewards.models.codex_image_qa.CodexImageQARewardModel.score_batch",
        fake_score_batch,
    )
    protocol = SimpleNamespace(
        targets=targets,
        prompts=(prompt,),
        training_reward_components=("codex_image_qa",),
        luna_config={"command": ["unused"], "images_per_call": len(targets)},
    )

    scored, blind_orders = checkpoint_eval._score_curve(protocol, generated)
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    staging_dir = output_dir / "contact_sheets" / ".staging"
    staging_dir.mkdir(parents=True)
    (staging_dir / ".prompt0000.png.dead.tmp").write_bytes(b"interrupted")
    checkpoint_eval._write_contact_sheets(scored, blind_orders, output_dir)
    monkeypatch.setattr(checkpoint_eval, "BOOTSTRAP_RESAMPLES", 50)
    summary = checkpoint_eval._summarize(
        protocol,
        scored,
        checkpoint_eval._dual_seed_diversity(scored),
    )

    assert len(scored) == 10
    assert set(blind_orders[0]) == set(labels)
    for sample_index in range(checkpoint_eval.SAMPLES_PER_PROMPT):
        sample_rows = [row for row in scored if row.image.sample_index == sample_index]
        assert [row.image.checkpoint_label for row in sample_rows] == list(blind_orders[0])
        assert [row.luna_score for row in sample_rows] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert (output_dir / "contact_sheets" / "blind" / "prompt0000.png").is_file()
    assert not staging_dir.exists()
    key = json.loads((output_dir / "blind_key.json").read_text(encoding="utf-8"))
    assert list(key["prompts"][0]["cell_to_arm"].values()) == list(blind_orders[0])
    assert set(summary["paired_vs_base"]) == set(labels[1:])
    assert summary["paired_vs_base"]["checkpoint-final"]["luna_by_bucket"]["hands"]["count"] == 1
    assert set(summary["paired_vs_base"]["checkpoint-final"]) >= {
        "saturation",
        "brightness",
        "edge_energy",
    }
    assessment = summary["endpoint_assessment"]
    assert assessment["luna_was_training_reward"] is True
    assert "both the training reward" in assessment["interpretation_limit"]

    ocr_protocol = SimpleNamespace(**vars(protocol))
    ocr_protocol.training_reward_components = ("ocr",)
    ocr_summary = checkpoint_eval._summarize(
        ocr_protocol,
        scored,
        checkpoint_eval._dual_seed_diversity(scored),
    )
    ocr_assessment = ocr_summary["endpoint_assessment"]
    assert ocr_assessment["luna_was_training_reward"] is False
    assert "not the training reward" in ocr_assessment["interpretation_limit"]


def test_paired_delta_bootstraps_prompt_means_and_counts_ties(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_eval, "BOOTSTRAP_RESAMPLES", 200)

    result = checkpoint_eval._paired_delta(
        {0: 0.4, 1: 0.5, 2: 0.6},
        {0: 0.5, 1: 0.51, 2: 0.7},
        tie_epsilon=0.02,
        bootstrap_seed=7,
    )

    assert result["mean_delta"] == pytest.approx(0.07)
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 0
    assert result["ci95_low"] <= result["mean_delta"] <= result["ci95_high"]


def test_luna_failure_retries_only_scoring_from_verified_generation_stage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, generated, output_dir = _build_generation_stage(tmp_path)
    partial_output = tmp_path / "partial"
    partial_output.mkdir()
    with pytest.raises(FileNotFoundError, match="no complete generation stage"):
        checkpoint_eval._load_generation_stage(protocol, partial_output)

    recovered = checkpoint_eval._load_generation_stage(protocol, output_dir)
    assert [image.image_sha256 for image in recovered] == [
        image.image_sha256 for image in generated
    ]

    monkeypatch.setattr(
        checkpoint_eval,
        "_resolve_protocol",
        lambda _run_dir, **_kwargs: protocol,
    )
    monkeypatch.setattr(
        checkpoint_eval,
        "_generate_curve",
        lambda *_args, **_kwargs: pytest.fail("verified images must not be regenerated"),
    )

    def fail_scoring(_protocol, retry_images):
        assert [image.path for image in retry_images] == [image.path for image in recovered]
        raise RuntimeError("spend limit")

    monkeypatch.setattr(checkpoint_eval, "_score_curve", fail_scoring)
    with pytest.raises(RuntimeError, match="spend limit"):
        checkpoint_eval.main(
            ["--run-dir", str(tmp_path), "--output-dir", str(output_dir)],
        )

    with pytest.raises(ValueError, match="different evaluation protocol"):
        checkpoint_eval._load_generation_stage(
            replace(protocol, config_sha256="f" * 64),
            output_dir,
        )
    Image.new("RGB", (8, 8), color="white").save(generated[0].path)
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint_eval._load_generation_stage(protocol, output_dir)


def test_completed_report_marker_is_integrity_checked_and_never_reopened(tmp_path) -> None:
    protocol, generated, output_dir = _build_generation_stage(tmp_path)
    for relative_path in (
        "samples.jsonl",
        "summary.json",
        "provenance.json",
        "blind_key.json",
        "contact_sheets/manifest.jsonl",
    ):
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    sheet_path = output_dir / "contact_sheets" / "blind" / "prompt0000.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(sheet_path)

    checkpoint_eval._write_completion_marker(protocol, output_dir)

    with pytest.raises(FileExistsError, match="refusing to overwrite completed"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)

    original_bytes = generated[0].path.read_bytes()
    Image.new("RGB", (8, 8), color="white").save(generated[0].path)
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)
    generated[0].path.write_bytes(original_bytes)

    generated[0].path.unlink()
    with pytest.raises(FileNotFoundError, match="generated image is missing"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)
    generated[0].path.write_bytes(original_bytes)

    extra_image = output_dir / "images" / "extra.png"
    Image.new("RGB", (8, 8)).save(extra_image)
    with pytest.raises(ValueError, match="generated image set is not exact"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)
    extra_image.unlink()

    extra_sheet = sheet_path.with_name("extra.png")
    Image.new("RGB", (8, 8)).save(extra_sheet)
    with pytest.raises(ValueError, match="blind contact-sheet set is not exact"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)
    extra_sheet.unlink()

    (output_dir / "summary.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="failed integrity check"):
        checkpoint_eval._reject_completed_output(protocol, output_dir)


def test_contact_sheet_writer_rejects_symlinked_output_parent(tmp_path) -> None:
    _protocol, _generated, output_dir = _build_generation_stage(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_dir / "contact_sheets").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        checkpoint_eval._write_contact_sheets([], {}, output_dir)

    assert list(outside.iterdir()) == []

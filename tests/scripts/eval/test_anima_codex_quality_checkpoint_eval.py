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

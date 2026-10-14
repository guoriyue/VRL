"""MiniMax-H3's two ``from_build`` loaders: what they ask diffusers for.

The modular pipeline is loaded for the ``t2va`` workflow only (the 33B
``transformer_ref`` partition stays on disk), the frozen conditioner follows the
rollout prompt dtype, both VAEs stay fp32, and the video scheduler is swapped
for the flow-convention subclass. The replay model loads the transformer and
the two schedulers from their subfolders and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import build_tiny_minimax_h3_transformer
from vrl.config.precision import RolePrecision
from vrl.models.families.minimax_h3.model import MiniMaxH3Model, MiniMaxH3PipelineModel
from vrl.models.families.minimax_h3.runtime import MiniMaxH3BatchExecutor
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

_PATH = "MiniMaxAI/MiniMax-H3"


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ((), "exactly one device per transformer block"),
        ((0, 1), "exactly one device per transformer block"),
        ((-1,), "explicit nonnegative CUDA indices"),
        ((True,), "explicit nonnegative CUDA indices"),
        (("cuda:0",), "explicit nonnegative CUDA indices"),
    ],
)
def test_partitioned_map_rejects_invalid_block_ownership(devices, message):
    from vrl.models.families.minimax_h3.placement import transformer_device_map

    with pytest.raises(ValueError, match=message):
        transformer_device_map(
            build_tiny_minimax_h3_transformer(), root_device=0, block_devices=devices
        )


def test_partitioned_map_is_complete_and_nonoverlapping():
    from vrl.models.families.minimax_h3.placement import transformer_device_map

    transformer = build_tiny_minimax_h3_transformer()
    mapping = transformer_device_map(transformer, root_device=0, block_devices=(1,))
    assert mapping["proj_out"] == mapping["audio_proj_out"] == mapping["norm_out"] == 0
    for name, _ in list(transformer.named_parameters()) + list(transformer.named_buffers()):
        assert sum(name == key or name.startswith(key + ".") for key in mapping) == 1


def _build(*, rollout: bool, num_steps: int | None = None, device: str = "cuda:0") -> ModelBuild:
    return ModelBuild(
        model_name_or_path=_PATH,
        revision="abc123",
        device=device,
        parameter_dtype=torch.bfloat16,
        family="minimax_h3",
        precision=RolePrecision("bf16", "tf32", outer_autocast=False),
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16) if rollout else None,
        sampling_config={"num_steps": num_steps} if num_steps is not None else None,
    )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("full", "requires LoRA"),
        ("compile", "does not yet support quantization or compile"),
        ("cpu", "explicit CUDA root device"),
        ("implicit_cuda", "explicit CUDA root device"),
    ],
)
def test_partitioned_loader_rejects_unverified_modes_before_loading(monkeypatch, invalid, message):
    from dataclasses import replace

    from diffusers import MiniMaxH3Transformer3DModel

    from vrl.models.families.minimax_h3.placement import load_partitioned_transformer

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid placement must fail before config or weight loading")

    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "load_config", forbidden)
    config = {"use_lora": invalid != "full", "torch_compile": {"enable": invalid == "compile"}}
    device = {"cpu": "cpu", "implicit_cuda": "cuda"}.get(invalid, "cuda:0")
    build = replace(_build(rollout=False, device=device), model_config=config)
    with pytest.raises(ValueError, match=message):
        load_partitioned_transformer(build, (0, 1))


def test_from_build_loads_the_t2va_workflow_and_freezes_the_generation_modules(
    tmp_path,
) -> None:
    """A real modular snapshot, loaded through diffusers by the family's own path."""

    from dataclasses import replace

    from diffusers import MiniMaxH3Transformer3DModel

    from tests.models.steps.denoise.fixtures import write_tiny_minimax_h3_snapshot

    snapshot = write_tiny_minimax_h3_snapshot(tmp_path / "minimax-h3")
    build = replace(
        _build(rollout=True, device="cpu"),
        model_name_or_path=str(snapshot),
        revision=None,
        parameter_dtype=torch.float32,
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16),
    )

    model = MiniMaxH3PipelineModel.from_build(build)

    assert isinstance(model.transformer, MiniMaxH3Transformer3DModel)
    assert model.transformer.dtype == torch.float32
    # The frozen conditioner follows the rollout prompt dtype; both VAEs stay fp32.
    text_encoder = model.pipeline.components["text_encoder"]
    assert {p.dtype for p in text_encoder.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in text_encoder.parameters())
    for vae in (model.pipeline.components["vae"], model.pipeline.components["audio_vae"]):
        assert not any(p.requires_grad for p in vae.parameters())
        assert {p.dtype for p in vae.parameters()} == {torch.float32}
    assert type(model.scheduler).__name__ == "MiniMaxH3FlowScheduler"
    assert model.scheduler.config.shift == 12.0
    assert model.audio_scheduler.config.shift == 3.0
    # The shell exposes every frozen module for the offload discipline.
    assert set(model.pipeline.components) == {"transformer", "vae", "audio_vae", "text_encoder"}


def test_replay_model_loads_transformer_and_both_schedulers_only(monkeypatch) -> None:
    import diffusers

    loads: list[dict[str, Any]] = []

    class _Scheduler(diffusers.MiniMaxH3Scheduler):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            loads.append({"cls": cls.__name__, "path": path, **kwargs})
            return cls(shift=3.0 if kwargs.get("subfolder") == "audio_scheduler" else 12.0)

    def fake_transformer_loader(build: ModelBuild, class_name: str, **kwargs: Any) -> Any:
        loads.append({"cls": class_name, "path": build.model_name_or_path, **kwargs})
        return build_tiny_minimax_h3_transformer()

    monkeypatch.setattr(diffusers, "MiniMaxH3Scheduler", _Scheduler)
    monkeypatch.setattr(
        "vrl.models.families.minimax_h3.model.build_flow_scheduler_class",
        lambda: _Scheduler,
    )
    monkeypatch.setattr("vrl.models.loader.load_diffusers_transformer", fake_transformer_loader)

    # CPU device: from_build eagerly sets both schedules on the build device.
    model = MiniMaxH3Model.from_build(_build(rollout=False, num_steps=4, device="cpu"))

    assert [load["cls"] for load in loads] == [
        "MiniMaxH3Transformer3DModel",
        "_Scheduler",
        "_Scheduler",
    ]
    assert [load.get("subfolder") for load in loads[1:]] == ["scheduler", "audio_scheduler"]
    assert all(load["path"] == _PATH and load["revision"] == "abc123" for load in loads[1:])
    assert model.scheduler.timesteps.numel() == 4
    assert model.audio_scheduler.timesteps.numel() == 4
    assert not hasattr(model, "decode_latents")


def test_batch_executor_pins_one_sample_and_carries_only_the_prompt() -> None:
    """Batch width is not a knob; the encoder is asked for the prompt with the
    request's text length, never a negative prompt."""
    encoded_calls: list[tuple[Any, ...]] = []

    class _Model:
        def encode_prompt(self, prompt: str, negative: Any, **kwargs: Any) -> dict[str, Any]:
            encoded_calls.append((prompt, negative, kwargs))
            return {"prompt_embeds": torch.zeros(1, 2, 3)}

    executor = MiniMaxH3BatchExecutor(_Model())

    assert executor.family == "minimax_h3"
    assert executor.default_num_frames == 124
    assert executor.default_fps == 24
    request = SimpleNamespace(inputs=[SimpleNamespace(prompt="a cat video")])
    params = SimpleNamespace(
        model_request=SimpleNamespace(negative_prompt="blurry"),
        text_encode_kwargs=lambda: {"guidance_scale": 1.0, "max_sequence_length": 64},
    )
    encoded = executor.encode_prompt_for_batch(
        generation_request=request,
        params=params,
        batch=SimpleNamespace(prompt_index=0, sample_count=1),
    )
    assert encoded_calls == [
        ("a cat video", None, {"guidance_scale": 1.0, "max_sequence_length": 64})
    ]
    passthrough = executor.expand_conditioning_to_batch(
        encoded=encoded,
        generation_request=request,
        batch=SimpleNamespace(prompt_index=0, sample_count=1),
    )
    assert passthrough["prompt_embeds"] is encoded["prompt_embeds"]

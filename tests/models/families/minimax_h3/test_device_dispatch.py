"""Opt-in CUDA dispatch gate; random weights do not validate released H3."""

import copy
import os

import pytest
import torch
from accelerate import dispatch_model

from tests.models.steps.denoise.fixtures import (
    TINY_MINIMAX_H3_PATCH_SIZE,
    build_tiny_minimax_h3_components,
    stamp_model_precision,
)
from vrl.config.precision import RolePrecision
from vrl.generation.types import DenoiseRequest
from vrl.models.families.minimax_h3.model import (
    MiniMaxH3PipelineModel,
    MiniMaxH3SamplingState,
    patchify_video_latents,
)
from vrl.models.families.minimax_h3.placement import transformer_device_map
from vrl.models.interfaces.runtime import ModelBuild

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
pytestmark = pytest.mark.gpu


def _model(**vae_stats: float) -> MiniMaxH3PipelineModel:
    model = MiniMaxH3PipelineModel(
        pipeline=build_tiny_minimax_h3_components(**vae_stats), device=torch.device("cpu")
    )
    stamp_model_precision(model)
    return model


def _sampling_state(model: MiniMaxH3PipelineModel) -> MiniMaxH3SamplingState:
    encoded = model.encode_prompt("a cat video", None, guidance_scale=1.0, max_sequence_length=8)
    request = DenoiseRequest(
        width=16,
        height=16,
        frame_count=8,
        num_steps=3,
        guidance_scale=1.0,
        seed=0,
        fps=24,
    )
    return model.prepare_sampling(request, encoded)


@pytest.mark.skipif(
    os.environ.get("VRL_H3_DISPATCH_CUDA") != "1",
    reason="Requires an explicit two-GPU hardware reservation",
)
@pytest.mark.parametrize("mixed", [False, True])
def test_cross_device_blocks_preserve_joint_outputs_and_gradients(mixed):
    from diffusers import MiniMaxH3Transformer3DModel
    from peft import LoraConfig

    assert torch.cuda.device_count() >= 2
    torch.manual_seed(731)
    model = _model()
    state = _sampling_state(model)
    transformer = MiniMaxH3Transformer3DModel.from_config(
        dict(model.transformer.config), num_layers=2
    )
    if mixed:
        transformer.to(dtype=torch.bfloat16)
        for name, module in transformer.named_modules():
            if any(pattern in name for pattern in transformer._keep_in_fp32_modules):
                module.to(dtype=torch.float32)
        transformer.requires_grad_(False)
        transformer.add_adapter(
            LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        )
        for name, parameter in transformer.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.01)
        assert all(p.dtype == torch.float32 for p in transformer.parameters() if p.requires_grad)
    reference = copy.deepcopy(transformer).to("cuda:0")
    # Keep packing projections and selection heads on the root device. Only
    # complete transformer blocks migrate, so functional indexing stays local.
    device_map = transformer_device_map(transformer, root_device=0, block_devices=(0, 1))
    split = dispatch_model(
        transformer,
        device_map=device_map,
        main_device=0,
        force_hooks=True,
    )
    assert {p.device.index for p in split.parameters()} == {0, 1}
    inputs = {
        "hidden_states": patchify_video_latents(state.latents, TINY_MINIMAX_H3_PATCH_SIZE),
        "audio_hidden_states": state.audio_rows,
        "encoder_hidden_states": state.prompt_embeds,
        "timestep": state.layout.row_timestep_plan[0][0],
        "timestep_indices": state.layout.row_timestep_plan[0][1],
        "return_dict": False,
        **state.layout.transformer_kwargs(),
    }
    inputs = {
        k: v.detach().to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    expected = reference(**inputs)
    actual = split(**inputs)
    atol = 1e-3 if mixed else 1e-5
    for left, right in zip(actual, expected, strict=True):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=atol, rtol=atol)
    sum(x.float().square().mean() for x in expected).backward()
    sum(x.float().square().mean() for x in actual).backward()
    checked = 0
    for (name, left), (other, right) in zip(
        split.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            assert torch.isfinite(left.grad).all(), name
            torch.testing.assert_close(left.grad.to("cuda:0"), right.grad, atol=atol, rtol=atol)
            checked += 1
    assert checked > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in split.transformer_blocks[1].parameters()
    )


@pytest.mark.skipif(
    os.environ.get("VRL_H3_DISPATCH_CUDA") != "1",
    reason="Requires an explicit two-GPU hardware reservation",
)
def test_partitioned_replay_loads_local_shards_and_keeps_native_lora_on_owners(tmp_path):
    from diffusers import MiniMaxH3Transformer3DModel

    from vrl.models.families.minimax_h3.model import MiniMaxH3Model, load_h3_replay_components
    from vrl.models.families.registry import get_model_family_entry

    fixture = _model()
    state = _sampling_state(fixture)
    source = MiniMaxH3Transformer3DModel.from_config(
        dict(fixture.transformer.config), num_layers=2
    )
    source.save_pretrained(tmp_path / "transformer", max_shard_size="50KB")
    fixture.scheduler.save_pretrained(tmp_path / "scheduler")
    fixture.audio_scheduler.save_pretrained(tmp_path / "audio_scheduler")
    build = ModelBuild(
        model_name_or_path=str(tmp_path),
        revision=None,
        device="cuda:0",
        parameter_dtype=torch.bfloat16,
        family="minimax_h3",
        precision=RolePrecision("bf16", "tf32", outer_autocast=False),
        sampling_config={"num_steps": 3},
        model_config={
            "use_lora": True,
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            },
        },
    )
    reference = (
        get_model_family_entry("minimax_h3").build_replay(build, expected_identity=None).model
    )
    # The registry has no partitioned replay; build the split model with the same
    # adapter the registry attaches to the reference.
    split = MiniMaxH3Model(**load_h3_replay_components(build, block_devices=(0, 1)))
    split.apply_lora(build)
    for name, parameter in reference.transformer.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    split.transformer.load_state_dict(reference.transformer.state_dict(), strict=True)
    owners = {name: p.device for name, p in split.transformer.named_parameters()}
    assert {device.index for device in owners.values()} == {0, 1}
    assert all(p.dtype == torch.float32 for p in split.transformer.parameters() if p.requires_grad)
    kwargs = {
        "hidden_states": patchify_video_latents(state.latents, TINY_MINIMAX_H3_PATCH_SIZE),
        "audio_hidden_states": state.audio_rows,
        "encoder_hidden_states": state.prompt_embeds,
        "timestep": state.layout.row_timestep_plan[0][0],
        "timestep_indices": state.layout.row_timestep_plan[0][1],
        "return_dict": False,
        **state.layout.transformer_kwargs(),
    }
    kwargs = {
        k: v.detach().to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }
    expected = reference.transformer(**kwargs)
    actual = split.transformer(**kwargs)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-3, rtol=1e-3)
    sum(x.float().square().mean() for x in expected).backward()
    sum(x.float().square().mean() for x in actual).backward()
    reference_parameters = dict(reference.transformer.named_parameters())
    nonzero = 0
    remote_nonzero = 0
    for name, parameter in split.transformer.named_parameters():
        assert parameter.device == owners[name]
        other = reference_parameters[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(
                parameter.grad.to("cuda:0"), other.grad, atol=1e-3, rtol=1e-3
            )
            nonzero += int(parameter.grad.abs().sum() > 0)
            remote_nonzero += int(parameter.device.index == 1 and parameter.grad.abs().sum() > 0)
    assert nonzero > 0
    assert remote_nonzero > 0
    before = {
        name: p.detach().clone()
        for name, p in split.transformer.named_parameters()
        if p.requires_grad
    }
    for model in (reference, split):
        optimizer = torch.optim.AdamW(
            [p for p in model.transformer.parameters() if p.requires_grad], lr=1e-4
        )
        optimizer.step()
    changed_devices = set()
    for name, parameter in split.transformer.named_parameters():
        assert parameter.device == owners[name]
        torch.testing.assert_close(
            parameter.to("cuda:0"), reference_parameters[name], atol=1e-3, rtol=1e-3
        )
        if name in before and not torch.equal(parameter, before[name]):
            changed_devices.add(parameter.device.index)
    assert changed_devices == {0, 1}


@pytest.mark.skipif(
    os.environ.get("VRL_H3_DISPATCH_CUDA") != "1",
    reason="Requires an explicit two-GPU hardware reservation",
)
@pytest.mark.parametrize("vae_device", ["cpu", "cuda:1"])
def test_decode_uses_vae_owner_and_returns_outputs_to_latent_owner(vae_device):
    model = _model(latents_mean=0.5, latents_std=2.0)
    state = _sampling_state(model)
    model.pipeline.vae.to(vae_device)
    latents = state.latents.detach().to("cuda:0")
    saved_video = latents.clone()
    with torch.no_grad():
        video = model.decode_latents(latents)
        expected_video = model.decode_latents(latents.to(vae_device))
    assert video.device == torch.device("cuda:0")
    assert expected_video.device == torch.device(vae_device)
    torch.testing.assert_close(video, expected_video.to("cuda:0"), atol=0, rtol=0)
    torch.testing.assert_close(latents, saved_video, atol=0, rtol=0)
    assert torch.isfinite(video).all()
    assert video.shape == (1, 3, 8, 16, 16)
    assert model.pipeline.vae.device == torch.device(vae_device)
    assert model.pipeline.vae.dtype == torch.float32

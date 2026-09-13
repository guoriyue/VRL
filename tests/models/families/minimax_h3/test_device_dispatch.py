"""Opt-in CUDA dispatch gate; random weights do not validate released H3."""

import copy
import os

import pytest
import torch
from accelerate import dispatch_model

from tests.models.families.minimax_h3.test_backbone_parity import _model, _sampling_state
from tests.models.steps.denoise.fixtures import TINY_MINIMAX_H3_PATCH_SIZE
from vrl.models.families.minimax_h3.model import patchify_video_latents
from vrl.models.families.minimax_h3.placement import transformer_device_map

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")


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
    from dataclasses import replace

    from diffusers import MiniMaxH3Transformer3DModel

    from tests.models.families.minimax_h3.test_model_loading import _build
    from vrl.models.families.minimax_h3.runtime import build_minimax_h3_replay_runtime_bundle

    fixture = _model()
    state = _sampling_state(fixture)
    source = MiniMaxH3Transformer3DModel.from_config(
        dict(fixture.transformer.config), num_layers=2
    )
    source.save_pretrained(tmp_path / "transformer", max_shard_size="50KB")
    fixture.scheduler.save_pretrained(tmp_path / "scheduler")
    fixture.audio_scheduler.save_pretrained(tmp_path / "audio_scheduler")
    build = replace(
        _build(rollout=False, num_steps=3),
        model_name_or_path=str(tmp_path),
        revision=None,
        model_config={
            "use_lora": True,
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            },
        },
    )
    reference = build_minimax_h3_replay_runtime_bundle(build)
    split = build_minimax_h3_replay_runtime_bundle(build, block_devices=(0, 1))
    assert build.defer_trainable_device_move is False
    for name, parameter in reference.model.transformer.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    split.model.transformer.load_state_dict(reference.model.transformer.state_dict(), strict=True)
    owners = {name: p.device for name, p in split.model.transformer.named_parameters()}
    assert {device.index for device in owners.values()} == {0, 1}
    assert all(
        p.dtype == torch.float32 for p in split.model.transformer.parameters() if p.requires_grad
    )
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
    expected = reference.model.transformer(**kwargs)
    actual = split.model.transformer(**kwargs)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-3, rtol=1e-3)
    sum(x.float().square().mean() for x in expected).backward()
    sum(x.float().square().mean() for x in actual).backward()
    reference_parameters = dict(reference.model.transformer.named_parameters())
    nonzero = 0
    remote_nonzero = 0
    for name, parameter in split.model.transformer.named_parameters():
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
        for name, p in split.model.transformer.named_parameters()
        if p.requires_grad
    }
    for bundle in (reference, split):
        optimizer = torch.optim.AdamW(
            [p for p in bundle.model.transformer.parameters() if p.requires_grad], lr=1e-4
        )
        optimizer.step()
    changed_devices = set()
    for name, parameter in split.model.transformer.named_parameters():
        assert parameter.device == owners[name]
        torch.testing.assert_close(
            parameter.to("cuda:0"), reference_parameters[name], atol=1e-3, rtol=1e-3
        )
        if name in before and not torch.equal(parameter, before[name]):
            changed_devices.add(parameter.device.index)
    assert changed_devices == {0, 1}


@pytest.mark.skipif(
    os.environ.get("VRL_H3_FOUR_GPU") != "1",
    reason="Requires an explicit four-GPU hardware reservation",
)
def test_partitioned_conditioner_native_prompt_and_four_device_forward(tmp_path):
    from dataclasses import replace

    from tests.models.families.minimax_h3.test_backbone_parity import _request
    from tests.models.families.minimax_h3.test_model_loading import _build
    from tests.models.steps.denoise.fixtures import stamp_model_precision
    from vrl.models.families.minimax_h3.model import MiniMaxH3Model
    from vrl.models.families.minimax_h3.placement import (
        load_partitioned_text_encoder,
        park_partitioned_text_encoder,
    )

    assert torch.cuda.device_count() >= 4
    fixture = _model()
    components = fixture.pipeline
    components.text_encoder.save_pretrained(tmp_path / "text_encoder", max_shard_size="20KB")
    reference_components = _model().pipeline
    reference_components.transformer.load_state_dict(
        components.transformer.state_dict(), strict=True
    )
    reference_components.text_encoder.load_state_dict(
        components.text_encoder.state_dict(), strict=True
    )
    reference_components.text_encoder.to("cuda:2", dtype=torch.bfloat16).requires_grad_(False)
    reference_components.transformer.to("cuda:0")
    reference = MiniMaxH3Model(pipeline=reference_components, device=torch.device("cuda:0"))
    stamp_model_precision(reference)
    build = replace(_build(rollout=True), model_name_or_path=str(tmp_path), revision=None)
    components.text_encoder = load_partitioned_text_encoder(
        build, root_device=2, layer_devices=(3, 2)
    )
    assert {p.device.index for p in components.text_encoder.parameters()} == {2, 3}
    assert not any(p.requires_grad for p in components.text_encoder.parameters())
    visits = []
    hook = components.text_encoder.model.language_model.layers[0].register_forward_hook(
        lambda module, args, output: visits.append(next(module.parameters()).device.index)
    )
    components.transformer = dispatch_model(
        components.transformer,
        device_map=transformer_device_map(
            components.transformer, root_device=0, block_devices=(1,)
        ),
        main_device=0,
        force_hooks=True,
    )
    model = MiniMaxH3Model(pipeline=components, device=torch.device("cuda:0"))
    stamp_model_precision(model)
    assert {p.device.index for p in model.transformer.parameters()} == {0, 1}
    with torch.no_grad():
        expected = reference.encode_prompt("a wooden block", max_sequence_length=8)
        actual = model.encode_prompt("a wooden block", max_sequence_length=8)
        assert visits == [3]
        assert actual["prompt_embeds"].device == torch.device("cuda:0")
        repeated = model.encode_prompt("a wooden block", max_sequence_length=8)
        assert visits == [3, 3]
        torch.testing.assert_close(
            repeated["prompt_embeds"], actual["prompt_embeds"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["prompt_embeds"], expected["prompt_embeds"], atol=1e-3, rtol=1e-3
        )
        reference_state = reference.prepare_sampling(_request(), expected)
        state = model.prepare_sampling(_request(), actual)
        torch.testing.assert_close(state.latents, reference_state.latents, atol=0, rtol=0)
        expected_output = reference.forward_step(reference_state, 0)
        actual_output = model.forward_step(state, 0)
        assert torch.isfinite(actual_output["noise_pred"]).all()
        torch.testing.assert_close(
            actual_output["noise_pred"], expected_output["noise_pred"], atol=1e-3, rtol=1e-3
        )
        encoder = components.text_encoder
        saved_map = dict(encoder.hf_device_map)
        saved_tensors = {
            name: (tensor.device, tensor.dtype, tensor.detach().cpu().clone())
            for name, tensor in list(encoder.named_parameters()) + list(encoder.named_buffers())
        }
        for inject_error in (False, True):
            try:
                with park_partitioned_text_encoder(encoder):
                    assert encoder.hf_device_map is None
                    assert all(
                        t.device.type == "cpu"
                        for t in list(encoder.parameters()) + list(encoder.buffers())
                    )
                    assert all(not hasattr(module, "_hf_hook") for module in encoder.modules())
                    with (
                        pytest.raises(ValueError, match="GPU-only"),
                        park_partitioned_text_encoder(encoder),
                    ):
                        pytest.fail("Nested parking must be rejected")
                    try:
                        components.vae.to("cuda:2")
                        components.audio_vae.to("cuda:3")
                        video = model.decode_latents(state.latents)
                        waveform, _ = model.decode_audio(state.audio_rows)
                        assert torch.isfinite(video).all() and torch.isfinite(waveform).all()
                        if inject_error:
                            raise RuntimeError("injected decode-stage failure")
                    finally:
                        components.vae.to("cpu")
                        components.audio_vae.to("cpu")
            except RuntimeError as error:
                assert inject_error and str(error) == "injected decode-stage failure"
            else:
                assert not inject_error
            assert encoder.hf_device_map == saved_map
            for name, tensor in list(encoder.named_parameters()) + list(encoder.named_buffers()):
                owner, dtype, value = saved_tensors[name]
                assert tensor.device == owner and tensor.dtype == dtype
                torch.testing.assert_close(tensor.cpu(), value, atol=0, rtol=0)
            restored = model.encode_prompt("a wooden block", max_sequence_length=8)
            torch.testing.assert_close(
                restored["prompt_embeds"], actual["prompt_embeds"], atol=0, rtol=0
            )
    hook.remove()


@pytest.mark.skipif(
    os.environ.get("VRL_H3_DISPATCH_CUDA") != "1",
    reason="Requires an explicit two-GPU hardware reservation",
)
@pytest.mark.parametrize("vae_device", ["cpu", "cuda:1"])
def test_decode_uses_vae_owner_and_returns_outputs_to_latent_owner(vae_device):
    model = _model(latents_mean=0.5, latents_std=2.0)
    state = _sampling_state(model)
    model.pipeline.vae.to(vae_device)
    model.pipeline.audio_vae.to(vae_device)
    latents = state.latents.detach().to("cuda:0")
    audio_rows = state.audio_rows.detach().to("cuda:0")
    saved_video = latents.clone()
    saved_audio = audio_rows.clone()
    with torch.no_grad():
        video = model.decode_latents(latents)
        expected_video = model.decode_latents(latents.to(vae_device))
        waveform, rate = model.decode_audio(audio_rows)
        expected_waveform, expected_rate = model.decode_audio(audio_rows.to(vae_device))
    assert video.device == waveform.device == torch.device("cuda:0")
    assert expected_video.device == expected_waveform.device == torch.device(vae_device)
    torch.testing.assert_close(video, expected_video.to("cuda:0"), atol=0, rtol=0)
    torch.testing.assert_close(waveform, expected_waveform.to("cuda:0"), atol=0, rtol=0)
    torch.testing.assert_close(latents, saved_video, atol=0, rtol=0)
    torch.testing.assert_close(audio_rows, saved_audio, atol=0, rtol=0)
    assert rate == expected_rate == 100
    assert torch.isfinite(video).all() and torch.isfinite(waveform).all()
    assert video.shape == (1, 3, 8, 16, 16)
    assert waveform.shape == (
        2,
        state.layout.num_audio_latents * model.pipeline.audio_vae.hop_length,
    )
    assert model.pipeline.vae.device == model.pipeline.audio_vae.device == torch.device(vae_device)
    assert model.pipeline.vae.dtype == model.pipeline.audio_vae.dtype == torch.float32


@pytest.mark.skipif(
    os.environ.get("VRL_H3_FOUR_GPU") != "1",
    reason="Requires an explicit four-GPU hardware reservation",
)
def test_unified_partitioned_generation_build_and_automatic_decode(tmp_path):
    from dataclasses import replace

    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    from tests.models.families.minimax_h3.test_backbone_parity import _rollout
    from tests.models.families.minimax_h3.test_model_loading import _build
    from vrl.models.families.minimax_h3.partitioned_generation import (
        H3GenerationPlacement,
        build_partitioned_h3_generation_runtime_bundle,
    )

    components = _model().pipeline
    pipeline = MiniMaxH3ModularPipeline(workflow="t2va")
    pipeline.register_components(
        **{
            name: getattr(components, name)
            for name in (
                "vae",
                "audio_vae",
                "tokenizer",
                "processor",
                "scheduler",
                "audio_scheduler",
                "transformer",
                "text_encoder",
            )
        },
    )
    pipeline.update_components(
        scheduler=MiniMaxH3Scheduler.from_config(dict(components.scheduler.config))
    )
    pipeline.save_pretrained(str(tmp_path), max_shard_size="50KB")
    assert (tmp_path / "modular_model_index.json").is_file()
    build = replace(
        _build(rollout=True, num_steps=3),
        generation_memory={"vae_decode": {"tiling": True}},
        model_name_or_path=str(tmp_path),
        revision=None,
        model_config={
            "use_lora": True,
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            },
        },
    )
    placement = H3GenerationPlacement((1,), 2, (3, 2), 2, 3)
    bundle = build_partitioned_h3_generation_runtime_bundle(build, placement)
    model = bundle.model
    model.pipeline.text_encoder_layer = 1
    assert build.defer_trainable_device_move is False
    assert model.pipeline.vae.dtype == model.pipeline.audio_vae.dtype == torch.float32
    assert bundle.loads_full_generation_modules
    assert {p.device.index for p in model.transformer.parameters()} == {0, 1}
    assert {p.device.index for p in model.pipeline.text_encoder.parameters()} == {2, 3}
    with torch.no_grad():
        state, _, predictions = _rollout(model)
        assert len(predictions) == 3 and all(torch.isfinite(p).all() for p in predictions)
        before = model.encode_prompt("a wooden block", max_sequence_length=8)["prompt_embeds"]
        video = model.decode_latents(state.latents)
        waveform, rate = model.decode_audio(model.final_audio_rows(state))
        after = model.encode_prompt("a wooden block", max_sequence_length=8)["prompt_embeds"]
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert video.shape == (1, 3, 8, 16, 16) and waveform.shape[0] == 2 and rate == 100
    assert torch.isfinite(video).all() and torch.isfinite(waveform).all()
    assert model.pipeline.vae.device.type == model.pipeline.audio_vae.device.type == "cpu"
    assert {p.device.index for p in model.pipeline.text_encoder.parameters()} == {2, 3}
    with pytest.raises(RuntimeError, match="whole-component"):
        model.move_frozen_components("cuda:0")
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.generation.types import GenerationRequest
    from vrl.models.families.minimax_h3.runtime import MiniMaxH3BatchExecutor

    request = GenerationRequest(
        request_id="h3-native-partitioned",
        family="minimax_h3",
        task="t2v",
        inputs=["a wooden block"],
        samples_per_prompt=1,
        sampling={
            "num_steps": 3,
            "height": 16,
            "width": 16,
            "num_frames": 8,
            "fps": 24,
            "guidance_scale": 1.0,
            "max_sequence_length": 8,
            "seed": 17,
        },
    )
    executor = MiniMaxH3BatchExecutor(model)
    for name, parameter in model.transformer.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    result = executor.forward_batch(
        request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
    )
    assert result.observations.shape == result.actions.shape
    assert result.log_probs.numel() > 0
    for tensor in (result.observations, result.actions, result.log_probs, result.video):
        assert torch.isfinite(tensor).all()
    assert result.video.shape == (1, 3, 8, 16, 16)
    assert "audio_rows_by_step" in result.replay_tensors
    assert model.pipeline.vae.device.type == model.pipeline.audio_vae.device.type == "cpu"
    assert {p.device.index for p in model.pipeline.text_encoder.parameters()} == {2, 3}
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.minimax_h3.runtime import build_minimax_h3_replay_runtime_bundle

    replay = build_minimax_h3_replay_runtime_bundle(
        replace(build, rollout=None, generation_memory=None), block_devices=(1,)
    )
    replay.model.transformer.load_state_dict(model.transformer.state_dict(), strict=True)
    params = executor.parse_sampling_params(request)
    replay_tensors = {name: tensor.to("cuda:0") for name, tensor in result.replay_tensors.items()}
    replayed_log_probs = []
    logprob_errors = []
    ratio_errors = []
    for index in reversed(range(result.log_probs.shape[1])):
        observation = result.observations[:, index].to("cuda:0")
        restored = replay.model.restore_eval_state(
            replay_tensors, result.context, observation, index
        )
        prediction = replay.model.forward_step(restored, index)["noise_pred"]
        rescored = sde_step_with_logprob(
            restored.scheduler,
            prediction,
            result.timesteps[:, index].to("cuda:0"),
            observation,
            prev_sample=result.actions[:, index].to("cuda:0"),
            noise_level=params.sde.noise_level,
            sde_type=params.sde.sde_type,
            step_index=index,
        ).log_prob
        assert torch.isfinite(rescored).all()
        old = result.log_probs[:, index].to("cuda:0")
        torch.testing.assert_close(rescored, old, atol=1e-3, rtol=0)
        torch.testing.assert_close((rescored - old).exp(), torch.ones_like(old), atol=1e-3, rtol=0)
        replayed_log_probs.append(rescored)
        logprob_errors.append(float((rescored - old).detach().abs().max()))
        ratio_errors.append(float(((rescored - old).exp() - 1).detach().abs().max()))
    (-torch.stack(replayed_log_probs).mean()).backward()
    gradients = [
        p.grad
        for p in replay.model.transformer.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    import json

    (tmp_path / "executor_replay_metrics.json").write_text(
        json.dumps(
            {
                "steps": len(replayed_log_probs),
                "logprob_max_abs": max(logprob_errors),
                "ratio_max_abs_from_one": max(ratio_errors),
                "absolute_tolerance": 1e-3,
                "gradient_tensors": len(gradients),
                "nonzero_gradient_tensors": sum(
                    int(gradient.abs().sum() > 0) for gradient in gradients
                ),
                "scope": "Tiny random checkpoint, native executor and independent partitioned replay; no reward or trainer update.",
            },
            indent=2,
        )
    )

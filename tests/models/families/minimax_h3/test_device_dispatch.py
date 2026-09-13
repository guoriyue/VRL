"""Opt-in CUDA dispatch gate; random weights do not validate released H3."""

import copy
import os

import pytest
import torch
from accelerate import dispatch_model

from tests.models.families.minimax_h3.test_backbone_parity import _model, _sampling_state
from tests.models.steps.denoise.fixtures import TINY_MINIMAX_H3_PATCH_SIZE
from vrl.models.families.minimax_h3.model import patchify_video_latents

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
    device_map = {
        name: 0 for name, _ in transformer.named_children() if name != "transformer_blocks"
    }
    device_map.update({"transformer_blocks.0": 0, "transformer_blocks.1": 1})
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

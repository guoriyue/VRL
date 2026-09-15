"""The fused LoRA-branch swap: same modules, same names, peft's exact numbers.

The rollout pass applies it from ``model.fused_lora_branch``; these tests pin
that the re-classed peft layers keep everything weight sync and adapter
switching rely on (object identity, state_dict names, ``set_adapter`` /
``disable_adapter``), that the no-grad forward is bit-identical to peft's for
every adapter dtype and scaling on the workload list, and that every path with
gradients enabled is peft's own forward.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from torch import nn

from vrl.nn.optimization import ROLLOUT_PASSES, apply_rollout_optimizations
from vrl.nn.optimization.fused_lora_branch import fuse_lora_branches


class _Block(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.ff = nn.Linear(dim, dim * 4)  # not a LoRA target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out(torch.tanh(self.to_q(x))) + self.ff(x)[..., : x.shape[-1]]


class _Policy(nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


def _lora_policy(
    *,
    base_dtype: torch.dtype,
    alpha: int,
    rank: int = 4,
    fp32_adapter: bool,
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    policy = _Policy().to(base_dtype)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_out"],
        init_lora_weights="gaussian",
    )
    wrapped = get_peft_model(policy, config, autocast_adapter_dtype=fp32_adapter)
    # Gaussian init leaves lora_B at zero; give the delta something to add.
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    return wrapped


def _pair(**kwargs) -> tuple[nn.Module, nn.Module]:
    reference = _lora_policy(**kwargs)
    fused = _lora_policy(**kwargs)
    fused.load_state_dict(reference.state_dict())
    assert fuse_lora_branches(fused) == 4, "two LoRA targets per block, two blocks"
    return reference, fused


def test_swap_reclasses_in_place_and_keeps_names() -> None:
    wrapped = _lora_policy(base_dtype=torch.bfloat16, alpha=8, fp32_adapter=True)
    before_params = dict(wrapped.named_parameters())
    before_modules = [module for module in wrapped.modules() if type(module) is LoraLinear]
    assert len(before_modules) == 4

    assert fuse_lora_branches(wrapped) == 4
    assert fuse_lora_branches(wrapped) == 0, "a second pass finds nothing left to swap"

    after_modules = [module for module in wrapped.modules() if isinstance(module, LoraLinear)]
    assert after_modules == before_modules, "the peft objects themselves stay in the tree"
    assert all(type(module) is not LoraLinear for module in after_modules)
    after_params = dict(wrapped.named_parameters())
    assert after_params.keys() == before_params.keys()
    assert all(after_params[name] is before_params[name] for name in before_params)
    assert list(wrapped.state_dict()) == list(before_params)


@pytest.mark.parametrize(
    ("base_dtype", "fp32_adapter", "alpha"),
    [
        (torch.bfloat16, True, 8),  # peft's default upcast, scaling 2 (rank 4)
        (torch.bfloat16, True, 4),  # scaling 1
        (torch.bfloat16, True, 6),  # scaling 1.5: the in-place multiply branch
        (torch.bfloat16, False, 8),  # adapter in the base dtype (wan default)
        (torch.float32, True, 8),  # fp32 base (sd3.5)
    ],
)
def test_no_grad_forward_is_bit_identical_to_peft(
    base_dtype: torch.dtype, fp32_adapter: bool, alpha: int
) -> None:
    reference, fused = _pair(base_dtype=base_dtype, fp32_adapter=fp32_adapter, alpha=alpha)
    x = torch.randn(3, 16, 32).to(base_dtype)
    with torch.no_grad():
        expected = reference(x)
        actual = fused(x)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_grad_enabled_forward_and_backward_are_peft() -> None:
    reference, fused = _pair(base_dtype=torch.bfloat16, fp32_adapter=True, alpha=8)
    x = torch.randn(3, 16, 32).to(torch.bfloat16)

    ref_out = reference(x)
    fused_out = fused(x)
    assert torch.equal(fused_out, ref_out)
    ref_out.float().sum().backward()
    fused_out.float().sum().backward()
    for (name, ref_param), fused_param in zip(
        reference.named_parameters(), fused.parameters(), strict=True
    ):
        if ref_param.grad is None:
            assert fused_param.grad is None, name
        else:
            assert torch.equal(fused_param.grad, ref_param.grad), name


def test_adapter_switching_still_reaches_peft_paths() -> None:
    reference, fused = _pair(base_dtype=torch.bfloat16, fp32_adapter=True, alpha=8)
    x = torch.randn(3, 16, 32).to(torch.bfloat16)
    with torch.no_grad():
        with reference.disable_adapter(), fused.disable_adapter():
            assert torch.equal(fused(x), reference(x))
        # A second active adapter accumulates in fp32 before the final rounding,
        # which is peft's path, so the fused module must defer to it.
        for model in (reference, fused):
            model.add_adapter("previous", model.peft_config["default"])
            model.base_model.set_adapter(["default", "previous"])
        assert torch.equal(fused(x), reference(x))
        for model in (reference, fused):
            model.merge_adapter(adapter_names=["default"])
        assert torch.equal(fused(x), reference(x))


def test_rollout_pass_is_registered_and_gated_by_the_flag() -> None:
    names = [optimization.name for optimization in ROLLOUT_PASSES]
    assert names.index("fused_lora_branch") < names.index("quantization")
    assert names.index("fused_lora_branch") < names.index("compile")

    class _Model:
        def __init__(self, core: nn.Module) -> None:
            self.policy_cores = {"transformer": core}

    class _Build:
        fused_lora_branch = True
        rollout = None
        precision = None
        torch_compile = None
        generation_memory = None

    wrapped = _lora_policy(base_dtype=torch.bfloat16, alpha=8, fp32_adapter=True)
    apply_rollout_optimizations(_Model(wrapped), _Build())
    assert all(
        type(module) is not LoraLinear
        for module in wrapped.modules()
        if isinstance(module, LoraLinear)
    )

    untouched = _lora_policy(base_dtype=torch.bfloat16, alpha=8, fp32_adapter=True)
    _Build.fused_lora_branch = False
    apply_rollout_optimizations(_Model(untouched), _Build())
    assert all(
        type(module) is LoraLinear
        for module in untouched.modules()
        if isinstance(module, LoraLinear)
    )

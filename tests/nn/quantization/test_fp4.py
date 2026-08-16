"""NVFP4 rollout linear math, ownership, targeting, and GPU parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.nn.quantization import (
    Fp4Linear,
    drop_quantized_masters,
)
from vrl.nn.quantization.formats import (
    E2M1_VALUES,
    NVFP4_BLOCK_SIZE,
    NVFP4_K_ALIGNMENT,
    NVFP4_N_ALIGNMENT,
)
from vrl.nn.quantization.fp4 import (
    nvfp4_available,
    quantize_nvfp4,
    to_blocked_scale_layout,
)


def _nvfp4_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        x4, x_scale, _ = quantize_nvfp4(x)
        weight4, weight_scale, _ = quantize_nvfp4(weight)
        torch._scaled_mm(
            x4,
            weight4.t(),
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        return True
    except Exception:
        return False


# This coexists with `@pytest.mark.gpu` on every kernel test below, and must: the
# lane marker only knows whether CUDA exists (tests/conftest.py), while this probe
# knows whether the card can actually run packed-NVFP4 `_scaled_mm`. Folding them
# into the marker would turn a skip into a real failure on pre-Blackwell hardware.
requires_nvfp4 = pytest.mark.skipif(
    not _nvfp4_capable(),
    reason="needs CUDA NVFP4 _scaled_mm",
)


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    codes = torch.empty(
        *raw.shape[:-1],
        raw.shape[-1] * 2,
        dtype=torch.uint8,
        device=raw.device,
    )
    codes[..., 0::2] = raw & 0xF
    codes[..., 1::2] = raw >> 4
    return codes


def _undo_blocked_scale_layout(
    scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    blocks = (
        scales.view(-1, 32, 16)
        .reshape(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(row_blocks, col_blocks, 128, 4)
    )
    padded = blocks.permute(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)
    return padded[:rows, :cols]


def _dequantize_nvfp4_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """Independent test-only decoder for packed values and swizzled scales."""

    rows, packed_k = packed.shape
    k = packed_k * 2
    codes = _unpack_codes(packed)
    magnitudes = torch.tensor(E2M1_VALUES, device=packed.device)[(codes & 0x7).long()]
    values = torch.where((codes & 0x8).bool(), -magnitudes, magnitudes)
    block_scales = _undo_blocked_scale_layout(
        scales,
        rows=rows,
        cols=k // NVFP4_BLOCK_SIZE,
    ).float()
    effective = block_scales * tensor_scale
    return (
        values.view(rows, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE) * effective[..., None]
    ).view(rows, k)


# --- quantization math (CPU) --------------------------------------------------


def test_quantize_dequant_error_is_bounded() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(64, 128)
    packed, scales, tensor_scale = quantize_nvfp4(tensor)
    dequantized = _dequantize_nvfp4_reference(packed, scales, tensor_scale)
    relative_error = float((dequantized - tensor).abs().mean() / tensor.abs().mean())
    assert relative_error < 0.2


def test_exact_grid_values_roundtrip_exactly() -> None:
    grid = torch.tensor(E2M1_VALUES)
    tensor = torch.cat([grid, -grid]).repeat(2)[: 2 * NVFP4_BLOCK_SIZE].view(2, NVFP4_BLOCK_SIZE)
    packed, scales, tensor_scale = quantize_nvfp4(tensor)
    dequantized = _dequantize_nvfp4_reference(packed, scales, tensor_scale)
    torch.testing.assert_close(dequantized, tensor, rtol=0, atol=0)


def test_midpoints_round_to_nearest_even_code() -> None:
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    tensor = torch.cat([midpoints, -midpoints])
    packed, _, _ = quantize_nvfp4(tensor.view(1, NVFP4_BLOCK_SIZE))

    assert _unpack_codes(packed).tolist() == [
        [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
    ]


def test_unaligned_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="K % 16"):
        quantize_nvfp4(torch.randn(4, 24))


def test_blocked_scale_layout_shape_and_origin() -> None:
    scales = torch.arange(6 * 8, dtype=torch.float32).view(6, 8).to(torch.float8_e4m3fn)
    flat = to_blocked_scale_layout(scales)
    assert flat.shape == (128 * 8,)
    assert float(flat.float()[0]) == float(scales.float()[0, 0])


# --- the hardware gate --------------------------------------------------------
#
# `nvfp4_available` decides whether vrl/models/loader.py takes the NVFP4 loading
# path, and every other reference to it in tests/ is a monkeypatch. These two are
# the only places the gate itself is asserted.


def test_nvfp4_gate_rejects_non_cuda_devices() -> None:
    """Packed NVFP4 is a cuBLAS scaled-mm path; no non-cuda device can satisfy it."""

    assert nvfp4_available("cpu") is False
    assert nvfp4_available("meta") is False


@pytest.mark.gpu
def test_nvfp4_gate_matches_the_hardware_it_claims_to_describe() -> None:
    """The >=SM10 capability check must agree with an actual packed-NVFP4 _scaled_mm.

    Deliberately NOT decorated with ``requires_nvfp4``: that skipif *is* the
    empirical probe, so gating on it would skip exactly the interesting case where
    the gate claims support the kernel cannot deliver, leaving a tautology.
    """

    assert nvfp4_available() is _nvfp4_capable()
    # On a machine where the gate says True, this proves the `target.type != "cuda"`
    # branch is what rejects cpu — not an incidental "no CUDA at all" result.
    assert nvfp4_available("cpu") is False


# --- module ownership (CPU) ---------------------------------------------------


def test_nvfp4_is_the_module_scheme_identity() -> None:
    assert Fp4Linear.quantization_scheme == "nvfp4"


def test_master_weight_is_the_only_state_dict_entry() -> None:
    quantized = Fp4Linear(nn.Linear(64, 32, bias=True))
    assert set(quantized.state_dict()) == {"weight", "bias"}


def test_requantizes_on_state_dict_load() -> None:
    quantized = Fp4Linear(nn.Linear(64, 32, bias=False))
    before = quantized.weight_fp4.view(torch.uint8).clone()
    quantized.load_state_dict({"weight": torch.randn(32, 64)})
    assert not torch.equal(before, quantized.weight_fp4.view(torch.uint8))


def test_drop_master_frees_and_blocks_base_load() -> None:
    quantized = Fp4Linear(nn.Linear(64, 32, bias=False))
    assert quantized.drop_master() == 32 * 64 * 4
    assert quantized.weight is None
    assert quantized.drop_master() == 0
    with pytest.raises(RuntimeError, match="master-free"):
        quantized.load_state_dict({"weight": torch.randn(32, 64)})
    quantized.load_state_dict({}, strict=False)


def test_drop_quantized_masters_covers_nvfp4() -> None:
    root = nn.Sequential(Fp4Linear(nn.Linear(64, 32, bias=False)))
    assert drop_quantized_masters(root) > 0
    assert root[0].weight is None


def test_unaligned_in_features_rejected() -> None:
    with pytest.raises(ValueError, match=rf"in_features % {NVFP4_K_ALIGNMENT}"):
        Fp4Linear(nn.Linear(48, 32))


def test_unaligned_out_features_rejected() -> None:
    with pytest.raises(ValueError, match=rf"out_features % {NVFP4_N_ALIGNMENT}"):
        Fp4Linear(nn.Linear(64, 33))


# --- targeting (CPU) ---------------------------------------------------------


class _Blockish(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)
        self.ff_up = nn.Linear(dim, 4 * dim)
        self.small = nn.Linear(dim, 64)
        self.odd_k = nn.Linear(dim + 16, dim)
        self.odd_n = nn.Linear(dim, dim + 8)
        self.proj_out = nn.Linear(dim, dim)


def test_swap_targets_aligned_big_mlp_linears_only() -> None:
    model = _Blockish(1024)
    swapped = Fp4Linear.swap_linears(model)
    assert swapped == ["ff_up"]
    assert isinstance(model.ff_up, Fp4Linear)
    assert isinstance(model.to_q, nn.Linear)
    assert isinstance(model.odd_k, nn.Linear)
    assert isinstance(model.odd_n, nn.Linear)
    assert isinstance(model.x_embedder, nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)


def test_attention_mlp_profile_includes_aligned_attention() -> None:
    model = _Blockish(1024)
    swapped = Fp4Linear.swap_linears(model, target_profile="attention_mlp")
    assert swapped == ["to_q", "ff_up"]
    assert isinstance(model.to_q, Fp4Linear)
    assert isinstance(model.ff_up, Fp4Linear)


def test_invalid_target_profile_raises_before_mutation() -> None:
    model = _Blockish(1024)
    with pytest.raises(ValueError, match="bogus"):
        Fp4Linear.swap_linears(model, target_profile="bogus")
    assert not any(isinstance(module, Fp4Linear) for module in model.modules())


# --- runtime wiring (CPU) -----------------------------------------------------


class _FakeModel:
    """A policy with one NVFP4-eligible MLP linear, driven through the real swap.

    The linear must sit on an MLP path (the NVFP4 profile is ``mlp_only``) and
    clear ``min_features``, or the swap would match nothing and the loader would
    fail loud for the wrong reason.
    """

    quantization_exclude: tuple[str, ...] = ()

    def __init__(self) -> None:
        root = nn.Module()
        root.ff = nn.Module()
        root.ff.net = nn.Linear(1024, 1024, bias=False)
        self._root = root

    @property
    def policy_cores(self) -> dict[str, nn.Module]:
        return {"transformer": self._root}

    @property
    def nvfp4_calls(self) -> int:
        """How many linears actually became NVFP4 — the effect, not the call."""

        return sum(isinstance(module, Fp4Linear) for module in self._root.modules())


def _rollout_spec(
    *,
    recipe: str | None = None,
    device: str | None = None,
    torch_compile: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        torch_compile=torch_compile,
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format="nvfp4", recipe=recipe),
        ),
        rollout=SimpleNamespace(
            base_weight_sync=True,
        ),
    )


def test_apply_rollout_quantization_dispatches_nvfp4(caplog, monkeypatch) -> None:
    from vrl.models.loader import apply_rollout_quantization

    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: True)
    model = _FakeModel()
    with caplog.at_level("INFO"):
        apply_rollout_quantization(model, _rollout_spec(device="cuda:0"))
    assert model.nvfp4_calls == 1
    assert "nvfp4" in caplog.text


def test_nvfp4_policy_rejects_fp8_recipes() -> None:
    with pytest.raises(ValueError, match="does not accept a recipe"):
        QuantizationPolicy(format="nvfp4", recipe="rowwise")


def test_loader_rejects_unsupported_target_before_mutation(monkeypatch) -> None:
    from vrl.models.loader import apply_rollout_quantization

    model = _FakeModel()
    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: False)
    with pytest.raises(RuntimeError, match="NVFP4-capable CUDA target"):
        apply_rollout_quantization(model, _rollout_spec(device="cuda:0"))
    assert model.nvfp4_calls == 0


def test_loader_allows_torch_compile_after_shape_gate(monkeypatch) -> None:
    from vrl.models.loader import apply_rollout_quantization

    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: True)
    model = _FakeModel()
    apply_rollout_quantization(
        model,
        _rollout_spec(torch_compile={"enable": True, "mode": "default"}),
    )
    assert model.nvfp4_calls == 1


def test_resolve_model_build_derives_nvfp4_from_nested_precision() -> None:
    from omegaconf import OmegaConf

    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "path": "x"},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {
                    "dtype": "bf16",
                    "quantization": {"format": "nvfp4"},
                },
            },
        },
    )
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    build = get_model_family_entry("sd3_5").resolve_model_build(
        root,
        "cuda",
        precision=precision,
    )
    assert build.rollout is not None
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == "nvfp4"
    assert build.precision.dtype == "bf16"
    assert build.precision.float32_precision == "tf32"
    assert build.precision.outer_autocast is True
    assert build.parameter_dtype is torch.bfloat16


# --- GPU parity and compile ---------------------------------------------------


@pytest.mark.gpu
@requires_nvfp4
def test_fused_cuda_quantizer_matches_cpu_reference_bytes() -> None:
    torch.manual_seed(0)
    random_values = torch.randn(64, 256, dtype=torch.bfloat16)
    padded_and_partial_chunk = torch.randn(37, 1600, dtype=torch.bfloat16)
    sparse_blocks = torch.zeros(8, 256, dtype=torch.bfloat16)
    sparse_blocks[:, 32:48] = 1.0
    midpoints = torch.tensor(
        [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            -6.0,
        ],
        dtype=torch.bfloat16,
    ).repeat(4, 16)

    for cpu in (random_values, padded_and_partial_chunk, sparse_blocks, midpoints):
        cpu_data, cpu_scales, cpu_tensor_scale = quantize_nvfp4(cpu)
        gpu_data, gpu_scales, gpu_tensor_scale = quantize_nvfp4(cpu.cuda())
        assert torch.equal(cpu_data.view(torch.uint8), gpu_data.cpu().view(torch.uint8))
        assert torch.equal(cpu_scales.view(torch.uint8), gpu_scales.cpu().view(torch.uint8))
        torch.testing.assert_close(gpu_tensor_scale.cpu(), cpu_tensor_scale, rtol=0, atol=0)


@pytest.mark.gpu
@requires_nvfp4
def test_linear_matches_dequantized_reference() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(256, 512, bias=True).cuda().to(torch.bfloat16)
    quantized = Fp4Linear(linear)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        output = quantized(x)
        x4, x_scale, x_tensor_scale = quantize_nvfp4(x)
        weight4, weight_scale, weight_tensor_scale = quantize_nvfp4(linear.weight.data)
        x_dequantized = _dequantize_nvfp4_reference(x4, x_scale, x_tensor_scale)
        weight_dequantized = _dequantize_nvfp4_reference(
            weight4,
            weight_scale,
            weight_tensor_scale,
        )
        reference = (x_dequantized @ weight_dequantized.t() + linear.bias.float()).to(
            torch.bfloat16,
        )
        relative_error = float(
            (output.float() - reference.float()).abs().mean() / reference.float().abs().mean(),
        )
    assert relative_error < 5e-3


@pytest.mark.gpu
@requires_nvfp4
def test_linear_preserves_leading_dims_and_dtype() -> None:
    quantized = Fp4Linear(nn.Linear(128, 256, bias=False).cuda().to(torch.bfloat16))
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.float32)
    output = quantized(x)
    assert output.shape == (2, 8, 256)
    assert output.dtype == torch.float32


@pytest.mark.gpu
@requires_nvfp4
def test_master_free_forward_still_runs() -> None:
    quantized = Fp4Linear(nn.Linear(128, 128, bias=False).cuda().to(torch.bfloat16))
    quantized.drop_master()
    output = quantized(torch.randn(16, 128, device="cuda", dtype=torch.bfloat16))
    assert output.shape == (16, 128)


@pytest.mark.gpu
@requires_nvfp4
def test_cache_survives_cpu_to_cuda_dtype_move() -> None:
    quantized = Fp4Linear(nn.Linear(64, 32, bias=False).to(torch.bfloat16))
    quantized.to("cuda", dtype=torch.bfloat16)
    assert quantized.weight_fp4.dtype is torch.float4_e2m1fn_x2
    assert quantized.weight_scale.dtype is torch.float8_e4m3fn
    assert quantized.weight_tensor_scale.dtype is torch.float32
    assert quantized(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)).shape == (
        128,
        32,
    )


@pytest.mark.gpu
@requires_nvfp4
def test_master_free_cache_moves_without_dtype_cast() -> None:
    quantized = Fp4Linear(nn.Linear(64, 32, bias=False).to(torch.bfloat16))
    quantized.drop_master()
    quantized.to("cuda", dtype=torch.bfloat16)
    assert quantized.weight_fp4.device.type == "cuda"
    assert quantized.weight_fp4.dtype is torch.float4_e2m1fn_x2
    assert quantized.weight_scale.dtype is torch.float8_e4m3fn
    assert quantized.weight_tensor_scale.dtype is torch.float32


@pytest.mark.gpu
@requires_nvfp4
def test_production_shape_torch_compile() -> None:
    quantized = Fp4Linear(nn.Linear(1024, 1024, bias=False).cuda().to(torch.bfloat16))
    compiled = torch.compile(quantized, fullgraph=True)
    output = compiled(torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16))
    assert output.shape == (128, 1024)
    assert output.dtype is torch.bfloat16


@pytest.mark.gpu
@requires_nvfp4
def test_compiled_linear_requantizes_after_weight_sync() -> None:
    torch.manual_seed(0)
    quantized = Fp4Linear(nn.Linear(128, 128, bias=False).cuda().to(torch.bfloat16))
    compiled = torch.compile(quantized, fullgraph=True)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    before_output = compiled(x)
    before = quantized.weight_fp4.view(torch.uint8).clone()

    updated = nn.Linear(128, 128, bias=False).cuda().to(torch.bfloat16)
    quantized.load_state_dict(updated.state_dict())
    reference = Fp4Linear(updated)(x)
    output = compiled(x)

    assert not torch.equal(before, quantized.weight_fp4.view(torch.uint8))
    assert not torch.equal(before_output, output)
    relative_error = float(
        (output.float() - reference.float()).abs().mean()
        / reference.float().abs().mean().clamp_min(1e-12),
    )
    assert relative_error < 5e-3

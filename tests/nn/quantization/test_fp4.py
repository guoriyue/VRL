"""nvfp4 rollout linear: quantization math + swap correctness (CPU), GEMM parity (GPU).

Mirrors test_fp8.py's split: structure/math tests run anywhere; the numeric
GEMM tests need a CUDA device whose ``torch._scaled_mm`` accepts nvfp4 operands
(Blackwell) and are skipped otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vrl.nn.quantization import Fp4Linear, drop_quantized_masters, swap_linears_to_fp4
from vrl.nn.quantization.fp4 import (
    E2M1_VALUES,
    FP4_BLOCK,
    FP4_K_ALIGNMENT,
    FP4_N_ALIGNMENT,
    quantize_nvfp4,
    to_blocked_scale_layout,
)


def _fp4_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        x4, xs, _ = quantize_nvfp4(x)
        w4, ws, _ = quantize_nvfp4(w)
        torch._scaled_mm(x4, w4.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


requires_fp4 = pytest.mark.skipif(not _fp4_capable(), reason="needs CUDA nvfp4 _scaled_mm")


# --- quantization math (CPU) --------------------------------------------------


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
        cols=k // FP4_BLOCK,
    ).float()
    effective = block_scales * tensor_scale
    return (values.view(rows, k // FP4_BLOCK, FP4_BLOCK) * effective[..., None]).view(
        rows,
        k,
    )


def test_quantize_dequant_error_is_bounded():
    torch.manual_seed(0)
    t = torch.randn(64, 128)
    packed, scales, tensor_scale = quantize_nvfp4(t)
    dequant = _dequantize_nvfp4_reference(packed, scales, tensor_scale)
    rel = float((dequant - t).abs().mean() / t.abs().mean())
    assert rel < 0.2  # RTN e2m1 with 1x16 block scales: ~0.13 measured on randn


def test_exact_grid_values_roundtrip_exactly():
    # amax=6 makes tensor_scale*block_scale == 1.0 exactly (448 is e4m3-exact),
    # so on-grid e2m1 values must dequantize bit-exactly.
    grid = torch.tensor(E2M1_VALUES)
    t = torch.cat([grid, -grid]).repeat(2)[: 2 * FP4_BLOCK].view(2, FP4_BLOCK)
    packed, scales, tensor_scale = quantize_nvfp4(t)
    dequant = _dequantize_nvfp4_reference(packed, scales, tensor_scale)
    torch.testing.assert_close(dequant, t, rtol=0, atol=0)


def test_midpoints_round_to_nearest_even_code():
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    t = torch.cat([midpoints, -midpoints])
    packed, _, _ = quantize_nvfp4(t.view(1, FP4_BLOCK))

    assert _unpack_codes(packed).tolist() == [
        [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
    ]


def test_unaligned_k_is_rejected():
    with pytest.raises(ValueError, match="K % 16"):
        quantize_nvfp4(torch.randn(4, 24))


def test_blocked_scale_layout_shape_and_origin():
    scales = torch.arange(6 * 8, dtype=torch.float32).view(6, 8).to(torch.float8_e4m3fn)
    flat = to_blocked_scale_layout(scales)
    # 6 rows pad to 128, 8 cols pad to 8 (2 col-blocks of 4): 128 * 8 total.
    assert flat.shape == (128 * 8,)
    assert float(flat.float()[0]) == float(scales.float()[0, 0])


# --- Fp4Linear ownership contract (CPU) ----------------------------------------


def test_master_weight_is_the_only_state_dict_entry():
    lin = nn.Linear(64, 32, bias=True)
    q = Fp4Linear(lin)
    keys = set(q.state_dict().keys())
    assert keys == {"weight", "bias"}  # fp4 cache + scales are non-persistent


def test_requantizes_on_state_dict_load():
    q = Fp4Linear(nn.Linear(64, 32, bias=False))
    before = q.weight_fp4.view(torch.uint8).clone()
    q.load_state_dict({"weight": torch.randn(32, 64)})
    assert not torch.equal(before, q.weight_fp4.view(torch.uint8))


def test_drop_master_frees_and_blocks_base_load():
    q = Fp4Linear(nn.Linear(64, 32, bias=False))
    assert q.drop_master() == 32 * 64 * 4  # fp32 master: numel x 4 bytes
    assert q.weight is None
    assert q.drop_master() == 0
    with pytest.raises(RuntimeError, match="master-free"):
        q.load_state_dict({"weight": torch.randn(32, 64)})
    # adapter-only sync (no base key) must stay a no-op, not raise
    q.load_state_dict({}, strict=False)


def test_drop_quantized_masters_covers_fp4():
    root = nn.Sequential(Fp4Linear(nn.Linear(64, 32, bias=False)))
    assert drop_quantized_masters(root) > 0
    assert root[0].weight is None


def test_unaligned_in_features_rejected():
    with pytest.raises(ValueError, match=rf"in_features % {FP4_K_ALIGNMENT}"):
        Fp4Linear(nn.Linear(48, 32))


def test_unaligned_out_features_rejected():
    with pytest.raises(ValueError, match=rf"out_features % {FP4_N_ALIGNMENT}"):
        Fp4Linear(nn.Linear(64, 33))


def test_invalid_recipe_rejected():
    with pytest.raises(ValueError, match="nvfp4"):
        Fp4Linear(nn.Linear(64, 32), recipe="rowwise")


# --- swap targeting (CPU) -------------------------------------------------------


class _Blockish(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(dim, dim)  # `embed` -> excluded
        self.to_q = nn.Linear(dim, dim)
        self.ff_up = nn.Linear(dim, 4 * dim)
        self.small = nn.Linear(dim, 64)  # below min_features
        self.odd_k = nn.Linear(dim + 16, dim)  # 1040 % 32 != 0 -> alignment skip
        self.odd_n = nn.Linear(dim, dim + 8)  # 1032 % 16 != 0 -> alignment skip
        self.proj_out = nn.Linear(dim, dim)  # excluded


def test_swap_targets_aligned_big_linears_only():
    model = _Blockish(1024)
    swapped = swap_linears_to_fp4(model)
    assert swapped == ["ff_up"]
    assert isinstance(model.to_q, nn.Linear)
    assert not isinstance(model.to_q, Fp4Linear)
    assert isinstance(model.odd_k, nn.Linear)
    assert isinstance(model.odd_n, nn.Linear)
    assert isinstance(model.x_embedder, nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)


def test_attention_mlp_target_profile_includes_aligned_attention():
    model = _Blockish(1024)

    swapped = swap_linears_to_fp4(model, target_profile="attention_mlp")

    assert swapped == ["to_q", "ff_up"]
    assert isinstance(model.to_q, Fp4Linear)
    assert isinstance(model.ff_up, Fp4Linear)
    assert isinstance(model.odd_k, nn.Linear)
    assert isinstance(model.odd_n, nn.Linear)


def test_invalid_target_profile_raises_before_fp4_mutation():
    model = _Blockish(1024)

    with pytest.raises(ValueError, match="bogus"):
        swap_linears_to_fp4(model, target_profile="bogus")

    assert not any(isinstance(module, Fp4Linear) for module in model.modules())


# --- runtime wiring (CPU) --------------------------------------------------------


class _FakeModel:
    def __init__(self) -> None:
        self.fp4_calls = 0

    def quantize_rollout_fp4(self) -> list[str]:
        self.fp4_calls += 1
        return ["blocks.0.attn.to_q"]


def test_apply_rollout_quantization_dispatches_fp4(caplog):
    from vrl.models.loader import apply_rollout_quantization

    model = _FakeModel()
    with caplog.at_level("INFO"):
        apply_rollout_quantization(
            model,
            SimpleNamespace(
                rollout_quantization="fp4",
                rollout_quantization_recipe=None,
            ),
        )
    assert model.fp4_calls == 1
    assert "fp4 rollout (recipe=nvfp4, profile=mlp_only)" in caplog.text


def test_fp4_rejects_fp8_recipes():
    from vrl.models.loader import apply_rollout_quantization

    with pytest.raises(ValueError, match="not an fp4 recipe"):
        apply_rollout_quantization(
            _FakeModel(),
            SimpleNamespace(rollout_quantization="fp4", rollout_quantization_recipe="rowwise"),
        )


def test_fp4_loader_rejects_unsupported_target_before_mutation(monkeypatch):
    from vrl.models.loader import apply_rollout_quantization

    model = _FakeModel()
    monkeypatch.setattr("vrl.nn.quantization.fp4.nvfp4_available", lambda _device: False)

    with pytest.raises(RuntimeError, match="NVFP4-capable CUDA target"):
        apply_rollout_quantization(
            model,
            SimpleNamespace(
                rollout_quantization="fp4",
                rollout_quantization_recipe=None,
                device="cuda:0",
            ),
        )

    assert model.fp4_calls == 0


def test_fp4_loader_allows_torch_compile_after_production_shape_gate():
    from vrl.models.loader import apply_rollout_quantization

    model = _FakeModel()
    apply_rollout_quantization(
        model,
        SimpleNamespace(
            rollout_quantization="fp4",
            rollout_quantization_recipe=None,
            torch_compile={"enable": True, "mode": "default"},
        ),
    )
    assert model.fp4_calls == 1


def test_extract_runtime_spec_derives_fp4_from_precision_rollout():
    from omegaconf import OmegaConf

    from vrl.models.runtime_config import extract_runtime_spec

    cfg = OmegaConf.create(
        {"model": {"path": "x"}, "precision": {"train": "bf16", "rollout": "fp4"}},
    )
    spec = extract_runtime_spec(cfg, "cuda", torch.bfloat16, task_variant="t2i")
    assert spec.rollout_quantization == "fp4"
    assert spec.dtype is torch.bfloat16  # storage stays the bf16 master


# --- GEMM parity (GPU, Blackwell) -------------------------------------------------


@requires_fp4
def test_fused_cuda_quantizer_matches_cpu_reference_bytes():
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


@requires_fp4
def test_fp4_linear_matches_dequant_reference():
    torch.manual_seed(0)
    lin = nn.Linear(256, 512, bias=True).cuda().to(torch.bfloat16)
    q = Fp4Linear(lin)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = q(x)
        x4, xs, xg = quantize_nvfp4(x)
        w4, ws, wg = quantize_nvfp4(lin.weight.data)
        x_dq = _dequantize_nvfp4_reference(x4, xs, xg)
        w_dq = _dequantize_nvfp4_reference(w4, ws, wg)
        ref = (x_dq @ w_dq.t() + lin.bias.float()).to(torch.bfloat16)
        # The GEMM must reproduce the dequantized reference; tolerance covers bf16
        # accumulation only, not quantization (that is baked into both sides).
        rel = float((out.float() - ref.float()).abs().mean() / ref.float().abs().mean())
    assert rel < 5e-3


@requires_fp4
def test_fp4_linear_preserves_leading_dims_and_dtype():
    q = Fp4Linear(nn.Linear(128, 256, bias=False).cuda().to(torch.bfloat16))
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.float32)
    out = q(x)
    assert out.shape == (2, 8, 256)
    assert out.dtype == torch.float32  # cast back to the input dtype


@requires_fp4
def test_master_free_fp4_forward_still_runs():
    q = Fp4Linear(nn.Linear(128, 128, bias=False).cuda().to(torch.bfloat16))
    q.drop_master()
    out = q(torch.randn(16, 128, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (16, 128)


@requires_fp4
def test_fp4_cache_survives_cpu_to_cuda_dtype_move():
    q = Fp4Linear(nn.Linear(64, 32, bias=False).to(torch.bfloat16))

    q.to("cuda", dtype=torch.bfloat16)

    assert q.weight_fp4.dtype is torch.float4_e2m1fn_x2
    assert q.weight_scale.dtype is torch.float8_e4m3fn
    assert q.weight_tensor_scale.dtype is torch.float32
    assert q(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)).shape == (128, 32)


@requires_fp4
def test_master_free_fp4_cache_moves_without_dtype_cast():
    q = Fp4Linear(nn.Linear(64, 32, bias=False).to(torch.bfloat16))
    q.drop_master()

    q.to("cuda", dtype=torch.bfloat16)

    assert q.weight_fp4.device.type == "cuda"
    assert q.weight_fp4.dtype is torch.float4_e2m1fn_x2
    assert q.weight_scale.dtype is torch.float8_e4m3fn
    assert q.weight_tensor_scale.dtype is torch.float32
    assert q(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)).shape == (128, 32)


@requires_fp4
def test_fp4_production_shape_torch_compile():
    q = Fp4Linear(nn.Linear(1024, 1024, bias=False).cuda().to(torch.bfloat16))
    compiled = torch.compile(q, fullgraph=True)

    out = compiled(torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16))

    assert out.shape == (128, 1024)
    assert out.dtype is torch.bfloat16

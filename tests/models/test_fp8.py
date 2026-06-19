"""fp8 rollout linear: swap-correctness (CPU) + end-to-end drift (GPU).

The structure tests (which Linears get quantized, which stay bf16) run anywhere.
The numeric tests need a CUDA device with fp8 _scaled_mm (Hopper/Blackwell) and
are skipped otherwise. The end-to-end test is the "not much accuracy drift" gate:
a realistic multi-block DiT stack, fp8-swapped, must track its bf16 twin within a
bounded relative error after depth accumulation.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vrl.models.fp8 import Fp8Linear, swap_linears_to_fp8


def _fp8_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones((), device="cuda")
        torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


requires_fp8 = pytest.mark.skipif(not _fp8_capable(), reason="needs CUDA fp8 _scaled_mm")


# --- a realistic DiT block with diffusers-like submodule names ----------------


class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])  # diffusers `to_out.0`

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        q, k, v = (
            t(x).reshape(b, n, h, d // h).transpose(1, 2)
            for t in (self.to_q, self.to_k, self.to_v)
        )
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.to_out[0](a.transpose(1, 2).reshape(b, n, d))


class _FF(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.net:
            x = m(x)
        return x


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attn(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _FF(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class _DiT(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(dim, dim)  # `embed` -> excluded
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.proj_out = nn.Linear(dim, dim)  # `proj_out` -> excluded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x)


# --- structure: which Linears get swapped ------------------------------------


def test_swap_targets_big_attention_and_mlp_only():
    dit = _DiT(dim=1024, heads=8, depth=2)
    swapped = swap_linears_to_fp8(dit, min_features=1024)

    # every block's attention + MLP linears are quantized...
    for i in range(2):
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            assert f"blocks.{i}.attn.{name}" in swapped
        assert f"blocks.{i}.ff.net.0" in swapped
        assert f"blocks.{i}.ff.net.2" in swapped
    # ...and the excluded-by-name embed/head linears are NOT.
    assert "x_embedder" not in swapped
    assert "proj_out" not in swapped
    # the swapped modules are really Fp8Linear now
    assert isinstance(dit.blocks[0].attn.to_q, Fp8Linear)
    assert isinstance(dit.x_embedder, nn.Linear) and not isinstance(dit.x_embedder, Fp8Linear)


def test_min_features_skips_small_linears():
    dit = _DiT(dim=512, heads=8, depth=1)
    swapped = swap_linears_to_fp8(dit, min_features=1024)  # 512 < 1024 → skip all
    assert swapped == []


def test_exclude_substring_is_respected():
    dit = _DiT(dim=1024, heads=8, depth=1)
    swapped = swap_linears_to_fp8(dit, min_features=1024, exclude=("to_q", "norm", "embed", "proj_out"))
    assert not any("to_q" in p for p in swapped)
    assert any("to_k" in p for p in swapped)  # to_k still swapped


def test_invalid_recipe_rejected():
    with pytest.raises(ValueError, match="recipe"):
        Fp8Linear(nn.Linear(16, 16), recipe="bogus")


# --- numeric: per-GEMM and end-to-end drift (GPU) ----------------------------


@requires_fp8
@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise"])
def test_fp8_linear_matches_bf16_within_tolerance(recipe):
    torch.manual_seed(0)
    lin = nn.Linear(2048, 2048).cuda().to(torch.bfloat16)
    fp8 = Fp8Linear(lin, recipe=recipe).cuda()
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
    ref, got = lin(x), fp8(x)
    rel = (got.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.06, f"{recipe} per-GEMM drift {rel:.4f} too high"
    assert got.shape == ref.shape and got.dtype == ref.dtype


@requires_fp8
def test_fp8_linear_preserves_leading_dims_and_bias():
    torch.manual_seed(0)
    lin = nn.Linear(2048, 4096, bias=True).cuda().to(torch.bfloat16)
    fp8 = Fp8Linear(lin).cuda()
    x = torch.randn(2, 64, 2048, device="cuda", dtype=torch.bfloat16)  # [B, N, K]
    out = fp8(x)
    assert out.shape == (2, 64, 4096)


# --- runtime wiring: the swap reaches the rollout builder from precision (CPU) --


class _FakeModel:
    def __init__(self) -> None:
        self.recipes: list[str] = []

    def quantize_transformer_fp8(self, recipe: str = "rowwise") -> list[str]:
        self.recipes.append(recipe)
        return ["blocks.0.attn.to_q", "blocks.0.ff.net.0"]


def test_apply_rollout_fp8_only_fires_for_fp8():
    from vrl.models.loader import apply_rollout_fp8

    model = _FakeModel()
    apply_rollout_fp8(model, SimpleNamespace(rollout_quantization=None))
    assert model.recipes == []  # bf16 rollout: no-op
    apply_rollout_fp8(model, SimpleNamespace(rollout_quantization="fp4"))
    assert model.recipes == []  # fp4 GEMM not built: no-op (online.py gates it)
    apply_rollout_fp8(model, SimpleNamespace(rollout_quantization="fp8"))
    assert model.recipes == ["rowwise"]  # fp8: swap fires


def test_extract_runtime_spec_derives_fp8_from_precision_rollout():
    from omegaconf import OmegaConf

    from vrl.models.runtime_config import extract_runtime_spec

    fp8_cfg = OmegaConf.create({"model": {"path": "x"}, "precision": {"forward": "bf16", "rollout": "fp8"}})
    spec = extract_runtime_spec(fp8_cfg, "cuda", torch.bfloat16, task_variant="t2i")
    assert spec.rollout_quantization == "fp8"
    assert spec.dtype is torch.bfloat16  # storage stays the bf16 master

    bf16_cfg = OmegaConf.create({"model": {"path": "x"}, "precision": "bf16"})
    spec2 = extract_runtime_spec(bf16_cfg, "cuda", torch.bfloat16, task_variant="t2i")
    assert spec2.rollout_quantization is None


@requires_fp8
def test_end_to_end_dit_stack_drift_is_bounded(capsys):
    """The accuracy gate: a 24-block DiT swapped to fp8 tracks its bf16 twin.

    Residual connections keep the bf16 master stream dominant, so the depth-
    accumulated output drift stays well under the per-GEMM ~3.7% blowing up.
    """
    torch.manual_seed(0)
    dim, depth = 1024, 24
    ref_model = _DiT(dim, heads=8, depth=depth).cuda().to(torch.bfloat16).eval()
    fp8_model = copy.deepcopy(ref_model)
    swapped = swap_linears_to_fp8(fp8_model, recipe="rowwise", min_features=1024)
    assert len(swapped) == depth * 6  # 4 attn + 2 mlp per block

    x = torch.randn(2, 256, dim, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref, got = ref_model(x), fp8_model(x)
    drift = float((got.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-9))
    with capsys.disabled():
        print(f"\n[fp8 DiT stack] depth={depth} swapped={len(swapped)} "
              f"end-to-end output drift={drift:.4f}")
    assert drift < 0.12, f"end-to-end fp8 drift {drift:.4f} exceeds bound"

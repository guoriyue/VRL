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

from vrl.nn.quantization import Fp8Linear, swap_linears_to_fp8


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


def _vllm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


requires_vllm_fp8 = pytest.mark.skipif(
    not (_fp8_capable() and _vllm_available()),
    reason="needs CUDA fp8 + vLLM block kernel",
)


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


def test_mlp_only_target_profile_excludes_attention():
    dit = _DiT(dim=1024, heads=8, depth=1)

    swapped = swap_linears_to_fp8(dit, target_profile="mlp_only")

    assert swapped == ["blocks.0.ff.net.0", "blocks.0.ff.net.2"]
    assert isinstance(dit.blocks[0].attn.to_q, nn.Linear)
    assert not isinstance(dit.blocks[0].attn.to_q, Fp8Linear)


def test_invalid_target_profile_raises_before_fp8_mutation():
    dit = _DiT(dim=1024, heads=8, depth=1)

    with pytest.raises(ValueError, match="bogus"):
        swap_linears_to_fp8(dit, target_profile="bogus")

    assert not any(isinstance(module, Fp8Linear) for module in dit.modules())


def test_min_features_skips_small_linears():
    dit = _DiT(dim=512, heads=8, depth=1)
    swapped = swap_linears_to_fp8(dit, min_features=1024)  # 512 < 1024 → skip all
    assert swapped == []


def test_exclude_substring_is_respected():
    dit = _DiT(dim=1024, heads=8, depth=1)
    swapped = swap_linears_to_fp8(
        dit, min_features=1024, exclude=("to_q", "norm", "embed", "proj_out")
    )
    assert not any("to_q" in p for p in swapped)
    assert any("to_k" in p for p in swapped)  # to_k still swapped


def test_swap_excludes_qwen_modulation_and_text_input():
    # qwen-image names AdaLN modulation `img_mod`/`txt_mod` and the text-
    # conditioning input `txt_in`; none contain "norm", so without explicit
    # excludes the most precision-sensitive GEMMs (per-block scale/shift/gate)
    # would be silently fp8-quantized. Pin that the exclude list catches the qwen
    # naming while attention/MLP still quantize.
    class _QwenBlock(nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.img_mod = nn.ModuleList([nn.SiLU(), nn.Linear(dim, 6 * dim)])
            self.txt_mod = nn.ModuleList([nn.SiLU(), nn.Linear(dim, 6 * dim)])
            self.attn = _Attn(dim, heads)
            self.img_mlp = _FF(dim)

    class _QwenDiT(nn.Module):
        def __init__(self, dim: int, depth: int) -> None:
            super().__init__()
            self.img_in = nn.Linear(64, dim)  # small in_features -> size-filtered
            self.txt_in = nn.Linear(dim, dim)  # big -> would quantize w/o exclude
            self.transformer_blocks = nn.ModuleList([_QwenBlock(dim, 8) for _ in range(depth)])
            self.proj_out = nn.Linear(dim, dim)

    dit = _QwenDiT(dim=1024, depth=2)
    swapped = swap_linears_to_fp8(dit, min_features=1024)

    # modulation + text-conditioning input stay bf16
    assert not any("_mod" in p for p in swapped), swapped
    assert "txt_in" not in swapped
    assert isinstance(dit.txt_in, nn.Linear) and not isinstance(dit.txt_in, Fp8Linear)
    assert not isinstance(dit.transformer_blocks[0].img_mod[1], Fp8Linear)
    # attention + MLP are still quantized (the real speed levers)
    assert any("attn.to_q" in p for p in swapped)
    assert any("img_mlp" in p for p in swapped)


def test_invalid_recipe_rejected():
    with pytest.raises(ValueError, match="recipe"):
        Fp8Linear(nn.Linear(16, 16), recipe="bogus")


# --- numeric: per-GEMM and end-to-end drift (GPU) ----------------------------


@requires_vllm_fp8
def test_blockwise_recipe_matches_bf16_via_vllm():
    """blockwise reuses vLLM's triton block kernel (not hand-rolled); 128-aligned."""
    torch.manual_seed(0)
    lin = nn.Linear(2048, 2048).cuda().to(torch.bfloat16)
    fp8 = Fp8Linear(lin, recipe="blockwise").cuda()
    assert fp8.recipe == "blockwise"
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    ref, got = lin(x), fp8(x)
    rel = (got.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.06, f"blockwise drift {rel:.4f} too high"


def test_blockwise_falls_back_to_rowwise_on_unaligned_dims():
    """A non-128-aligned linear can't use the block kernel → silently uses rowwise."""
    fp8 = Fp8Linear(nn.Linear(2000, 2048), recipe="blockwise")  # 2000 % 128 != 0
    assert fp8.recipe == "rowwise"


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
def test_fp8_linear_exposes_master_weight_and_requantizes_on_sync():
    """Full-finetune weight-sync loads the bf16 `weight` key; the fp8 cache must
    re-derive from it (regression: cosmos full fine-tune sync failed because the
    swapped module had weight_fp8, not weight)."""
    torch.manual_seed(0)
    fp8 = Fp8Linear(nn.Linear(2048, 2048).cuda().to(torch.bfloat16)).cuda()
    sd = fp8.state_dict()
    assert "weight" in sd  # the trainable key the sync expects
    assert "weight_fp8" not in sd and "weight_scale" not in sd  # cache is derived
    # simulate a weight sync: load a fresh weight, output must track it
    new = nn.Linear(2048, 2048).cuda().to(torch.bfloat16)
    fp8.load_state_dict(new.state_dict())
    x = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    ref = new(x)
    rel = (fp8(x).float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.06, f"fp8 output should track the synced weight (drift {rel:.4f})"


def test_master_free_fp8_survives_adapter_only_weight_sync():
    """LoRA sync recurses through frozen fp8 children without base weights.

    A master-free rollout intentionally drops the bf16 ``weight``. PyTorch still
    calls every child's ``_load_from_state_dict`` for an adapter-only payload, so
    the fp8 cache must stay intact instead of trying to requantize a missing master.
    """
    from vrl.models.utils import load_weights_into

    class _AdapterPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = nn.Linear(16, 16, bias=False).requires_grad_(False)
            self.base_layer = Fp8Linear(base)
            self.lora = nn.Parameter(torch.zeros(2, 16))

    policy = _AdapterPolicy()
    policy.base_layer.drop_master()
    cached_weight = policy.base_layer.weight_fp8.clone()
    update = torch.ones_like(policy.lora)

    load_weights_into(
        policy,
        {"transformer.lora": update},
        prefix="transformer",
        label="adapter policy",
    )

    assert policy.base_layer.weight is None
    assert torch.equal(policy.base_layer.weight_fp8, cached_weight)
    assert torch.equal(policy.lora, update)


def test_drop_fp8_masters_compatibility_facade():
    from vrl.nn.quantization import drop_fp8_masters

    fp8 = Fp8Linear(nn.Linear(16, 16, bias=False))

    assert drop_fp8_masters(fp8) == 16 * 16 * 4
    assert fp8.weight is None


@requires_fp8
def test_fp8_cache_survives_cpu_to_cuda_dtype_move():
    """A model-level dtype move must rebuild, not cast, the derived fp8 cache."""

    fp8 = Fp8Linear(nn.Linear(128, 128, bias=False).to(torch.bfloat16))

    fp8.to("cuda", dtype=torch.bfloat16)

    assert fp8.weight_fp8.dtype is torch.float8_e4m3fn
    assert fp8.weight_scale.dtype is torch.float32
    out = fp8(torch.randn(16, 128, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (16, 128)


@requires_fp8
def test_master_free_fp8_cache_moves_without_dtype_cast():
    fp8 = Fp8Linear(nn.Linear(128, 128, bias=False).to(torch.bfloat16))
    fp8.drop_master()

    fp8.to("cuda", dtype=torch.bfloat16)

    assert fp8.weight_fp8.device.type == "cuda"
    assert fp8.weight_fp8.dtype is torch.float8_e4m3fn
    assert fp8.weight_scale.dtype is torch.float32
    out = fp8(torch.randn(16, 128, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (16, 128)


@requires_fp8
def test_fp8_linear_handles_fp32_activations():
    """DiT adaLN / pooled-projection paths feed fp32 activations, and rowwise
    _scaled_mm rejects an fp32 output. Fp8Linear must accumulate in bf16 and
    return the input dtype (regression: a live SD3.5 rollout crashed here)."""
    torch.manual_seed(0)
    lin = nn.Linear(2048, 2048).cuda().to(torch.bfloat16)
    fp8 = Fp8Linear(lin, recipe="rowwise").cuda()
    x = torch.randn(512, 2048, device="cuda", dtype=torch.float32)
    out = fp8(x)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@requires_fp8
def test_fp8_linear_preserves_leading_dims_and_bias():
    torch.manual_seed(0)
    lin = nn.Linear(2048, 4096, bias=True).cuda().to(torch.bfloat16)
    fp8 = Fp8Linear(lin).cuda()
    x = torch.randn(2, 64, 2048, device="cuda", dtype=torch.bfloat16)  # [B, N, K]
    out = fp8(x)
    assert out.shape == (2, 64, 4096)


# --- fp8 rollout guards: no silent bf16 when precision.rollout=fp8 (CPU) ---------


class _SwapModel:
    def __init__(self, swapped: list[str]) -> None:
        self._swapped = swapped
        self.recipe_seen: str | None = None

    def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
        self.recipe_seen = recipe
        return self._swapped


def test_apply_rollout_quantization_raises_when_swap_matches_nothing():
    from vrl.models.loader import apply_rollout_quantization

    with pytest.raises(RuntimeError, match="matched 0 linears"):
        apply_rollout_quantization(_SwapModel([]), SimpleNamespace(rollout_quantization="fp8"))


def test_apply_rollout_quantization_noop_and_count_when_not_fp8_or_swapped():
    from vrl.models.loader import apply_rollout_quantization

    assert (
        apply_rollout_quantization(_SwapModel([]), SimpleNamespace(rollout_quantization=None)) == 0
    )
    assert (
        apply_rollout_quantization(
            _SwapModel(["a", "b"]), SimpleNamespace(rollout_quantization="fp8")
        )
        == 2
    )


def test_apply_rollout_quantization_rejects_blockwise_with_compile():
    """blockwise graph-breaks inductor (compiled ~10x slower than eager) — refuse."""
    from vrl.models.loader import apply_rollout_quantization

    with pytest.raises(ValueError, match=r"blockwise.*torch_compile"):
        apply_rollout_quantization(
            _SwapModel(["a"]),
            SimpleNamespace(
                rollout_quantization="fp8",
                rollout_quantization_recipe="blockwise",
                torch_compile={"enable": True, "mode": "default"},
            ),
        )
    # blockwise without compile, and rowwise with compile, both stay allowed.
    model = _SwapModel(["a"])
    apply_rollout_quantization(
        model,
        SimpleNamespace(
            rollout_quantization="fp8",
            rollout_quantization_recipe="blockwise",
            torch_compile=None,
        ),
    )
    assert model.recipe_seen == "blockwise"
    model = _SwapModel(["a"])
    apply_rollout_quantization(
        model,
        SimpleNamespace(
            rollout_quantization="fp8",
            rollout_quantization_recipe="rowwise",
            torch_compile={"enable": True, "mode": "default"},
        ),
    )
    assert model.recipe_seen == "rowwise"


def test_apply_rollout_quantization_passes_recipe_through():
    """precision.rollout_recipe reaches the fp8 swap; absent → rowwise default."""
    from vrl.models.loader import apply_rollout_quantization

    model = _SwapModel(["a"])
    apply_rollout_quantization(
        model,
        SimpleNamespace(rollout_quantization="fp8", rollout_quantization_recipe="blockwise"),
    )
    assert model.recipe_seen == "blockwise"

    model = _SwapModel(["a"])
    apply_rollout_quantization(model, SimpleNamespace(rollout_quantization="fp8"))
    assert model.recipe_seen == "rowwise"


@pytest.mark.parametrize("scheme", ["fp8", "fp4"])
def test_backstop_raises_for_any_requested_scheme_when_unquantized(scheme):
    """Any requested rollout quantization with no matching module fails loud."""
    from vrl.models.loader import assert_rollout_quantization_applied

    model = SimpleNamespace(
        transformer=nn.Sequential(nn.Linear(16, 16))
    )  # plain, no QuantizedLinear
    with pytest.raises(RuntimeError, match=rf"0 {scheme} quantized linear"):
        assert_rollout_quantization_applied(
            model, SimpleNamespace(rollout_quantization=scheme, task_variant="t2i")
        )


def test_backstop_rejects_a_different_quantization_scheme():
    """An FP4 request must not pass merely because an FP8 module exists."""
    from vrl.models.loader import assert_rollout_quantization_applied

    model = SimpleNamespace(transformer=nn.Sequential(Fp8Linear(nn.Linear(16, 16))))

    with pytest.raises(RuntimeError, match="0 fp4 quantized linear"):
        assert_rollout_quantization_applied(
            model,
            SimpleNamespace(rollout_quantization="fp4", task_variant="t2i"),
        )


def test_backstop_ok_when_quantized_module_present_incl_compiled():
    from vrl.models.loader import assert_rollout_quantization_applied
    from vrl.nn.quantization import QuantizedLinear

    assert isinstance(
        Fp8Linear(nn.Linear(16, 16)), QuantizedLinear
    )  # scheme subclasses the marker
    real = nn.Sequential(Fp8Linear(nn.Linear(16, 16)))  # CPU construct is fine
    assert_rollout_quantization_applied(
        SimpleNamespace(transformer=real), SimpleNamespace(rollout_quantization="fp8")
    )
    # torch.compile wrapper exposes _orig_mod — the guard must unwrap it
    compiled = SimpleNamespace(_orig_mod=real)
    assert_rollout_quantization_applied(
        SimpleNamespace(transformer=compiled), SimpleNamespace(rollout_quantization="fp8")
    )


def test_backstop_noop_when_no_quantization_requested():
    from vrl.models.loader import assert_rollout_quantization_applied

    assert_rollout_quantization_applied(
        SimpleNamespace(transformer=None), SimpleNamespace(rollout_quantization=None)
    )


# --- runtime wiring: the swap reaches the rollout builder from precision (CPU) --


class _FakeModel:
    def __init__(self) -> None:
        self.recipes: list[str] = []

    def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
        self.recipes.append(recipe)
        return ["blocks.0.attn.to_q", "blocks.0.ff.net.0"]


def test_apply_rollout_quantization_dispatches_by_scheme():
    from vrl.models.loader import apply_rollout_quantization

    model = _FakeModel()
    # bf16/fp16/fp32 rollout: a load-time dtype, nothing to swap
    apply_rollout_quantization(model, SimpleNamespace(rollout_quantization=None))
    assert model.recipes == []
    # an unimplemented scheme must NOT silently no-op (that was the old footgun) — raise
    with pytest.raises(NotImplementedError, match="no rollout swap"):
        apply_rollout_quantization(model, SimpleNamespace(rollout_quantization="int8"))
    assert model.recipes == []
    # fp8: the swap fires
    apply_rollout_quantization(model, SimpleNamespace(rollout_quantization="fp8"))
    assert model.recipes == ["rowwise"]


def test_extract_runtime_spec_derives_fp8_from_precision_rollout():
    from omegaconf import OmegaConf

    from vrl.models.runtime_config import extract_runtime_spec

    fp8_cfg = OmegaConf.create(
        {"model": {"path": "x"}, "precision": {"forward": "bf16", "rollout": "fp8"}}
    )
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
    drift = float(
        (got.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-9)
    )
    with capsys.disabled():
        print(
            f"\n[fp8 DiT stack] depth={depth} swapped={len(swapped)} "
            f"end-to-end output drift={drift:.4f}"
        )
    assert drift < 0.12, f"end-to-end fp8 drift {drift:.4f} exceeds bound"

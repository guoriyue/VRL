from __future__ import annotations

import json
from types import SimpleNamespace

from omegaconf import OmegaConf

from tests.tiny_real_stack import TinySanaPipeline, tiny_sana_online_config
from vrl.scripts.generation import full_sequence_probe


def test_probe_replays_a_recipe_rollout_at_the_trainer_batch(monkeypatch, tmp_path) -> None:
    # The SANA online recipe on tiny real modules: fp32, two steps, generation
    # batch 2 replayed as batch 1, so replay runs a different batch shape than
    # the rollout through the production loop, executor hooks and SDE math.
    cfg = tiny_sana_online_config(tmp_path, overrides=("actor.training_microbatch_size=1",))
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    config = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, config)

    full_sequence_probe.main(["--config", str(config), "--artifact-dir", str(tmp_path / "probe")])

    report = json.loads((tmp_path / "probe" / "result.json").read_text())
    assert report["passed"]
    assert (report["rollout_batch"], report["replay_batch"]) == (2, 1)
    assert [step["step"] for step in report["steps"]] == [0, 1]
    assert report["max_logprob_abs"] <= report["max_abs_logprob_diff_gate"]
    assert report["module_hooks"] and "first_divergent_module" in report["steps"][0]
    assert (tmp_path / "probe" / "image.png").is_file()


def test_probe_without_module_hooks_compares_per_step_only(monkeypatch, tmp_path) -> None:
    cfg = tiny_sana_online_config(tmp_path, overrides=("actor.training_microbatch_size=1",))
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    config = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, config)

    full_sequence_probe.main(
        ["--config", str(config), "--artifact-dir", str(tmp_path / "probe"), "--no-module-hooks"]
    )

    report = json.loads((tmp_path / "probe" / "result.json").read_text())
    assert report["passed"] and not report["module_hooks"]
    assert "first_divergent_module" not in report["steps"][0]
    assert all("logprob_max_abs" in step for step in report["steps"])


def test_probe_block_audit_matches_float64_on_an_fp32_policy(monkeypatch, tmp_path) -> None:
    # An fp32 policy differs from its float64 reference only by fp32 rounding,
    # so the audit must see near-identical outputs and gradients; SANA's
    # cross-attention exercises the reference SDPA. Two layers make the blocks
    # a repeated list.
    from tests.models.steps.denoise import fixtures

    build = fixtures.build_tiny_transformer
    monkeypatch.setattr(
        fixtures,
        "build_tiny_transformer",
        lambda name, **overrides: build(name, **{"num_layers": 2, **overrides}),
    )
    cfg = tiny_sana_online_config(tmp_path, overrides=("actor.training_microbatch_size=1",))
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    config = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, config)

    full_sequence_probe.main(
        ["--config", str(config), "--artifact-dir", str(tmp_path / "probe"), "--block-audit"]
    )

    report = json.loads((tmp_path / "probe" / "result.json").read_text())
    audit = report["block_audit"]
    assert report["passed"] and audit["attention_calls"] > 0 and audit["attention_kernel_passed"]
    assert max(audit["forward_rel_l2"]) < 1e-5
    # The gradient sums signed terms, so its fp32 relative error sits above eps.
    assert audit["grad_rel_l2"] < 1e-3 and audit["grad_cosine"] > 1 - 1e-8
    # fp32 rounding sits far below a bf16-sized perturbation of the inputs.
    assert max(audit["forward_over_floor"]) < 0.1 and audit["grad_over_floor"] < 0.1
    assert all(step["density_fp64_max_abs"] < 1e-5 for step in report["steps"])
    residuals = [step["residual"]["passed"] for step in report["steps"] if "residual" in step]
    assert residuals and all(residuals)


def test_residual_check_rejects_a_sampler_off_its_density() -> None:
    import torch

    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(2, 4, 64, 64, generator=generator, dtype=torch.float64)
    std_dev_t, sqrt_neg_dt = torch.tensor(0.4, dtype=torch.float64), torch.tensor(0.3)
    reference = SimpleNamespace(
        std_dev_t=std_dev_t, sqrt_neg_dt=sqrt_neg_dt, prev_sample_mean=mean
    )
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)

    drawn = full_sequence_probe._residual_stats(
        reference, mean + std_dev_t * sqrt_neg_dt * noise, sde_type="flow_grpo"
    )
    too_wide = full_sequence_probe._residual_stats(
        reference, mean + 1.1 * std_dev_t * sqrt_neg_dt * noise, sde_type="flow_grpo"
    )
    shifted = full_sequence_probe._residual_stats(
        reference, mean + std_dev_t * sqrt_neg_dt * (noise + 0.1), sde_type="flow_grpo"
    )

    assert drawn["passed"] and not too_wide["passed"] and not shifted["passed"]


def test_reference_attention_is_sdpa_in_float64() -> None:
    import torch
    import torch.nn.functional as F

    native = F.scaled_dot_product_attention
    generator = torch.Generator().manual_seed(0)
    query, key, value = (
        torch.randn(2, 3, 37, 8, generator=generator, requires_grad=True) for _ in range(3)
    )
    keep = torch.rand(2, 1, 1, 37, generator=generator) > 0.3
    keep[..., 0] = True
    additive = torch.randn(2, 1, 37, 37, generator=generator)

    for mask in (None, keep, additive):
        # Native SDPA mis-adds an fp32 additive mask to float64 scores.
        double_mask = mask if mask is None or mask.dtype == torch.bool else mask.double()
        expected = F.scaled_dot_product_attention(
            *(t.double() for t in (query, key, value)), attn_mask=double_mask
        )
        with full_sequence_probe._reference_attention(query_chunk=8) as calls:
            got = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        assert len(calls) == 1 and got.dtype == torch.float32
        torch.testing.assert_close(got.double(), expected, rtol=1e-6, atol=1e-6)
        (grad_got,) = torch.autograd.grad(got.sum(), [query])
        (grad_expected,) = torch.autograd.grad(expected.sum(), [query])
        torch.testing.assert_close(grad_got, grad_expected, rtol=1e-5, atol=1e-6)
    assert F.scaled_dot_product_attention is native


def test_attention_audit_fails_a_kernel_that_computes_something_else(monkeypatch) -> None:
    import torch
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(0)
    query, key, value = (torch.randn(1, 2, 40, 8, generator=generator) for _ in range(3))
    with full_sequence_probe._audited_attention(query_chunk=16) as errors:
        F.scaled_dot_product_attention(query, key, value)
    assert errors[0] < 1e-6

    # A kernel that drops the softmax scale: plausible output, wrong function.
    native = F.scaled_dot_product_attention
    monkeypatch.setattr(
        F, "scaled_dot_product_attention", lambda q, k, v, **kw: native(q, k, v, scale=1.0)
    )
    with full_sequence_probe._audited_attention(query_chunk=16) as errors:
        F.scaled_dot_product_attention(query, key, value)
    assert errors[0] > full_sequence_probe.ATTENTION_KERNEL_MAX_REL_L2

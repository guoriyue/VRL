"""The frame-shared AdaLN swap: same modules, same names, per-frame conditioning.

Both roles apply it from ``model.frame_shared_adaln``; these tests pin what the
swap must preserve for weight sync (state_dict names, parameter identity), for
the trainer (gradients through the re-classed norms), and for the replay ratio
(numerics against the per-token diffusers path on a real Cosmos DiT with the
5-D per-frame timestep VRL's runner issues), plus the fallbacks: a 1-D timestep
and a copied model keep the reference math.
"""

from __future__ import annotations

import copy

import torch
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAdaLayerNorm,
    CosmosAdaLayerNormZero,
)

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS_TEXT_DIM,
    build_tiny_transformer,
)
from vrl.nn.optimization import ROLLOUT_PASSES, apply_rollout_optimizations
from vrl.nn.optimization.frame_shared_adaln import share_adaln_across_frames

_BATCH, _FRAMES = 2, 3


def _per_frame_kwargs(seed: int = 1) -> dict:
    """A forward with one timestep per (sample, frame), the Predict2.5 contract."""

    torch.manual_seed(seed)
    return dict(
        hidden_states=torch.randn(_BATCH, 5, _FRAMES, 4, 4),
        timestep=torch.rand(_BATCH, 1, _FRAMES, 1, 1),
        encoder_hidden_states=torch.randn(_BATCH, 3, TINY_COSMOS_TEXT_DIM),
        padding_mask=torch.zeros(1, 1, 4, 4),
        return_dict=False,
    )


def test_swap_keeps_parameter_identity_and_state_dict_names() -> None:
    transformer = build_tiny_transformer("cosmos")
    before = {name: param for name, param in transformer.named_parameters()}

    count = share_adaln_across_frames(transformer)

    assert count == 4, "norm1/norm2/norm3 of the one block plus norm_out"
    block = transformer.transformer_blocks[0]
    assert isinstance(block.norm1, CosmosAdaLayerNormZero), "re-classed, still the diffusers type"
    assert isinstance(transformer.norm_out, CosmosAdaLayerNorm)
    assert block.norm1.frame_layout is transformer.frame_layout
    after = {name: param for name, param in transformer.named_parameters()}
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before), "weight sync targets moved"
    assert list(transformer.state_dict()) == list(before)


def test_per_frame_timestep_forward_and_backward_match_the_per_token_path() -> None:
    reference = build_tiny_transformer("cosmos")
    shared = build_tiny_transformer("cosmos")
    assert share_adaln_across_frames(shared) == 4
    kwargs = _per_frame_kwargs()

    ref_x = kwargs["hidden_states"].clone().requires_grad_(True)
    shared_x = kwargs["hidden_states"].clone().requires_grad_(True)
    ref_out = reference(**{**kwargs, "hidden_states": ref_x})[0]
    shared_out = shared(**{**kwargs, "hidden_states": shared_x})[0]

    # Post-patch (H/2) * (W/2) tokens per frame, T frames of them.
    assert shared.frame_layout.tokens_per_frame == 4
    assert shared.frame_layout.seq_len == _FRAMES * 4
    torch.testing.assert_close(shared_out, ref_out, rtol=1e-5, atol=1e-6)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    shared_out.backward(grad)
    torch.testing.assert_close(shared_x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)
    for (name, ref_param), shared_param in zip(
        reference.named_parameters(), shared.parameters(), strict=True
    ):
        # Reduced over every token: fp32 accumulation-order noise scales with the
        # gradient's magnitude, so the absolute tolerance does too.
        torch.testing.assert_close(
            shared_param.grad,
            ref_param.grad,
            rtol=1e-5,
            atol=1e-6 * ref_param.grad.abs().max().item(),
            msg=name,
        )


def test_one_timestep_per_sample_takes_the_reference_path() -> None:
    reference = build_tiny_transformer("cosmos")
    shared = build_tiny_transformer("cosmos")
    share_adaln_across_frames(shared)
    kwargs = {**_per_frame_kwargs(), "timestep": torch.full((_BATCH,), 0.75)}

    with torch.no_grad():
        shared_out = shared(**kwargs)[0]
        ref_out = reference(**kwargs)[0]

    assert shared.frame_layout.seq_len is None, "a 1-D timestep records no frame layout"
    torch.testing.assert_close(shared_out, ref_out)


def test_copied_transformer_keeps_its_own_layout() -> None:
    """A previous-policy copy must not read the layout of the model it was copied from."""

    shared = build_tiny_transformer("cosmos")
    share_adaln_across_frames(shared)
    kwargs = _per_frame_kwargs()
    with torch.no_grad():
        expected = shared(**kwargs)[0]

    copied = copy.deepcopy(shared)

    assert copied.frame_layout is not shared.frame_layout
    assert copied.transformer_blocks[0].norm1.frame_layout is copied.frame_layout
    with torch.no_grad():
        torch.testing.assert_close(copied(**kwargs)[0], expected)


def test_compiled_forward_takes_the_shared_path_in_one_graph() -> None:
    """The root pre-hook and the layout it writes trace into the production compiled root."""

    import torch._dynamo as dynamo

    shared = build_tiny_transformer("cosmos")
    share_adaln_across_frames(shared)
    kwargs = _per_frame_kwargs()
    with torch.no_grad():
        expected = shared(**kwargs)[0]

    dynamo.reset()
    explanation = dynamo.explain(shared)(**kwargs)
    assert explanation.graph_break_count == 0
    compiled = torch.compile(shared, backend="aot_eager", fullgraph=False)
    with torch.no_grad():
        torch.testing.assert_close(compiled(**kwargs)[0], expected)


def test_runner_per_frame_timestep_takes_the_shared_path() -> None:
    """The Predict2.5 runner's ``[B, 1, T, 1, 1]`` timestep is what the hook keys on."""

    from types import SimpleNamespace

    from tests.models.steps.denoise.fixtures import stamp_model_precision
    from vrl.models.families.cosmos.predict2_5.model import (
        CosmosPredict25Model,
        CosmosPredict25SamplingState,
    )

    def state() -> CosmosPredict25SamplingState:
        torch.manual_seed(3)
        # First frame conditioned (clamped to the ground-truth velocity at the
        # conditioning timestep), the rest sampled at the step's sigma.
        cond_mask = torch.zeros(_BATCH, 1, _FRAMES, 4, 4)
        cond_mask[:, :, 0] = 1.0
        cond_indicator = torch.zeros(_BATCH, 1, _FRAMES, 1, 1)
        cond_indicator[:, :, 0] = 1.0
        return CosmosPredict25SamplingState(
            latents=torch.randn(_BATCH, 4, _FRAMES, 4, 4),
            timesteps=torch.tensor([0.0]),
            scheduler=SimpleNamespace(sigmas=torch.tensor([0.75])),
            prompt_embeds=torch.randn(_BATCH, 3, TINY_COSMOS_TEXT_DIM),
            negative_prompt_embeds=torch.randn(_BATCH, 3, TINY_COSMOS_TEXT_DIM),
            guidance_scale=2.0,
            do_cfg=True,
            cond_latent=torch.randn(_BATCH, 4, _FRAMES, 4, 4),
            cond_mask=cond_mask,
            cond_indicator=cond_indicator,
            padding_mask=torch.zeros(1, 1, 4, 4),
            height=4,
            width=4,
            num_frames=_FRAMES,
            fps=16,
        )

    def noise_pred(transformer: torch.nn.Module) -> torch.Tensor:
        model = CosmosPredict25Model(
            pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
            device=torch.device("cpu"),
        )
        stamp_model_precision(model)
        return model.forward_step(state(), 0)["noise_pred"]

    shared = build_tiny_transformer("cosmos")
    share_adaln_across_frames(shared)

    shared_pred = noise_pred(shared)

    assert shared.frame_layout.seq_len == _FRAMES * 4, "the runner's forward recorded a layout"
    torch.testing.assert_close(
        shared_pred, noise_pred(build_tiny_transformer("cosmos")), rtol=1e-5, atol=1e-5
    )


def test_rollout_pass_reaches_every_core_only_when_enabled() -> None:
    from types import SimpleNamespace

    from vrl.config.precision import RolePrecision

    names = [optimization.name for optimization in ROLLOUT_PASSES]
    assert names.index("frame_shared_adaln") < names.index("quantization")
    assert names.index("frame_shared_adaln") < names.index("compile")

    def build(flag: bool) -> SimpleNamespace:
        return SimpleNamespace(
            device="cpu",
            family="test",
            torch_compile=None,
            frame_shared_adaln=flag,
            precision=RolePrecision("bf16", "tf32", None),
            rollout=SimpleNamespace(base_weight_sync=True),
        )

    def hooked_roots(model: SimpleNamespace) -> int:
        return sum(hasattr(core, "frame_layout") for core in model.policy_cores.values())

    model = SimpleNamespace(
        policy_cores={
            "transformer": build_tiny_transformer("cosmos"),
            "transformer_2": build_tiny_transformer("cosmos"),
        },
        quantization_exclude=(),
    )
    apply_rollout_optimizations(model, build(False))
    assert hooked_roots(model) == 0

    apply_rollout_optimizations(model, build(True))
    assert hooked_roots(model) == 2, "both experts must condition through one path"

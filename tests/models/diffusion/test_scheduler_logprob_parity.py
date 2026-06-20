"""Per-family scheduler log-prob parity (sample -> replay, ratio == 1).

The predict2 parity bug lived in one family's REAL scheduler sigma table (EDM
domain, sigma_max=80), not in the shared SDE math. Each case here drives
``sde_step_with_logprob`` with the family's actual scheduler: pipeline-owned
schedulers load their checkpoint config from the local HF cache (skipped when
the cache is absent, e.g. clean CI); anima constructs its scheduler explicitly
in code, mirroring vrl/models/diffusion/cosmos/anima/runtime.py.

Coverage split (what runs where):
- The EDM->flow CONVERSION BRANCH itself (vrl/math/diffusion/flow_matching.py)
  is already covered cache-free in EVERY CI run by
  tests/math/test_diffusion_flow_matching.py, which feeds a synthetic EDM-domain
  table (sigma_max > 1) through ``sde_step_with_logprob`` and pins the same
  ratio==1 / std_dev / prev_sample_mean invariants. So clean CI never loses the
  conversion-math coverage even though the cache-loaded families below skip.
- The REAL per-family sigma TABLES (the actual predict2 EDM sigma_max=80 table,
  etc.) only execute when their HF checkpoint config is cached. On a clean CI
  runner with no cache those families skip, so the real tables currently run
  only under the gated e2e suite (tests/.../test_real_checkpoint_rl.py). The
  anima case is the lone family that runs cache-free here (in-code scheduler).

TODO(option-a): publish/pin a tiny scheduler-only HF repo per family
(scheduler_config.json is a few KB, not weights) and load it via ``from_config``
at a pinned revision instead of ``local_files_only`` cache lookup, so the real
per-family sigma tables run on every CI runner without the gated e2e suite.
Keep the ratio==1 invariant + ``sigma.max() > 1`` magnitude pin; do NOT assert
literal sigma values (config-as-declaration).

Pinned invariant per family x sde_type x step: replaying the recorded
prev_sample under unchanged inputs reproduces the collection log-prob exactly,
at O(1) magnitude.

AR families (janus/nextstep) are out of scope: token-logit log-probs, no SDE
replay path.
"""

from __future__ import annotations

import pytest
import torch

from vrl.math.diffusion.flow_matching import sde_step_with_logprob

_NUM_STEPS = 10
_BATCH = 2


def _hf_scheduler(repo: str, revision: str | None = None):
    diffusers = pytest.importorskip("diffusers")

    try:
        # load_config is class-agnostic for schedulers (they all read
        # scheduler_config.json); dispatch the real class via _class_name.
        config = diffusers.FlowMatchEulerDiscreteScheduler.load_config(
            repo,
            subfolder="scheduler",
            revision=revision,
            local_files_only=True,
        )
    except Exception as exc:  # cache miss / offline CI
        # Silent no-op on clean CI: the cache-loaded families skip here. The
        # conversion-math itself stays covered cache-free by
        # tests/math/test_diffusion_flow_matching.py; see module docstring +
        # the TODO(option-a) tiny-scheduler-repo plan to run the real tables.
        pytest.skip(f"scheduler config for {repo} not in local HF cache: {exc}")
    scheduler_cls = getattr(diffusers, config["_class_name"])
    return scheduler_cls.from_config(config)


def _anima_scheduler():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    return scheduler


_FAMILY_SCHEDULERS = {
    "sd3_5": lambda: _hf_scheduler("stabilityai/stable-diffusion-3.5-medium"),
    "wan_2_1": lambda: _hf_scheduler("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    "cosmos-predict2": lambda: _hf_scheduler("nvidia/Cosmos-Predict2-2B-Video2World"),
    "cosmos-predict2.5": lambda: _hf_scheduler(
        "nvidia/Cosmos-Predict2.5-2B",
        revision="diffusers/base/post-trained",
    ),
    "cosmos-anima": _anima_scheduler,
}


def _timestep(scheduler, step_index: int) -> torch.Tensor:
    return torch.full((_BATCH,), float(scheduler.timesteps[step_index]))


@pytest.mark.parametrize("sde_type", ["flow_grpo", "cps"])
@pytest.mark.parametrize("family", sorted(_FAMILY_SCHEDULERS))
def test_family_scheduler_sample_replay_parity(family: str, sde_type: str) -> None:
    """Checks the GRPO ratio==1 invariant on each family's real sigma table."""
    scheduler = _FAMILY_SCHEDULERS[family]()
    scheduler.set_timesteps(_NUM_STEPS)
    assert scheduler.sigmas is not None

    # First, middle, and last-but-one step; the final row's sigma_prev is 0 and
    # carries no stochastic step to replay.
    for step_index in (0, _NUM_STEPS // 2, _NUM_STEPS - 2):
        gen = torch.Generator().manual_seed(100 + step_index)
        sample = torch.randn(_BATCH, 3, 4, generator=gen)
        model_output = torch.randn(_BATCH, 3, 4, generator=gen)
        timestep = _timestep(scheduler, step_index)

        sampled = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            generator=torch.Generator().manual_seed(7),
            noise_level=1.0,
            sde_type=sde_type,
            step_index=step_index,
        )
        replayed = sde_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            prev_sample=sampled.prev_sample,
            noise_level=1.0,
            sde_type=sde_type,
            step_index=step_index,
        )

        torch.testing.assert_close(
            replayed.log_prob,
            sampled.log_prob,
            msg=f"{family} step {step_index}: replay log-prob != sample log-prob",
        )
        # Magnitude pin: EDM-domain sigmas fed raw into the flow formulas put
        # log-probs at ~ -sigma^2 (-4635 observed on predict2).
        assert sampled.log_prob.abs().max().item() < 50.0, (
            f"{family} step {step_index}: log-prob magnitude "
            f"{sampled.log_prob.abs().max().item():.1f} is not O(1)"
        )


def test_predict2_scheduler_exercises_edm_conversion() -> None:
    """Checks the predict2 case actually covers the EDM->flow conversion path.

    If a checkpoint update ever normalizes predict2's sigma table to [0, 1],
    this test documents that the EDM branch lost its only real-family coverage
    (rather than silently passing through the legacy path).
    """
    scheduler = _FAMILY_SCHEDULERS["cosmos-predict2"]()
    scheduler.set_timesteps(_NUM_STEPS)
    assert float(scheduler.sigmas.max()) > 1.0

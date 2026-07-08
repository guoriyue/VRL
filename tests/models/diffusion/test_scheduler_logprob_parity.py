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
from omegaconf import OmegaConf

from vrl.config.loading import CONFIGS_ROOT, load_config
from vrl.math.diffusion.flow_matching import sde_step_with_logprob
from vrl.rollouts.families import FAMILY_REGISTRY, normalize_rollout_family

_NUM_STEPS = 10
_BATCH = 2


def _load_hf_scheduler(repo: str, revision: str | None = None):
    """Build a real scheduler from its cached ``scheduler_config.json``, or return
    None on a cache miss (clean/offline CI) so the caller can try the next config
    or skip. The conversion-math itself stays covered cache-free by
    tests/math/test_diffusion_flow_matching.py; see the module docstring."""
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
    except Exception:  # cache miss / offline CI
        return None
    scheduler_cls = getattr(diffusers, config["_class_name"])
    return scheduler_cls.from_config(config)


def _anima_scheduler():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    return scheduler


def _diffusion_families() -> list[str]:
    return [f for f, e in FAMILY_REGISTRY.items() if e.collector.kind == "diffusion"]


def _model_configs_by_family() -> dict[str, list[str]]:
    """Discover ``family -> [model config, ...]`` by reading the declared
    ``model.family`` from every diffusion model config.

    The config files are the single source of truth, so a newly added family
    config is picked up automatically — there is no hand-maintained family->path
    table to drift. Declared aliases (e.g. ``family: wan``) normalize to their
    canonical registry key; configs sort so the smaller, more likely-cached
    checkpoint is tried first.
    """
    out: dict[str, list[str]] = {}
    for path in sorted((CONFIGS_ROOT / "model" / "diffusion").rglob("*.yaml")):
        declared = OmegaConf.load(path).get("model", {}).get("family")
        if declared is None:
            continue
        family = normalize_rollout_family(str(declared))
        rel = path.relative_to(CONFIGS_ROOT).with_suffix("").as_posix()
        out.setdefault(family, []).append(rel)
    return out


_MODEL_CONFIGS_BY_FAMILY = _model_configs_by_family()

# anima is the lone diffusion family whose scheduler is built in code
# (vrl/.../cosmos/anima/runtime.py), not downloaded — mirror that construction.
# Keyed off a registry family so a rename/typo fails the guard below instead of
# silently dropping anima's cache-free coverage.
def _mochi_scheduler():
    # Mochi ships invert_sigmas=true (ascending time); the FAMILY standardizes
    # onto the descending domain before any SDE math runs — mirror that, since
    # the raw checkpoint table is never what the rollout uses
    # (vrl/models/diffusion/mochi/model.py:standard_mochi_scheduler).
    shipped = _load_hf_scheduler("genmo/mochi-1-preview")
    if shipped is None:
        return None
    from vrl.models.diffusion.mochi.model import standard_mochi_scheduler

    scheduler = standard_mochi_scheduler(shipped.config, _NUM_STEPS, "cpu")
    scheduler._vrl_timesteps_ready = True
    return scheduler


_MANUAL_SCHEDULERS = {
    "cosmos-predict2-anima": _anima_scheduler,
    "mochi": _mochi_scheduler,
}
assert set(_MANUAL_SCHEDULERS) <= set(_diffusion_families())

# alphas_cumprod-ladder families: their rollout/replay log-probs run through
# sde_type="ddim", never the flow SDE — the parity invariant is checked there.
_DDIM_FAMILIES = {"cogvideox", "pixart_sigma"}
assert _DDIM_FAMILIES <= set(_diffusion_families())


def _scheduler_for(family: str):
    if family in _MANUAL_SCHEDULERS:
        scheduler = _MANUAL_SCHEDULERS[family]()
        if scheduler is None:
            pytest.skip(f"no cached scheduler config for {family!r}")
        return scheduler
    # Try the family's configs in order; the first with a cached scheduler wins.
    # path/revision come from the YAML, never re-typed here.
    for cfg_name in _MODEL_CONFIGS_BY_FAMILY.get(family, ()):
        model = load_config(cfg_name).model
        scheduler = _load_hf_scheduler(model.path, revision=model.get("revision"))
        if scheduler is not None:
            return scheduler
    pytest.skip(f"no cached scheduler for diffusion family {family!r}")


def _set_timesteps(scheduler) -> None:
    # Flow-match schedulers with dynamic shifting (flux, qwen_image) require a
    # resolution-derived ``mu`` at set_timesteps; the sample<->replay parity
    # invariant is independent of the shift, so pin mu=0 (identity shift) for a
    # deterministic, valid sigma table. Fixed-schedule schedulers ignore mu.
    if getattr(scheduler, "_vrl_timesteps_ready", False):
        return  # family construction already set its own ladder (mochi)
    if getattr(scheduler.config, "use_dynamic_shifting", False):
        scheduler.set_timesteps(_NUM_STEPS, mu=0.0)
    else:
        scheduler.set_timesteps(_NUM_STEPS)


def _timestep(scheduler, step_index: int) -> torch.Tensor:
    return torch.full((_BATCH,), float(scheduler.timesteps[step_index]))


@pytest.mark.parametrize("sde_type", ["flow_grpo", "cps"])
@pytest.mark.parametrize("family", sorted(_diffusion_families()))
def test_family_scheduler_sample_replay_parity(family: str, sde_type: str) -> None:
    """Checks the GRPO ratio==1 invariant on each family's real sigma table."""
    if family in _DDIM_FAMILIES:
        if sde_type == "cps":
            pytest.skip("ddim family: one parity row is enough")
        sde_type = "ddim"
    scheduler = _scheduler_for(family)
    _set_timesteps(scheduler)

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
    scheduler = _scheduler_for("cosmos-predict2")
    _set_timesteps(scheduler)
    assert float(scheduler.sigmas.max()) > 1.0

"""Per-family scheduler log-prob parity (sample -> replay, ratio == 1).

The predict2 parity bug lived in one family's real scheduler sigma table (EDM
domain, sigma_max=80), not in the shared SDE math. Each case drives
``sde_step_with_logprob`` with the family's scheduler declaration. Checkpoint-
owned declarations are committed as small fixtures with their Hugging Face
source and immutable revision; code-owned schedulers mirror the production
constructor. The suite therefore exercises every family offline and never
depends on a developer's cache.

The fixture is a long-term correctness asset, not a synthetic schedule: update
it only by copying the named ``scheduler_config.json`` at the pinned source
revision. Literal sigma values remain intentionally unpinned because the
configuration is the declaration under test.

Pinned invariant per family x sde_type x step: replaying the recorded
prev_sample under unchanged inputs reproduces the collection log-prob exactly,
at O(1) magnitude.

Token-AR families (janus/nextstep) are out of scope: token-logit log-probs, no
SDE replay path. Chunk-AR denoise families are also out of scope: they own
grouped temporal transitions rather than the generic full-sequence scheduler
and SDE evaluator exercised here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from vrl.config.loading import list_bundled_configs, load_config
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.names import normalize_model_family
from vrl.models.families.registry import FAMILY_REGISTRY

_NUM_STEPS = 10
_BATCH = 2
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scheduler_configs.json"


def _anima_scheduler():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.register_to_config(sigma_data=1.0, sigma_max=1.0)
    return scheduler


def _echo_scheduler():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)


def _full_sequence_diffusion_families() -> list[str]:
    """Return families backed by the generic full-sequence SDE evaluator."""

    return [
        family
        for family, entry in FAMILY_REGISTRY.items()
        if entry.policy_semantics.generation_regime == "full_sequence"
    ]


def _model_repos_by_family() -> dict[str, set[str]]:
    """Derive denoise-family repositories from family-first model presets."""

    out: dict[str, set[str]] = {}
    denoise_families = set(_full_sequence_diffusion_families())
    for name in list_bundled_configs("model"):
        model = load_config(name).get("model", {})
        declared = model.get("family")
        repo_id = model.get("path")
        if declared is None or repo_id is None:
            continue
        family = normalize_model_family(str(declared))
        # Family-first config layout deliberately mixes token and denoise model
        # presets under ``model/``. Scheduler provenance belongs only to the
        # denoise-step families covered by this test.
        if family not in denoise_families:
            continue
        out.setdefault(family, set()).add(str(repo_id))
    return out


_SCHEDULER_FIXTURES: dict[str, dict[str, object]] = json.loads(
    _FIXTURE_PATH.read_text(),
)

_MANUAL_SCHEDULERS = {
    "cosmos-predict2-anima": _anima_scheduler,
    "echo": _echo_scheduler,
}
assert set(_full_sequence_diffusion_families()) >= set(_MANUAL_SCHEDULERS)

# alphas_cumprod-ladder families: their rollout/replay log-probs run through
# sde_type="ddim", never the flow SDE — the parity invariant is checked there.
_DDIM_FAMILIES = {"cogvideox", "pixart_sigma"}
assert set(_full_sequence_diffusion_families()) >= _DDIM_FAMILIES


def _scheduler_for(family: str):
    if family in _MANUAL_SCHEDULERS:
        return _MANUAL_SCHEDULERS[family]()

    diffusers = pytest.importorskip("diffusers")
    config = dict(_SCHEDULER_FIXTURES[family]["config"])
    scheduler_cls = getattr(diffusers, config.pop("_class_name"))
    scheduler = scheduler_cls.from_config(config)

    if family == "mochi":
        # Mochi ships invert_sigmas=true (ascending time), while rollout and
        # replay standardize onto the descending domain before SDE math.
        from vrl.models.families.mochi.model import standard_mochi_scheduler

        scheduler = standard_mochi_scheduler(scheduler.config, _NUM_STEPS, "cpu")
        scheduler._vrl_timesteps_ready = True
    elif family == "pixart_sigma":
        # The shipped DPM solver is deterministic; production rebuilds a DDIM
        # scheduler on the same trained beta ladder for RL log-probs.
        from vrl.models.families.pixart_sigma.model import pixart_ddim_scheduler

        scheduler = pixart_ddim_scheduler(scheduler.config, _NUM_STEPS, "cpu")
        scheduler._vrl_timesteps_ready = True
    elif family in {"hunyuan_image", "hunyuan_video"}:
        # These pipelines explicitly pass this linear sigma declaration rather
        # than relying on the scheduler's default timestep construction.
        sigmas = torch.linspace(1.0, 0.0, _NUM_STEPS + 1)[:-1].numpy()
        scheduler.set_timesteps(sigmas=sigmas)
        scheduler._vrl_timesteps_ready = True
    return scheduler


def _parity_cases() -> list[object]:
    cases = []
    for family in sorted(_full_sequence_diffusion_families()):
        sde_types = ("ddim",) if family in _DDIM_FAMILIES else ("flow_grpo", "cps")
        cases.extend(
            pytest.param(family, sde_type, id=f"{family}-{sde_type}") for sde_type in sde_types
        )
    return cases


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


def test_scheduler_fixtures_have_complete_pinned_provenance() -> None:
    """Every checkpoint-owned scheduler is pinned and tied to a model YAML."""
    checkpoint_owned = set(_full_sequence_diffusion_families()) - set(
        _MANUAL_SCHEDULERS,
    )
    assert set(_SCHEDULER_FIXTURES) == checkpoint_owned

    model_repos = _model_repos_by_family()
    for family, fixture in _SCHEDULER_FIXTURES.items():
        assert fixture["model_repo_id"] in model_repos[family]
        assert re.fullmatch(r"[^/]+/[^/]+", str(fixture["scheduler_repo_id"]))
        assert re.fullmatch(r"[0-9a-f]{40}", str(fixture["revision"]))
        assert fixture["scheduler_config_path"] == "scheduler/scheduler_config.json"
        assert isinstance(fixture["config"], dict)
        assert "_class_name" in fixture["config"]
        if fixture["scheduler_repo_id"] != fixture["model_repo_id"]:
            assert isinstance(fixture.get("note"), str)


@pytest.mark.parametrize(("family", "sde_type"), _parity_cases())
def test_family_scheduler_sample_replay_parity(family: str, sde_type: str) -> None:
    """Checks the GRPO ratio==1 invariant on each family's real sigma table."""
    scheduler = _scheduler_for(family)
    _set_timesteps(scheduler)

    # First, middle, and last-but-one step; the final row's sigma_prev is 0 and
    # carries no stochastic step to replay.
    for step_index in (0, _NUM_STEPS // 2, _NUM_STEPS - 2):
        gen = torch.Generator().manual_seed(100 + step_index)
        sample = torch.randn(_BATCH, 3, 4, generator=gen)
        model_output = torch.randn(_BATCH, 3, 4, generator=gen)
        timestep = torch.full((_BATCH,), float(scheduler.timesteps[step_index]))

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

"""Shared sampling-config projection for the eval and generation scripts.

Every value comes from the parsed ``sampling`` / ``rollout`` / ``model.executor``
sections. This module owns no defaults: a key a script needs must be declared
by the config (or supplied on the CLI), because an evaluation that silently ran
with a hyper-parameter the training config never set would be measuring the
wrong thing. The family keys fall back to ``model.executor`` exactly as the
training runtime does (``GenericDenoiseBatchExecutor`` applies its
``default_*`` values when a request carries none). The dict is intentionally untyped: it is a per-request runtime payload
handed straight to the generation request, not a launch-time config object.

Note: ``sana_aesthetic_report.resolve_sampling`` is deliberately NOT routed here
— it returns the frozen ``SANA_EVAL_SAMPLING_CONFIG`` to keep reproducibility
independent of the training SDE, and must stay separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

# Keys every denoise family declares; the projection always carries them.
_REQUIRED_SAMPLING_KEYS: tuple[tuple[str, type], ...] = (
    ("width", int),
    ("height", int),
    ("num_steps", int),
    ("guidance_scale", float),
)


def family_sampling_keys(section: Any) -> tuple[str, ...]:
    """Sampling keys the family's section declares beyond the shared denoise geometry.

    Video frame count and rate, a text-encoder length knob, Qwen-Image-2.1's
    reference preprocessing: whatever the family schema adds is carried, so a
    new family field reaches evaluation without a hand-kept list here.
    """

    from vrl.config.sampling_schema import DenoiseImageSamplingSection

    shared = DenoiseImageSamplingSection.model_fields
    return tuple(name for name in type(section).model_fields if name not in shared)


def resolve_eval_sampling(
    root: RootConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the parsed sampling/rollout sections into the runtime sampling dict.

    ``overrides`` supplies optional CLI-flag values keyed by the output field
    name. A falsy numeric override falls back to the config value;
    ``guidance_scale`` falls back only when the override is ``None`` so an
    explicit ``0`` (CFG disabled) is preserved. Between the CLI and the training
    ``sampling`` sits ``eval``: the values evaluation runs with when
    they differ from training (a longer denoise schedule). A key that ends up
    unset is an error naming its config path.
    """

    sampling = root.sampling
    if sampling is None:
        raise ValueError("config missing required field: sampling")
    cli = dict(overrides or {})
    eval_sampling = root.eval

    executor = root.model.executor if root.model is not None else None

    def pick(name: str, cast: type | None, *, executor_fallback: bool = False) -> Any:
        override = cli.get(name)
        use_override = override is not None if name == "guidance_scale" else bool(override)
        value = override if use_override else getattr(eval_sampling, name, None)
        if value is None:
            value = getattr(sampling, name, None)
        if value is None and executor_fallback and executor is not None:
            value = getattr(executor, name, None)
        if value is None:
            raise ValueError(f"config missing required field: sampling.{name}")
        return value if cast is None else cast(value)

    out: dict[str, Any] = {name: pick(name, cast) for name, cast in _REQUIRED_SAMPLING_KEYS}
    # Family keys arrive typed by the family's own schema; nothing to cast.
    for name in family_sampling_keys(sampling):
        out[name] = pick(name, None, executor_fallback=True)

    rollout = root.rollout
    if rollout is not None:
        if rollout.denoise_mode is not None:
            out["denoise_mode"] = str(rollout.denoise_mode)
        if rollout.noise_level is not None:
            out["noise_level"] = float(rollout.noise_level)
        if rollout.sde is not None:
            out["sde_type"] = str(rollout.sde.type)
    return out

"""Shared bases for autoregressive model wrappers on the RL path.

AR families (janus_pro, nextstep_1, ...) each wrap a ``self.language_model``
trunk and sync trainable weights under the ``model.`` prefix. The behaviors that
were byte-identical across families live here once — trainable-state load,
adapter disable, the decoder-trunk accessor, the loaded-checkpoint shape guard,
and the single-segment ``image_tokens`` replay prologue — so a new AR family
inherits them instead of copying. This is the AR analog of
:class:`vrl.models.steps.denoise.base.DiffusionModelBase`, minus the
diffusion-only pieces (versioned slots, denoise ``forward_step``): AR wrappers
keep their own family-specific forward / replay / LoRA-attach math.

Unlike ``DiffusionModelBase`` this base is intentionally NOT an ABC — AR families
expose different rollout/replay surfaces, so there is no single abstract contract
to enforce here beyond ``nn.Module``.

:class:`ARReplayCore` is the trainer-side counterpart: the minimal module set a
family needs to recompute log-probs, plus the strict checkpoint load that every
family's replay path performs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import torch
import torch.nn as nn

from vrl.generation.steps.token import (
    TokenStepBatch,
    TokenStepOutput,
)
from vrl.models.interfaces.replay import (
    ReplayRequest,
    require_replay_segments,
    require_zero_replay_timestep,
)
from vrl.models.peft_adapter import disable_adapter_on, peel_peft
from vrl.models.weight_utils import load_weights_into
from vrl.nn.quantization.targeting import LM_EXCLUDE


class ARModelBase(nn.Module):
    """Shared model base for autoregressive families on the RL path."""

    def vocab_head_split(self):
        """Separable final vocab projection, or None to replay eager logits.

        Families whose generation head ends in a plain final Linear override
        this with a :class:`vrl.models.steps.token.vocab_head.VocabHeadSplit`
        so replay can use the fused log-prob path (never materializing
        ``[B, L, V]`` logits). The default opts out — the correct answer
        whenever the head cannot be split exactly (LoRA-wrapped final layer,
        per-position structural masks on logits, projection fused into the
        trunk forward).
        """

        return None

    # What a correctly loaded checkpoint should look like, e.g. "an
    # Emu3ForConditionalGeneration checkpoint". It names an UPSTREAM class, so
    # it cannot be derived from ``type(self).__name__``; ``_require_module_attrs``
    # reads it to build the shape-mismatch message.
    checkpoint_description: ClassVar[str] = ""

    def _require_module_attrs(
        self,
        module: Any,
        attrs: tuple[str, ...],
        *,
        prefix: str = "",
    ) -> None:
        """Fail loud if ``module`` is missing any of ``attrs`` (checkpoint-shape check).

        Every AR family runs the same guard right after load: probe the freshly
        loaded module for the handful of submodules its forward path needs, and
        raise "does not look like {checkpoint_description}" on the first one
        absent. ``prefix`` only names a nested surface in the message (e.g.
        ``"model."``); callers pass the already-resolved inner module, so it
        never changes which object is probed.
        """

        for attr in attrs:
            if not hasattr(module, attr):
                raise RuntimeError(
                    f"Loaded model is missing `{prefix}{attr}` — "
                    f"does not look like {self.checkpoint_description}.",
                )

    def _lm_trunk(self) -> nn.Module:
        """Return the decoder trunk that emits ``last_hidden_state``.

        For emu3 / glm_image / nextstep_1 / llamagen the family's
        ``language_model`` IS the bare trunk (Emu3TextModel, GlmImageTextModel,
        the Qwen-style NextStep decoder, LlamaGen's vendored Transformer): the
        vocabulary projection lives in a separate top-level head, so calling the
        trunk already yields hidden states. Peeling PEFT hands attention
        backends the raw module without changing the math — PEFT replaces the
        target ``nn.Linear`` submodules in place, so the peeled trunk still
        routes through the LoRA layers.

        Janus overrides this: its ``language_model`` is a ``LlamaForCausalLM``
        whose call returns text-vocab logits, so the trunk is one hop further in.
        """

        return peel_peft(self.language_model)

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load only the trainable AR parameters from a rollout sync state.

        The payload carries ``model.*`` keys for this wrapper's ``requires_grad``
        parameters, so the wrapper itself — base or replay subclass — is the
        module the payload is validated against.
        """
        return load_weights_into(self, state_dict, prefix="model")

    @property
    def policy_cores(self) -> dict[str, Any]:
        """The one module root the rollout optimization passes walk.

        VQ decoders / vision towers live outside ``language_model`` and are
        deliberately absent: they are not the sampled policy.
        """

        return {"language_model": self.language_model}

    # Vocabulary heads (lm_head / gen_head / llamagen's ``output``) and embeddings
    # stay in the base dtype on top of the structural exclusions: the per-token
    # log-probs the RL loss consumes are computed from them.
    quantization_exclude: ClassVar[tuple[str, ...]] = LM_EXCLUDE

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable the LoRA adapter for a reference forward, or no-op when absent."""
        return disable_adapter_on(self.language_model)

    @property
    def adapter_roots(self) -> dict[str, Any]:
        """The one PEFT-adapter root, keyed by its checkpoint root name.

        Token families register the whole wrapper as the single checkpoint root
        (``trainable_modules={"model": self}``) but attach LoRA one hop in, on
        ``language_model``. Unfiltered on purpose, unlike the diffusion twin
        (:meth:`DiffusionModelBase.adapter_roots`): with exactly one root, a
        trunk that cannot export must raise, not silently publish nothing.
        """

        return {"model": self.language_model}

    def _resolve_image_token_replay(
        self,
        batch: Any,
        timestep_idx: int,
        request: ReplayRequest | None,
    ) -> tuple[dict[str, Any], Any]:
        """Shared prefix for the single-segment ``image_tokens`` replay path.

        Validates the replay request against the ``("image_tokens",)`` segment
        set and the no-timestep-axis contract, then resolves the recorded
        trajectory into its replay tensor dict and the sampled action tensor.
        Families read ``prompt_input_ids`` / ``prompt_attention_mask`` (and any
        family-specific keys such as ``uncond_*`` / ``saved_noise``) off the
        returned dict, and own the embed + forward + wrap tail.
        """

        owner = type(self).__name__
        require_zero_replay_timestep(timestep_idx, owner=owner)
        require_replay_segments(request, ("image_tokens",), owner=owner)
        from vrl.trajectory import TrajectoryResolver

        resolver = TrajectoryResolver.from_batch(batch)
        replay = resolver.replay_tensor_dict("image_tokens")
        action = resolver.role_value("image_tokens", "action")
        return replay, action


class ARReplayRolloutStubs:
    """Rollout-only surface stubs shared by AR replay models.

    The AR analog of ``vrl.models.steps.denoise.base.ReplayRolloutStubs``: replay
    models load only the modules needed to recompute log-probs, so the
    rollout-side decode surface is unreachable by construction and raises with
    the concrete class name instead of each family re-writing the stub.
    """

    def decode_image_tokens(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} cannot decode image tokens")


class ARReplayCore(nn.Module):
    """The minimal module set an AR family needs to replay recorded actions.

    Trainer replay only recomputes log-probs, so a family core constructs the
    text trunk plus its generation head and skips the VQ decoder / vision tower
    that the rollout wrapper owns. Subclasses build those submodules under the
    upstream checkpoint's own names in ``__init__`` — that is what lets the
    checkpoint state dict load by name — and inherit the load path below.

    The Hugging Face format boundary lives in
    :func:`vrl.models.steps.token.loader.load_ar_replay_checkpoint`, which skips
    shards carrying no core keys at all. A GLM-Image-sized checkpoint therefore
    leaves its vision/VQ shards on disk untouched.
    """

    # Checkpoint subfolder owning this core's config and weights — GLM-Image's
    # hybrid repo keeps the AR section under ``vision_language_encoder``.
    # ``None`` means the weights sit at the repository root.
    checkpoint_subfolder: ClassVar[str | None] = None

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: Any,
        dtype: Any,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> Self:
        """Config-init this core and strict-load its checkpoint weights.

        The shared replay-loader shape across every AR family: ``AutoConfig``
        -> ``cls(config)`` -> strict checkpoint adapter -> place on
        ``device``/``dtype`` and ``eval()``. ``trust_remote_code`` stays a
        keyword rather than a class attribute because it is a user-facing
        config key (Janus' custom config class needs it). Families keep only
        their own pre-flight, e.g. Janus' upstream-package import check.
        """

        from transformers import AutoConfig

        from vrl.models.steps.token import loader

        subfolder = cls.checkpoint_subfolder
        config_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        # transformers defaults ``subfolder`` to "", so only pass a real one.
        if subfolder is not None:
            config_kwargs["subfolder"] = subfolder
        core = cls(AutoConfig.from_pretrained(model_path, **config_kwargs))
        loader.load_ar_replay_checkpoint(
            core,
            loader.resolve_hf_checkpoint_dir(
                model_path,
                subfolder=subfolder,
                revision=revision,
            ),
        )
        return core.to(device=device, dtype=dtype).eval()


@dataclass(slots=True, kw_only=True)
class ARDiscreteTokenState:
    """Shared scheduler-visible state for discrete-token AR loops."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor

    @property
    def total_token_num(self) -> int:
        """Return the generated sequence length owned by ``token_ids``."""

        return int(self.token_ids.shape[1])


class ARDiscreteTokenRunner:
    """Shared step/finalize bookkeeping for discrete-token family runners.

    Families still own prefill, sampling, and cache advancement. This base owns
    only the engine contract that was identical across paged-CFG, GLM-Image,
    and LlamaGen: validate the scheduled rows, run one family step, and return
    ``(token_ids, logprobs)`` at finalization.
    """

    family: str = ""
    validation_family: str = ""

    @torch.no_grad()
    def step_token(
        self,
        state: ARDiscreteTokenState,
        batch: TokenStepBatch,
    ) -> TokenStepOutput:
        self._validate_ar_step_batch(state, batch)
        row_updates = self._sample_ar_step(state, batch)
        return TokenStepOutput(updated_row_lanes=row_updates)

    @torch.no_grad()
    def finalize_token(
        self,
        state: ARDiscreteTokenState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state.token_ids, state.logprobs

    def _validate_ar_step_batch(
        self,
        state: ARDiscreteTokenState,
        batch: TokenStepBatch,
    ) -> None:
        row_indices = batch.row_indices
        if any(row >= state.token_ids.shape[0] for row in row_indices):
            label = self.validation_family or self.family
            raise ValueError(f"invalid {label} row indices: {row_indices}")
        if batch.position >= state.total_token_num:
            raise ValueError(f"{type(state).__name__} has already finished sampling")
        self._validate_family_step(state, batch)

    def _validate_family_step(
        self,
        state: ARDiscreteTokenState,
        batch: TokenStepBatch,
    ) -> None:
        del state, batch

    def _sample_ar_step(
        self,
        state: ARDiscreteTokenState,
        batch: TokenStepBatch,
    ) -> dict[str, Any]:
        raise NotImplementedError


__all__ = [
    "ARDiscreteTokenRunner",
    "ARDiscreteTokenState",
    "ARModelBase",
    "ARReplayCore",
    "ARReplayRolloutStubs",
]

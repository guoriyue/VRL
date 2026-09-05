"""CPU contracts for NextStep rollout and replay token advancement."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.composition.token_autoregressive.token_loop import (
    TokenAutoregressiveLoop,
)
from vrl.models.families.nextstep_1.config import NextStep1Config
from vrl.models.families.nextstep_1.model import NextStep1Model, NextStep1ReplayModel
from vrl.models.families.nextstep_1.runner import (
    NextStep1ARModelRunner,
    NextStep1ARState,
)

_BATCH_SIZE = 2
_HIDDEN_SIZE = 2

# Carried per test rather than by a module pytestmark: the two NextStep1ARState
# tests below build no backend at all, so a file-wide label would claim a double
# on their behalf.
_RECORDING_BACKEND_STANDS_IN = pytest.mark.real_cover(
    "tests/generation/bindings/token_autoregressive/test_nextstep_vllm_paged_attention_backend.py"
    "::test_nextstep_vllm_paged_attention_matches_hf_qwen_one_step",
    why=(
        "the recording backend returns hidden + 0.125 so advance counts and rollout/replay "
        "logprob parity are assertable on CPU; a real backend needs CUDA plus vLLM's worker "
        "internals, and the gpu-lane test checks its output against HF Qwen instead"
    ),
)


@dataclass(frozen=True, slots=True)
class _SequenceState:
    branch: str
    row: int
    step_count: int = 0


class _RecordingAttentionBackend:
    def __init__(self) -> None:
        self.step_requests: list[Any] = []

    def prefill(self, request: Any) -> Any:
        batch_size = request.inputs_embeds.shape[0]
        return SimpleNamespace(
            last_hidden=request.inputs_embeds[:, -1, :],
            sequence_states=tuple(
                _SequenceState(branch=request.branch, row=row) for row in range(batch_size)
            ),
        )

    def step(self, request: Any) -> Any:
        self.step_requests.append(request)
        return SimpleNamespace(
            last_hidden=request.input_embeds[:, 0, :] + 0.125,
            sequence_states=tuple(
                _SequenceState(
                    branch=state.branch,
                    row=state.row,
                    step_count=state.step_count + 1,
                )
                for state in request.sequence_states
            ),
        )


class _FlowHead:
    input_dim = _HIDDEN_SIZE

    @staticmethod
    def net(
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return 0.15 * x + 0.1 * cond + 0.05 * timestep.unsqueeze(-1)


class _CheckpointLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_head = _FlowHead()
        self.image_in_projector = torch.nn.Identity()


class _RolloutModel:
    dtype = torch.float32

    def __init__(self, token_count: int) -> None:
        self.config = SimpleNamespace(
            guidance_scale=1.5,
            num_steps=2,
            noise_level=0.4,
            image_token_num=token_count,
            token_dim=_HIDDEN_SIZE,
        )
        self.image_head = _FlowHead()
        self._image_in_projector = torch.nn.Identity()


class _ReplayHarness:
    def __init__(self, rollout_model: _RolloutModel) -> None:
        self.config = rollout_model.config
        self.image_head = rollout_model.image_head
        self._image_in_projector = rollout_model._image_in_projector
        self.step_branches: list[str] = []
        self._next_branch = iter(("cond", "uncond"))

    def _init_kv(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        del mask
        return {
            "branch": next(self._next_branch),
            "last_hidden": embeds[:, -1, :],
        }

    @staticmethod
    def _last_hidden(kv: dict[str, Any]) -> torch.Tensor:
        return kv["last_hidden"]

    def _step_llm(
        self,
        kv: dict[str, Any],
        new_embed: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        self.step_branches.append(kv["branch"])
        hidden = new_embed + 0.125
        return {
            "branch": kv["branch"],
            "last_hidden": hidden,
        }, hidden


def test_state_derives_image_token_count_from_tokens() -> None:
    token_count = 3
    state = NextStep1ARState(
        tokens=torch.zeros(_BATCH_SIZE, token_count, _HIDDEN_SIZE),
        saved_noise=torch.zeros(_BATCH_SIZE, token_count, _HIDDEN_SIZE),
        logprobs=torch.zeros(_BATCH_SIZE, token_count),
        guidance_scale=1.0,
        num_steps=2,
        noise_level=0.5,
        paged_cond_states=[object() for _ in range(_BATCH_SIZE)],
    )

    assert state.image_token_num == token_count


@_RECORDING_BACKEND_STANDS_IN
def test_init_preserves_real_no_cfg_branch() -> None:
    token_count = 2
    model = _RolloutModel(token_count)
    backend = _RecordingAttentionBackend()
    prompt_embeds = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    prompt_mask = torch.ones(1, 2, dtype=torch.long)

    init = NextStep1ARModelRunner(model, attention_backend=backend).init_token(
        prompt_embeds,
        None,
        prompt_mask,
        None,
        image_token_num=token_count,
    )

    assert len(init.state.paged_cond_states) == 1
    assert init.state.paged_uncond_states is None
    assert set(init.row_lanes) == {"c_cond", "cond_attn"}


@_RECORDING_BACKEND_STANDS_IN
def test_checkpoint_token_dim_overwrite_drives_runner_allocations() -> None:
    config = NextStep1Config(
        use_lora=False,
        device="cpu",
        dtype="float32",
        image_token_num=3,
        token_dim=999,
    )
    model = NextStep1ReplayModel(
        config,
        language_model=_CheckpointLanguageModel(),
    )
    backend = _RecordingAttentionBackend()

    init = NextStep1ARModelRunner(model, attention_backend=backend).init_token(
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        None,
        torch.ones(1, 2, dtype=torch.long),
        None,
    )

    assert config.token_dim == _HIDDEN_SIZE
    assert init.state.tokens.shape == (1, 3, _HIDDEN_SIZE)
    assert init.state.saved_noise.shape == (1, 3, _HIDDEN_SIZE)


@_RECORDING_BACKEND_STANDS_IN
@pytest.mark.parametrize(
    ("token_count", "expected_advance_count"),
    [(1, 0), (3, 2)],
)
def test_rollout_advances_attention_only_between_tokens(
    token_count: int,
    expected_advance_count: int,
) -> None:
    _result, backend, _model, _conditioning = _run_rollout(token_count)

    assert len(backend.step_requests) == expected_advance_count


@_RECORDING_BACKEND_STANDS_IN
@pytest.mark.parametrize(
    ("token_count", "expected_advance_count"),
    [(1, 0), (3, 2)],
)
def test_replay_advances_each_cfg_branch_only_between_tokens(
    token_count: int,
    expected_advance_count: int,
) -> None:
    result, _backend, model, conditioning = _run_rollout(token_count)
    tokens, saved_noise, _rollout_logprobs = result
    replay = _ReplayHarness(model)

    NextStep1Model.recompute_logprobs(
        replay,
        *conditioning,
        tokens,
        saved_noise,
    )

    assert replay.step_branches == ["cond", "uncond"] * expected_advance_count


@_RECORDING_BACKEND_STANDS_IN
def test_skipping_final_advance_preserves_rollout_replay_logprob_parity() -> None:
    result, _backend, model, conditioning = _run_rollout(token_count=3)
    tokens, saved_noise, rollout_logprobs = result
    replay = _ReplayHarness(model)

    replay_logprobs = NextStep1Model.recompute_logprobs(
        replay,
        *conditioning,
        tokens,
        saved_noise,
    )

    torch.testing.assert_close(replay_logprobs, rollout_logprobs)


def _run_rollout(
    token_count: int,
) -> tuple[
    Any,
    _RecordingAttentionBackend,
    _RolloutModel,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    prompt_embeds = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
    )
    uncond_embeds = torch.tensor(
        [
            [[-0.1, -0.2], [-0.3, -0.4]],
            [[-0.5, -0.6], [-0.7, -0.8]],
        ],
    )
    prompt_mask = torch.ones(_BATCH_SIZE, 2, dtype=torch.long)
    uncond_mask = torch.ones_like(prompt_mask)
    conditioning = (
        prompt_embeds,
        uncond_embeds,
        prompt_mask,
        uncond_mask,
    )
    model = _RolloutModel(token_count)
    backend = _RecordingAttentionBackend()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    result = TokenAutoregressiveLoop(
        runner=NextStep1ARModelRunner(
            model,
            attention_backend=backend,
        ),
        scheduler_batch_size=_BATCH_SIZE,
        init_args=conditioning,
        init_kwargs={
            "guidance_scale": model.config.guidance_scale,
            "num_steps": model.config.num_steps,
            "noise_level": model.config.noise_level,
            "image_token_num": token_count,
            "generator": generator,
        },
    ).run()
    return result, backend, model, conditioning

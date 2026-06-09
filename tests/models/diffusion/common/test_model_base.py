"""Tests for the shared diffusion model base."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import DiffusionModelBase
from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2Model
from vrl.models.diffusion.sd3_5.model import SD3_5Model
from vrl.models.diffusion.wan_2_1.model import WanT2VDiffusersModel
from vrl.models.interfaces import ReplayResult
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_diffusion_trajectory, build_training_view


class _AdapterTransformer(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class _CompiledWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module


class _ModelBaseStub(DiffusionModelBase):
    family = "stub"

    def __init__(self) -> None:
        super().__init__()
        pipeline = SimpleNamespace(transformer=_AdapterTransformer())
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self.forward_models: list[Any] = []

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, negative_prompt, kwargs
        return {}

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        del request, encoded, kwargs
        return object()

    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        del state, step_idx
        self.forward_models.append(self.transformer)
        return {"noise_pred": torch.ones(1)}

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del replay_tensors, batch_context, latents, step_idx
        return object()

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return None

    @property
    def backend_handle(self) -> Any:
        return self.pipeline


class _BackendPipelineStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.device = torch.device("cpu")


def test_diffusion_model_base_registers_only_transformer_child() -> None:
    """Checks diffusion model base registers only transformer child."""
    runtime = _ModelBaseStub()

    assert isinstance(runtime, nn.Module)
    assert dict(runtime.named_children()) == {"transformer": runtime.transformer}
    keys = set(runtime.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith("pipeline.") for key in keys)


@pytest.mark.parametrize(
    "runtime_cls",
    [SD3_5Model, WanT2VDiffusersModel, CosmosPredict2Model],
)
def test_concrete_diffusion_runtimes_register_only_transformer(
    runtime_cls: type[DiffusionModelBase],
) -> None:
    """Checks concrete diffusion runtimes register only transformer."""
    pipeline = _BackendPipelineStub()
    runtime = runtime_cls(pipeline=pipeline, device=torch.device("cpu"))

    assert isinstance(runtime, nn.Module)
    assert runtime.pipeline is pipeline
    assert runtime.transformer is pipeline.transformer
    assert dict(runtime.named_children()) == {"transformer": pipeline.transformer}
    keys = set(runtime.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith(("vae.", "text_encoder.", "_pipeline.")) for key in keys)


@pytest.mark.parametrize(
    "runtime_cls",
    [SD3_5Model, WanT2VDiffusersModel, CosmosPredict2Model],
)
def test_concrete_diffusion_runtimes_keep_pipeline_transformer_in_sync(
    runtime_cls: type[DiffusionModelBase],
) -> None:
    """Checks concrete diffusion runtimes keep pipeline transformer in sync."""
    pipeline = _BackendPipelineStub()
    runtime = runtime_cls(pipeline=pipeline, device=torch.device("cpu"))
    replacement = nn.Linear(2, 2)

    runtime._set_transformer(replacement)

    assert runtime.transformer is replacement
    assert runtime.pipeline.transformer is replacement
    assert dict(runtime.named_children()) == {"transformer": replacement}


def test_forward_resolves_runtime_self_to_registered_transformer() -> None:
    """Checks forward resolves runtime self to registered transformer."""
    runtime = _ModelBaseStub()

    runtime.forward(object(), 0)

    assert runtime.forward_models == [runtime.transformer]


def test_replay_forward_returns_typed_replay_result() -> None:
    """Checks replay forward returns typed replay result."""
    runtime = _ModelBaseStub()
    observations = torch.zeros(2, 2, 1)
    actions = torch.ones(2, 2, 1)
    old_log_prob = torch.zeros(2, 2)
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="diffusion",
            family="sd3_5",
            task="t2i",
            prompts=["a", "b"],
            samples_per_prompt=1,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=index,
                sample_index=0,
                prompt=prompt,
                prompt_id=f"p{index}",
                group_id=f"g{index}",
                sample_id=f"s{index}",
                trajectory_id=f"t{index}",
                seed=None,
            )
            for index, prompt in enumerate(("a", "b"))
        ],
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=torch.tensor([[1, 0], [1, 0]]),
        kl=torch.zeros(2, 2),
        replay_tensors={"prompt_embeds": torch.zeros(2, 3, 4)},
        context={"scheduler": "stub"},
    )
    batch = RolloutBatch(
        observations=observations,
        actions=actions,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 1]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
        context=trajectory.context,
    )

    result = runtime.replay_forward(batch, timestep_idx=1)

    assert isinstance(result, ReplayResult)
    assert result.segments["denoise"].values["noise_pred"].shape == (1,)


def test_disable_adapter_forwards_to_transformer_context() -> None:
    """Checks disable adapter forwards to transformer context."""
    runtime = _ModelBaseStub()

    with runtime.disable_adapter():
        assert runtime.transformer.disabled is True

    assert runtime.transformer.disabled is False


def test_disable_adapter_without_transformer_adapter_is_noop() -> None:
    """Checks disable adapter without transformer adapter is no-op."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(nn.Linear(2, 2))

    with runtime.disable_adapter():
        assert runtime.transformer.training is True


def test_load_trainable_state_accepts_trainable_keys() -> None:
    """Checks load trainable state accepts trainable keys."""
    runtime = _ModelBaseStub()
    replacement = {
        "weight": torch.full_like(runtime.transformer.weight, 2.0),
        "bias": torch.full_like(runtime.transformer.bias, 3.0),
    }
    state = {f"transformer.{key}": value for key, value in replacement.items()}

    runtime.load_trainable_state(state)

    assert torch.equal(runtime.transformer.weight, replacement["weight"])
    assert torch.equal(runtime.transformer.bias, replacement["bias"])


def test_load_trainable_state_accepts_compiled_transformer_wrapper() -> None:
    """Checks weight sync loads into torch.compile wrapped modules."""
    runtime = _ModelBaseStub()
    original = runtime.transformer
    runtime._set_transformer(_CompiledWrapper(original))
    replacement = {
        "weight": torch.full_like(original.weight, 2.0),
        "bias": torch.full_like(original.bias, 3.0),
    }
    state = {f"transformer.{key}": value for key, value in replacement.items()}

    runtime.load_trainable_state(state)

    assert torch.equal(original.weight, replacement["weight"])
    assert torch.equal(original.bias, replacement["bias"])


def test_load_trainable_state_rejects_all_unmatched_keys() -> None:
    """Checks load trainable state rejects all unmatched keys."""
    runtime = _ModelBaseStub()

    with pytest.raises(ValueError, match="trainable keys prefixed"):
        runtime.load_trainable_state({"unknown": torch.ones(1)})

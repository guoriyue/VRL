from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.models.interfaces.runtime import RuntimeBuildSpec
from vrl.models.replay_loading import (
    MINIMAL_REPLAY_RUNTIME_ROLE,
    module_loading_profile_from_metadata,
    require_minimal_replay_bundle,
)


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype


class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0])

    def set_timesteps(self, n: int, device: Any = None) -> None:
        self.timesteps = torch.arange(n, device=device, dtype=torch.float32)
        self.sigmas = torch.ones(n, device=device, dtype=torch.float32)


class _TinyRuntimeModel(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.language_model = nn.Linear(1, 1)

    def replay_forward(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> Any:
        return self.load_state_dict(state_dict, strict=False)


def _spec(**overrides: Any) -> RuntimeBuildSpec:
    values = {
        "model_name_or_path": "fake/repo",
        "device": "cpu",
        "dtype": torch.float32,
        "backend_preference": ("diffusers",),
        "task_variant": "t2i",
        "use_lora": False,
        "scheduler_config": {"num_steps": 2},
    }
    values.update(overrides)
    return RuntimeBuildSpec(**values)


@pytest.mark.parametrize(
    ("module_path", "builder_name"),
    [
        (
            "vrl.models.diffusion.sd3_5.runtime",
            "build_sd3_5_replay_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.wan_2_1.runtime",
            "build_wan_2_1_replay_runtime_bundle",
        ),
        (
            "vrl.models.diffusion.cosmos.predict2.runtime",
            "build_cosmos_predict2_replay_runtime_bundle",
        ),
    ],
)
def test_diffusion_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    builder_name: str,
) -> None:
    module = __import__(module_path, fromlist=[builder_name])
    monkeypatch.setattr(
        module,
        "load_diffusers_transformer_component",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        module,
        "load_flow_match_scheduler_component",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )

    bundle = getattr(module, builder_name)(_spec())

    require_minimal_replay_bundle(bundle)
    profile = module_loading_profile_from_metadata(bundle.metadata)
    assert profile.runtime_role == MINIMAL_REPLAY_RUNTIME_ROLE
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert "pipeline" not in vars(bundle.model)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


def test_cosmos_predict25_replay_builder_keeps_diffusion_nft_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.diffusion.cosmos import predict2_5
    from vrl.models.diffusion.cosmos.predict2_5 import runtime

    monkeypatch.setattr(
        runtime,
        "load_diffusers_transformer_component",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(
        runtime,
        "load_diffusers_scheduler_component",
        lambda *_args, **_kwargs: _TinyScheduler(),
    )
    monkeypatch.setattr(
        predict2_5.model.CosmosPredict25ReplayModel,
        "apply_lora",
        lambda self, _spec: self.transformer.requires_grad_(True),
    )

    bundle = runtime.build_cosmos_predict25_replay_runtime_bundle(
        _spec(
            task_variant="text2world",
            use_lora=True,
            lora_config={"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
        ),
    )

    require_minimal_replay_bundle(bundle)
    assert bundle.backend_handle is None
    assert callable(bundle.model.diffusion_nft_prepare_transformer_input)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


def test_anima_replay_builder_uses_only_transformer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.diffusion.cosmos.anima import runtime

    monkeypatch.setattr(
        runtime,
        "_load_anima_transformer_component",
        lambda _spec: _TinyTransformer(),
    )

    bundle = runtime.build_anima_replay_runtime_bundle(
        _spec(
            task_variant="text_to_image",
            extra={
                "resolved_paths": {"transformer": "/tmp/anima-preview3-base.safetensors"},
                "scheduler_shift": 3.0,
            },
        ),
    )

    require_minimal_replay_bundle(bundle)
    profile = module_loading_profile_from_metadata(bundle.metadata)
    assert profile.runtime_role == MINIMAL_REPLAY_RUNTIME_ROLE
    assert profile.generation_only_modules == (
        "text_encoder",
        "llm_adapter",
        "vae",
        "tokenizers",
    )
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    assert not hasattr(bundle.model, "text_encoder")
    assert not hasattr(bundle.model, "vae")
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline
    with pytest.raises(RuntimeError, match="encode prompts"):
        bundle.model.encode_prompt("prompt")


def test_anima_empty_prompts_are_replaced_before_tokenization() -> None:
    from vrl.models.diffusion.cosmos.anima.model import _non_empty_prompts

    assert _non_empty_prompts(["", "  ", "anime"]) == [".", ".", "anime"]


@pytest.mark.parametrize(
    ("module_path", "builder_name", "model_attr", "spec_kwargs"),
    [
        (
            "vrl.models.ar.janus_pro.runtime",
            "build_janus_pro_replay_runtime_bundle",
            "JanusProReplayModel",
            {"backend_preference": ("native",), "task_variant": "ar_t2i"},
        ),
        (
            "vrl.models.ar.nextstep_1.runtime",
            "build_nextstep_1_replay_runtime_bundle",
            "NextStep1ReplayModel",
            {"backend_preference": ("native",), "task_variant": "ar_t2i"},
        ),
    ],
)
def test_ar_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    builder_name: str,
    model_attr: str,
    spec_kwargs: dict[str, Any],
) -> None:
    module = __import__(module_path, fromlist=[builder_name])
    monkeypatch.setattr(module, model_attr, _TinyRuntimeModel)

    bundle = getattr(module, builder_name)(_spec(**spec_kwargs))

    require_minimal_replay_bundle(bundle)
    assert bundle.backend_handle is None
    assert set(bundle.trainable_modules) == {"model"}


def test_sd3_after_bundle_hook_ignores_replay_model_without_pipeline() -> None:
    from vrl.scripts.diffusion.sd3_5.train import _offload_driver_frozen_modules

    class _ReplayLike:
        @property
        def pipeline(self) -> Any:
            raise RuntimeError("no pipeline")

    _offload_driver_frozen_modules(_ReplayLike())

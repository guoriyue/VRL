"""Download-free ``from_build`` loading test for Wan 2.1 Image-to-Video.

Mirrors ``tests/models/sd3_5/test_model_loading.py``: monkeypatch the
diffusers pipeline ``from_pretrained`` to return a fake pipeline (no Hub fetch,
no real weights) and assert the loader-constructed state, NOT literal YAML
config values. The Wan I2V wrapper has a load-time branch the sd3 test does not
exercise: single-GPU offload mode is selected from the ``model.offload_mode`` config
key
(``WanI2VDiffusersModel.from_build`` in ``vrl/models/wan_2_1/model.py``):

  * ``offload_mode: sequential`` -> frozen modules stay on CPU until the shared
    rollout builder attaches LoRA, then ``enable_sequential_cpu_offload`` installs
    the streaming hooks;
  * ``offload_mode: model`` follows the same adapter-before-hook ordering with
    ``enable_model_cpu_offload``;
  * ``offload_mode: none`` -> vae fp32 + text_encoder/image_encoder build dtype staged to
    the build device.

In every branch the generation-only modules (vae / text_encoder / image_encoder)
are frozen and the progress bar is disabled. Wan 2.2 A14B dual-stage pipelines
are accepted when they expose ``transformer_2``; expand-timesteps pipelines
remain unsupported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import RecordingModule
from vrl.config.precision import RolePrecision
from vrl.models.interfaces.runtime import ModelBuild, PipelineOffloadMode, RolloutBuildOptions


def _rollout_build_options(offload_mode: str) -> RolloutBuildOptions:
    return RolloutBuildOptions(
        prompt_encoder_dtype=torch.bfloat16,
        pipeline_offload_mode=offload_mode,
    )


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = RecordingModule()
        self.transformer_2 = None
        self.vae = RecordingModule()
        self.text_encoder = RecordingModule()
        self.image_encoder = RecordingModule()
        self.device = "cpu"
        # Single-transformer Wan 2.1 I2V: no boundary_ratio / expand_timesteps.
        self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=False)
        self.progress_bar_disabled: bool | None = None
        self.sequential_offload_gpu: int | None = None
        self.model_offload_gpu: int | None = None

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_bar_disabled = disable

    def enable_sequential_cpu_offload(self, *, gpu_id: int) -> None:
        self.sequential_offload_gpu = gpu_id

    def enable_model_cpu_offload(self, *, gpu_id: int) -> None:
        self.model_offload_gpu = gpu_id


def _canonical_model_config(**overrides: Any) -> dict[str, Any]:
    return {
        "boundary_ratio": None,
        "trainable_transformers": ["transformer"],
        **overrides,
    }


def _i2v_build(
    *,
    model_name_or_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    device: torch.device | None = None,
    rollout: RolloutBuildOptions | None = None,
    **model_config: Any,
) -> ModelBuild:
    """A Wan I2V build; keyword overrides ride the canonical ``model_config`` block."""
    return ModelBuild(
        model_name_or_path=model_name_or_path,
        revision=None,
        device=device if device is not None else torch.device("cuda:0"),
        parameter_dtype=torch.bfloat16,
        family="wan_2_1_i2v",
        precision=RolePrecision("bf16", "tf32"),
        model_config=_canonical_model_config(**model_config),
        rollout=rollout,
    )


def _patch_from_pretrained(monkeypatch) -> tuple[_FakePipeline, list[dict[str, Any]]]:
    from diffusers import WanImageToVideoPipeline

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return pipeline

    monkeypatch.setattr(
        WanImageToVideoPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    return pipeline, calls


def _assert_frozen_and_loaded(pipeline: _FakePipeline, calls: list[dict[str, Any]]) -> None:
    """Shared assertions for every offload branch (load call + freezing)."""
    assert calls == [
        {
            "model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "torch_dtype": {
                "default": torch.bfloat16,
                "vae": torch.float32,
                "text_encoder": torch.bfloat16,
                "image_encoder": torch.bfloat16,
            },
        },
    ]
    assert pipeline.progress_bar_disabled is True
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.image_encoder):
        assert module.requires_grad_enabled is False


@pytest.mark.parametrize("family", ("wan_2_1", "wan_2_1_i2v"))
def test_wan_resolver_projects_pipeline_offload_to_rollout_only(family: str, monkeypatch) -> None:
    from diffusers import DiffusionPipeline
    from omegaconf import OmegaConf

    from vrl.models.families.registry import get_model_family_entry

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(lambda *a, **k: {"boundary_ratio": None}),
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "family": family,
                "path": "fake/repo",
                "revision": "a" * 40,
                "offload_mode": "sequential",
                "use_lora": False,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"dtype": "bf16"},
            },
            "distributed": {"training": {"strategy": "single_process"}},
        },
    )
    entry = get_model_family_entry(family)
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    rollout = entry.resolve_model_build(root, "cpu", precision=precision, for_rollout=True)
    replay = entry.resolve_model_build(root, "cpu", precision=precision, for_rollout=False)

    assert cfg.model.offload_mode == "sequential"
    assert rollout.require_rollout().pipeline_offload_mode == "sequential"
    assert replay.rollout is None
    assert "offload_mode" not in (rollout.model_config or {})
    assert "offload_mode" not in (replay.model_config or {})


@pytest.mark.parametrize(
    ("mode", "device_index", "hooked", "untouched"),
    [
        ("sequential", 3, "sequential_offload_gpu", "model_offload_gpu"),
        ("model", 2, "model_offload_gpu", "sequential_offload_gpu"),
    ],
)
def test_wan_i2v_offload_hooks_are_deferred_until_after_the_loader(
    monkeypatch, mode: str, device_index: int, hooked: str, untouched: str
) -> None:
    """Either offload mode installs its hooks only in ``apply_generation_offload``,
    on the build device's index, and stages every module dtype-only on CPU until
    Accelerate streams it."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = _i2v_build(
        device=torch.device(f"cuda:{device_index}"),
        rollout=_rollout_build_options(mode),
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert getattr(pipeline, hooked) is None
    model.apply_generation_offload(build)
    assert model._pipeline_offload is PipelineOffloadMode(mode)
    # Installing the committed mode twice must not touch the hook graph again.
    monkeypatch.setattr(
        model,
        "_enable_pipeline_offload",
        lambda _mode: pytest.fail("committed hooks were installed twice"),
    )
    model.apply_generation_offload(build)
    with pytest.raises(RuntimeError, match="offload mode changed"):
        model.apply_generation_offload(
            _i2v_build(device=build.device, rollout=_rollout_build_options("none")),
        )
    assert model._pipeline_offload is PipelineOffloadMode(mode)
    assert model.pipeline_cpu_offload_healthy
    assert getattr(pipeline, hooked) == device_index
    assert getattr(pipeline, untouched) is None
    assert pipeline.vae.to_calls == [(None, torch.float32)]
    assert pipeline.text_encoder.to_calls == [(None, torch.bfloat16)]
    assert pipeline.image_encoder.to_calls == [(None, torch.bfloat16)]


def test_wan_i2v_sequential_offload_attaches_lora_without_full_gpu_move(monkeypatch) -> None:
    """The 16.4B rollout stays on CPU until its final module tree is hooked."""

    import peft

    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    monkeypatch.setattr(
        peft,
        "get_peft_model",
        lambda transformer, _cfg, **_kwargs: transformer,
    )
    build = _i2v_build(
        rollout=_rollout_build_options("sequential"),
        use_lora=True,
        lora={"rank": 2, "alpha": 2, "target_modules": ["to_q"]},
    )

    model = WanI2VDiffusersModel.from_build(build)
    model.apply_lora(build)

    assert pipeline.transformer.to_calls == []
    assert pipeline.sequential_offload_gpu is None
    model.apply_generation_offload(build)
    assert pipeline.sequential_offload_gpu == 0


def test_wan_full_finetune_normalizes_rollout_parameter_dtype() -> None:
    """Rollout storage must accept the exact BF16 payload exported by FSDP."""
    from torch import nn

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    class _MixedTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bf16_weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.fp32_table = nn.Parameter(torch.ones(2, dtype=torch.float32))

    pipeline = SimpleNamespace(
        transformer=_MixedTransformer(),
        transformer_2=None,
        config=SimpleNamespace(boundary_ratio=None),
    )
    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cpu"))
    build = SimpleNamespace(
        parameter_dtype=torch.bfloat16,
        defer_trainable_device_move=False,
        model_config=_canonical_model_config(),
    )

    model.apply_full_finetune(build)
    payload = {
        f"transformer.{name}": torch.zeros_like(parameter)
        for name, parameter in pipeline.transformer.named_parameters()
    }
    model.validate_trainable_state(payload)

    assert {parameter.dtype for parameter in pipeline.transformer.parameters()} == {
        torch.bfloat16,
    }
    assert all(parameter.requires_grad for parameter in pipeline.transformer.parameters())


def test_wan_full_finetune_defers_dtype_normalization_to_fsdp() -> None:
    """Replay construction leaves CPU storage untouched until FSDP shards it."""
    from torch import nn

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    pipeline = SimpleNamespace(
        transformer=nn.Linear(2, 2, dtype=torch.float32),
        transformer_2=None,
        config=SimpleNamespace(boundary_ratio=None),
    )
    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cuda:0"))
    build = SimpleNamespace(
        parameter_dtype=torch.bfloat16,
        defer_trainable_device_move=True,
        model_config=_canonical_model_config(),
    )

    model.apply_full_finetune(build)

    assert {parameter.dtype for parameter in pipeline.transformer.parameters()} == {
        torch.float32,
    }
    assert all(parameter.device.type == "cpu" for parameter in pipeline.transformer.parameters())


@pytest.mark.parametrize("family", ["WanT2VReplayModel", "WanI2VReplayModel"])
@pytest.mark.parametrize("dual", [False, True])
def test_wan_replay_loads_trainable_state_without_pipeline(family, dual) -> None:
    from vrl.models.families.wan_2_1 import model as wan

    model = getattr(wan, family)(
        transformer=torch.nn.Linear(2, 2),
        transformer_2=torch.nn.Linear(2, 2) if dual else None,
        scheduler=object(),
        device=torch.device("cpu"),
        boundary_ratio=0.875 if dual else None,
        trainable_transformers="both" if dual else "transformer",
    )
    payload = {}
    for root, module in model.trainable_modules.items():
        module.requires_grad_(True)
        payload.update(
            {
                f"{root}.{name}": torch.full_like(value, 0.25)
                for name, value in module.named_parameters()
            }
        )
    model.load_trainable_state(payload)
    model.verify_trainable_state(payload)
    assert not model.uses_pipeline_cpu_offload
    assert model.pipeline_cpu_offload_healthy
    model.reset_pipeline_cpu_offload()
    assert "_pipeline_offload" not in vars(model)
    assert "_pipeline" not in vars(model)


def test_wan_replay_full_finetune_ignores_rollout_pipeline_offload() -> None:
    """A normalized replay build moves its transformer through the trainer path."""
    from vrl.models.families.wan_2_1.model import WanT2VReplayModel

    transformer = RecordingModule()
    model = WanT2VReplayModel(
        transformer=transformer,
        scheduler=object(),
        device=torch.device("cuda:1"),
    )
    build = SimpleNamespace(
        parameter_dtype=torch.bfloat16,
        defer_trainable_device_move=False,
        model_config=_canonical_model_config(),
        rollout=None,
    )

    model.apply_full_finetune(build)

    assert transformer.to_calls == [(torch.device("cuda:1"), torch.bfloat16)]


def test_wan_sequential_offload_weight_sync_changes_forward() -> None:
    """Weight sync detaches public hooks, updates LoRA, and reinstalls streaming."""

    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    from torch import nn

    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    class _FailingLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(3, 2, bias=False)
            self.fail_forward = False

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            output = super().forward(value)
            if self.fail_forward:
                raise RuntimeError("injected leaf failure")
            return output

    class _TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale_shift_table = nn.Parameter(torch.ones(2))
            self.proj = _FailingLinear()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.proj(value) * self.scale_shift_table

    class _HookedPipeline:
        def __init__(self) -> None:
            self.transformer = _TinyTransformer()
            self.original_proj = self.transformer.proj
            self.transformer_2 = None
            self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=False)
            self.enable_calls = 0
            self.remove_calls = 0

        def enable_sequential_cpu_offload(self, *, gpu_id: int) -> None:
            assert gpu_id == 0
            self.enable_calls += 1
            cpu_offload(
                self.transformer,
                execution_device=torch.device("cpu"),
                offload_buffers=True,
            )

        def remove_all_hooks(self) -> None:
            self.remove_calls += 1
            remove_hook_from_module(self.transformer, recurse=True)

    pipeline = _HookedPipeline()
    build = ModelBuild(
        model_name_or_path="unused",
        revision=None,
        family="wan_2_1_i2v",
        precision=RolePrecision("fp32", "tf32"),
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        defer_trainable_device_move=False,
        model_config=_canonical_model_config(
            use_lora=True,
            lora={"rank": 2, "alpha": 2, "target_modules": ["proj"]},
        ),
        rollout=_rollout_build_options("sequential"),
    )
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=build.device,
    )
    model.apply_lora(build)
    model.apply_generation_offload(build)
    assert model.uses_pipeline_cpu_offload
    assert all(parameter.device.type == "meta" for parameter in model.transformer.parameters())

    with pytest.raises(ValueError, match="unexpected trainable state keys"):
        model.load_trainable_state({"unknown.weight": torch.ones(1)})
    assert pipeline.remove_calls == 0
    assert pipeline.enable_calls == 1
    assert model.pipeline_cpu_offload_healthy

    sample = torch.tensor([[0.25, -0.5, 1.0]])
    before = model.transformer(sample).detach().clone()
    payload = {
        f"transformer.{name}": torch.full(parameter.shape, 0.5, dtype=parameter.dtype)
        for name, parameter in model.transformer.named_parameters()
        if parameter.requires_grad
    }
    model.load_trainable_state(payload)
    after = model.transformer(sample).detach()

    assert not torch.equal(after, before)
    assert pipeline.remove_calls == 1
    assert pipeline.enable_calls == 2
    assert model.uses_pipeline_cpu_offload
    assert all(parameter.device.type == "meta" for parameter in model.transformer.parameters())

    # Accelerate does not run a leaf's post-forward hook when that leaf raises.
    # The public reset must materialize the stranded weight on CPU, then re-arm
    # the full tree before the worker is allowed to retry or hand off its GPU.
    pipeline.original_proj.fail_forward = True
    with pytest.raises(RuntimeError, match="injected leaf failure"):
        model.transformer(sample)
    assert pipeline.original_proj.weight.device.type != "meta"

    pipeline.original_proj.fail_forward = False
    model.reset_pipeline_cpu_offload()

    assert pipeline.remove_calls == 2
    assert pipeline.enable_calls == 3
    assert model.pipeline_cpu_offload_healthy
    assert all(parameter.device.type == "meta" for parameter in model.transformer.parameters())
    assert model.transformer(sample).shape == (1, 2)

    model.verify_trainable_state(payload)
    assert model.pipeline_cpu_offload_healthy
    assert all(parameter.device.type == "meta" for parameter in model.transformer.parameters())
    assert torch.equal(model.transformer(sample).detach(), after)
    wrong = {name: value + 1 for name, value in payload.items()}
    with pytest.raises(RuntimeError, match="trainable weight verification failed") as error:
        model.verify_trainable_state(wrong)
    assert "installed weight content differs" in str(error.value.__cause__)
    assert pipeline.remove_calls == 4
    assert pipeline.enable_calls == 5
    assert not model.pipeline_cpu_offload_healthy
    assert all(parameter.device.type == "meta" for parameter in model.transformer.parameters())


class _BlockOffloadPipeline:
    """Pipeline double for ``offload_mode: block``: real hooks, tiny modules."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.transformer = transformer
        self.transformer_2 = None
        self.vae = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=False)

    @property
    def components(self) -> dict[str, Any]:
        return {
            "transformer": self.transformer,
            "transformer_2": self.transformer_2,
            "vae": self.vae,
            "scheduler": object(),
        }

    def remove_all_hooks(self) -> None:
        raise AssertionError("block offload must not rely on the Accelerate hook removal")


def test_wan_block_offload_weight_sync_changes_forward() -> None:
    """Block offload cycles its own hook registry around weight sync and reset."""

    from diffusers.hooks.group_offloading import _get_top_level_group_offload_hook
    from torch import nn

    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel
    from vrl.models.peft_adapter import peel_peft

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 3, bias=False)
            self.fail_forward = False

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if self.fail_forward:
                raise RuntimeError("injected block failure")
            return self.proj(value)

    class _TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block(), _Block()])
            self.out = nn.Linear(3, 2, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                value = block(value)
            return self.out(value)

    def hooked(module: nn.Module) -> bool:
        return _get_top_level_group_offload_hook(peel_peft(module)) is not None

    pipeline = _BlockOffloadPipeline(_TinyTransformer())
    build = ModelBuild(
        model_name_or_path="unused",
        revision=None,
        family="wan_2_1_i2v",
        precision=RolePrecision("fp32", "tf32"),
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        defer_trainable_device_move=False,
        model_config=_canonical_model_config(
            use_lora=True,
            lora={"rank": 2, "alpha": 2, "target_modules": ["proj"]},
        ),
        rollout=_rollout_build_options("block"),
    )
    model = WanI2VDiffusersModel(pipeline=pipeline, device=build.device)
    model.apply_lora(build)
    model.apply_generation_offload(build)
    assert model.uses_pipeline_cpu_offload
    # Hooks sit on the adapted transformer's own block list, behind the PEFT
    # wrapper: grouping the wrapper's single child would stream the whole model.
    inner = peel_peft(model.transformer)
    assert inner is not model.transformer
    assert hooked(inner) and hooked(inner.blocks[0]) and hooked(pipeline.vae)

    sample = torch.tensor([[0.25, -0.5, 1.0]])
    before = model.transformer(sample).detach().clone()
    payload = {
        f"transformer.{name}": torch.full(parameter.shape, 0.5, dtype=parameter.dtype)
        for name, parameter in model.transformer.named_parameters()
        if parameter.requires_grad
    }
    model.load_trainable_state(payload)
    after = model.transformer(sample).detach()

    assert not torch.equal(after, before)
    assert model.pipeline_cpu_offload_healthy
    assert hooked(inner) and hooked(inner.blocks[1])
    # The reinstalled groups own the synced weights: a forward after the sync
    # reads the same values the verification pass reads.
    model.verify_trainable_state(payload)
    assert torch.equal(model.transformer(sample).detach(), after)

    # A block that raises skips its post-forward offload. The public reset must
    # re-arm every registry so the next forward starts from a clean hook chain.
    block = inner.blocks[0]
    block.fail_forward = True
    with pytest.raises(RuntimeError, match="injected block failure"):
        model.transformer(sample)
    block.fail_forward = False
    model.reset_pipeline_cpu_offload()
    assert model.pipeline_cpu_offload_healthy
    assert hooked(inner) and hooked(block)
    assert torch.equal(model.transformer(sample).detach(), after)


def test_wan_block_offload_sees_blocks_behind_a_compile_wrapper() -> None:
    """``torch.compile`` wraps the transformer in an OptimizedModule whose only
    child is the whole model; block offload must group the inner ``blocks`` or
    per-block streaming degrades into one whole-model move."""

    from torch import nn

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                value = block(value)
            return value

    transformer = _Transformer()
    compiled = torch.compile(transformer)
    pipeline = _BlockOffloadPipeline(compiled)

    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cpu"))
    components = dict(model._offload_components())

    assert components["transformer"] is transformer
    assert next(iter(components["transformer"].named_children()))[0] == "blocks"


def test_wan_block_offload_streams_only_the_transformers(monkeypatch) -> None:
    """Experts prefetch on a copy stream; the VAE and encoders move synchronously.

    The VAE's tiled decode runs its modules in a data-dependent order, so a
    traced prefetch chain would not match the next call; the transformers run
    the same block order on every step.
    """

    import diffusers.hooks

    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    calls: dict[str, dict[str, Any]] = {}

    def record(component: torch.nn.Module, **kwargs: Any) -> None:
        calls[kwargs["name"]] = kwargs

    def fake_apply_group_offloading(component: torch.nn.Module, **kwargs: Any) -> None:
        for name, candidate in pipeline.components.items():
            if candidate is component:
                record(component, name=name, **kwargs)

    monkeypatch.setattr(diffusers.hooks, "apply_group_offloading", fake_apply_group_offloading)

    pipeline = _BlockOffloadPipeline(torch.nn.Linear(3, 2))
    pipeline.transformer_2 = torch.nn.Linear(3, 2)
    build = SimpleNamespace(
        device=torch.device("cuda:1"),
        model_config=_canonical_model_config(),
        rollout=_rollout_build_options("block"),
    )
    model = WanI2VDiffusersModel(pipeline=pipeline, device=build.device)
    model.apply_generation_offload(build)

    assert set(calls) == {"transformer", "transformer_2", "vae"}
    assert {name for name, call in calls.items() if call["use_stream"]} == {
        "transformer",
        "transformer_2",
    }
    assert {call["onload_device"] for call in calls.values()} == {torch.device("cuda:1")}
    assert {call["num_blocks_per_group"] for call in calls.values()} == {1}


def test_wan_pipeline_offload_remove_failure_is_permanently_broken() -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline = _FakePipeline()
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=torch.device("cuda:0"),
    )
    build = SimpleNamespace(
        device=model.device,
        model_config=_canonical_model_config(),
        rollout=_rollout_build_options("sequential"),
    )
    model.apply_generation_offload(build)
    pipeline.remove_all_hooks = lambda: (_ for _ in ()).throw(RuntimeError("remove failed"))

    with pytest.raises(RuntimeError, match="hook removal failed"):
        model.reset_pipeline_cpu_offload()
    with pytest.raises(RuntimeError, match="not reusable"):
        model.reset_pipeline_cpu_offload()

    assert model.uses_pipeline_cpu_offload
    assert not model.pipeline_cpu_offload_healthy


def test_wan_pipeline_offload_initial_install_failure_is_permanently_broken() -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline = _FakePipeline()
    pipeline.enable_sequential_cpu_offload = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("install failed"),
    )
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=torch.device("cuda:0"),
    )
    build = SimpleNamespace(
        device=model.device,
        model_config=_canonical_model_config(),
        rollout=_rollout_build_options("sequential"),
    )

    with pytest.raises(RuntimeError, match="install failed"):
        model.apply_generation_offload(build)
    with pytest.raises(RuntimeError, match="not reusable"):
        model.apply_generation_offload(build)

    assert model.uses_pipeline_cpu_offload
    assert not model.pipeline_cpu_offload_healthy


def test_wan_pipeline_offload_operation_failure_never_publishes_healthy_hooks() -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline = _FakePipeline()
    pipeline.remove_all_hooks = lambda: None
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=torch.device("cuda:0"),
    )
    model.apply_generation_offload(
        SimpleNamespace(
            device=model.device,
            model_config=_canonical_model_config(),
            rollout=_rollout_build_options("sequential"),
        ),
    )

    with pytest.raises(RuntimeError, match="partial state") as failure:
        model._with_pipeline_cpu_offload_suspended(
            lambda: (_ for _ in ()).throw(RuntimeError("copy failed")),
            operation="trainable weight sync",
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert str(failure.value.__cause__) == "copy failed"
    assert pipeline.sequential_offload_gpu == 0
    assert not model.pipeline_cpu_offload_healthy


def test_wan_pipeline_offload_reinstall_failure_is_permanently_broken() -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline = _FakePipeline()
    pipeline.remove_all_hooks = lambda: None
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=torch.device("cuda:0"),
    )
    model.apply_generation_offload(
        SimpleNamespace(
            device=model.device,
            model_config=_canonical_model_config(),
            rollout=_rollout_build_options("sequential"),
        ),
    )
    pipeline.enable_sequential_cpu_offload = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("reinstall failed"),
    )

    with pytest.raises(RuntimeError, match="hook reinstall failed") as failure:
        model.reset_pipeline_cpu_offload()

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert str(failure.value.__cause__) == "reinstall failed"
    assert not model.pipeline_cpu_offload_healthy


def test_wan_model_cpu_offload_reset_cycles_public_pipeline_hooks() -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline = _FakePipeline()
    remove_calls: list[bool] = []
    pipeline.remove_all_hooks = lambda: remove_calls.append(True)
    build = SimpleNamespace(
        device=torch.device("cuda:2"),
        model_config=_canonical_model_config(),
        rollout=_rollout_build_options("model"),
    )
    model = WanI2VDiffusersModel(
        pipeline=pipeline,
        device=build.device,
    )

    model.apply_generation_offload(build)
    model.reset_pipeline_cpu_offload()

    assert remove_calls == [True]
    assert pipeline.model_offload_gpu == 2
    assert model.pipeline_cpu_offload_healthy


def test_wan_i2v_from_build_no_offload_stages_frozen_modules(monkeypatch) -> None:
    """No-offload stages VAE fp32 plus encoders at the build dtype, without hooks."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, calls = _patch_from_pretrained(monkeypatch)

    build = _i2v_build()

    model = WanI2VDiffusersModel.from_build(build)

    assert model.pipeline is pipeline
    _assert_frozen_and_loaded(pipeline, calls)
    assert pipeline.sequential_offload_gpu is None
    assert pipeline.model_offload_gpu is None
    # vae stays fp32; text/image encoders ride the build dtype.
    assert pipeline.vae.to_calls == [(build.device, torch.float32)]
    assert pipeline.text_encoder.to_calls == [(build.device, torch.bfloat16)]
    assert pipeline.image_encoder.to_calls == [(build.device, torch.bfloat16)]


def test_wan_i2v_from_build_honors_local_files_only(monkeypatch) -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    _, calls = _patch_from_pretrained(monkeypatch)
    build = _i2v_build(local_files_only=True)

    WanI2VDiffusersModel.from_build(build)

    assert calls[0]["local_files_only"] is True


def test_wan_i2v_from_build_rejects_legacy_offload_keys(monkeypatch) -> None:
    """Legacy offload bools fail loud instead of becoming no-op runtime keys."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    _patch_from_pretrained(monkeypatch)

    build = _i2v_build(enable_model_cpu_offload=True)

    with pytest.raises(ValueError, match=r"model\.enable_model_cpu_offload"):
        WanI2VDiffusersModel.from_build(build)


def test_wan_i2v_from_build_accepts_dual_stage_pipeline(monkeypatch) -> None:
    """Wan 2.2 A14B dual-stage pipelines train the low-noise transformer by default."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = RecordingModule()

    build = _i2v_build(
        boundary_ratio=0.5,
        trainable_transformers=["transformer_2"],
    )

    model = WanI2VDiffusersModel.from_build(build)

    assert model.boundary_ratio == 0.5
    assert model.transformer_2 is pipeline.transformer_2
    assert model.trainable_modules == {"transformer_2": pipeline.transformer_2}


def test_wan_i2v_from_build_rejects_expand_timesteps_pipeline(monkeypatch) -> None:
    """Wan 2.2 5B expand-timesteps pipelines still need a separate runner contract."""
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=True)
    pipeline.transformer_2 = RecordingModule()

    build = _i2v_build(
        model_name_or_path="Wan-AI/Wan2.2-I2V-5B-Diffusers",
        boundary_ratio=0.5,
        trainable_transformers=["transformer_2"],
    )

    with pytest.raises(NotImplementedError, match="expand_timesteps"):
        WanI2VDiffusersModel.from_build(build)


def test_wan_model_build_normalization_is_shared_by_replay_and_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DiffusionPipeline
    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    revision = "a" * 40
    calls: list[dict[str, Any]] = []

    def fake_load_config(model_name_or_path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return {"boundary_ratio": 0.9, "expand_timesteps": False}

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(fake_load_config),
    )
    root = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "wan_2_1_i2v",
                    "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                    "revision": revision,
                    "trainable_transformers": "both",
                    "use_lora": False,
                },
                "precision": {
                    "float32_precision": "ieee",
                    "training": {"dtype": "fp32"},
                    "rollout": {"dtype": "fp32"},
                },
            },
        ),
    )
    precision = PrecisionPolicy.from_section(root.precision)
    entry = get_model_family_entry("wan_2_1_i2v")

    replay = entry.resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )
    rollout = entry.resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=True,
    )

    expected_topology = {
        "boundary_ratio": 0.9,
        "trainable_transformers": ["transformer", "transformer_2"],
    }
    assert {key: replay.model_config[key] for key in expected_topology} == expected_topology
    assert {key: rollout.model_config[key] for key in expected_topology} == expected_topology
    assert calls == [
        {
            "model_name_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "revision": revision,
        },
        {
            "model_name_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "revision": revision,
        },
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("transformer_2",)),
        (
            ["transformer_2", "transformer", "transformer_2"],
            ("transformer", "transformer_2"),
        ),
    ],
)
def test_wan_trainable_transformer_spellings_share_one_canonical_order(
    value: object,
    expected: tuple[str, ...],
) -> None:
    from vrl.models.families.wan_2_1.config import (
        normalize_wan_trainable_transformers,
    )

    assert normalize_wan_trainable_transformers(value, dual_stage=True) == expected


def test_wan_single_stage_rejects_low_noise_transformer_selection() -> None:
    from vrl.models.families.wan_2_1.config import (
        normalize_wan_trainable_transformers,
    )

    with pytest.raises(ValueError, match=r"transformer_2.*allowed=.*transformer"):
        normalize_wan_trainable_transformers(
            ["transformer_2"],
            dual_stage=False,
        )


def test_wan_build_normalization_rejects_unpinned_remote_before_config_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DiffusionPipeline

    from vrl.models.families.wan_2_1.config import normalize_wan_model_build

    calls: list[str] = []

    def fake_load_config(model_name_or_path: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(model_name_or_path)
        return {"boundary_ratio": 0.9, "expand_timesteps": False}

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(fake_load_config),
    )
    build = SimpleNamespace(
        family="wan_2_1_i2v",
        model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision=None,
        model_config={"trainable_transformers": ["transformer_2"]},
    )

    with pytest.raises(ValueError, match=r"40-character commit"):
        normalize_wan_model_build(build)
    assert calls == []


def test_wan_rollout_rejects_source_change_after_build_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel

    pipeline, _ = _patch_from_pretrained(monkeypatch)
    pipeline.config = SimpleNamespace(boundary_ratio=0.5, expand_timesteps=False)
    pipeline.transformer_2 = RecordingModule()
    build = _i2v_build(
        model_name_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        boundary_ratio=0.9,
        trainable_transformers=["transformer_2"],
    )

    with pytest.raises(
        ValueError,
        match=r"pipeline boundary_ratio disagrees.*pipeline=0\.5.*build=0\.9",
    ):
        WanI2VDiffusersModel.from_build(build)


def test_wan_t2v_loads_vae_in_fp32_before_offload_staging(monkeypatch) -> None:
    from diffusers import WanPipeline

    from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

    pipeline = _FakePipeline()
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(WanPipeline, "from_pretrained", staticmethod(load))
    build = _i2v_build()
    WanT2VDiffusersModel.from_build(build)
    assert calls[0]["torch_dtype"] == {
        "default": torch.bfloat16,
        "vae": torch.float32,
        "text_encoder": torch.bfloat16,
    }
    assert pipeline.vae.to_calls == [(build.device, torch.float32)]

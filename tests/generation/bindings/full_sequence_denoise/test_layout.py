"""Tests for diffusion request layout helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DenoiseRequestLayout,
    GenericDenoiseBatchExecutor,
)
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import DenoiseRequest, GenerationRequest


def test_denoise_options_reject_oversized_sde_window() -> None:
    """A window wider than its range is refused when the options are built."""
    with pytest.raises(ValueError, match="window_size"):
        DenoiseRequestOptions(sde_window_size=10, sde_window_range=(0, 5))


def test_diffusion_layout_rejects_sde_window_past_the_schedule() -> None:
    """The range upper bound is checked against the request's num_steps."""
    request = _request(denoise=DenoiseRequestOptions(sde_window_range=(0, 30)))

    with pytest.raises(ValueError, match="num_steps"):
        _layout().parse_sampling_params(request)


@pytest.mark.parametrize("denoise_mode", ["native", "sde"])
def test_diffusion_layout_always_builds_sde_math_params(denoise_mode: str) -> None:
    """Checks both denoise modes carry the non-optional loop math contract."""
    request = _request(denoise=DenoiseRequestOptions(denoise_mode=denoise_mode))

    params = _layout().parse_sampling_params(request)

    assert params.denoise_mode == denoise_mode
    assert params.sde.sde_type == "flow_grpo"
    assert params.sde_window is None


def test_diffusion_layout_selects_request_owned_sde_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks request window policy resolves before entering the denoise loop."""
    import pickle
    from dataclasses import replace

    draws = []

    def draw(bits):
        draws.append(bits)
        return 1234

    monkeypatch.setattr("vrl.generation.types.random.getrandbits", draw)
    request = _request(denoise=DenoiseRequestOptions(sde_window_size=2, sde_window_range=(3, 8)))
    assert draws == [64]
    assert request.sampling.get("seed") is None
    copies = [
        request,
        replace(request, samples_per_generation_batch=1),
        pickle.loads(pickle.dumps(request)),
    ]
    windows = [_layout().parse_sampling_params(copy).sde_window for copy in copies]
    assert windows[0] is not None
    assert all(window == windows[0] for window in windows)
    assert draws == [64]
    assert _layout().parse_sampling_params(_request()).sde_window is None
    assert draws == [64]


@pytest.mark.parametrize(
    "window_range",
    [(2, 2), "bad", "05", (0, 5, 10), (0,), (), (0.5, 5), (0, 5.5), (False, 5), ("0", 5)],
)
def test_denoise_options_reject_invalid_sde_window_range(window_range: object) -> None:
    """A malformed window range fails when the typed options are built."""
    with pytest.raises(ValueError, match="window_range"):
        DenoiseRequestOptions(sde_window_range=window_range)  # type: ignore[arg-type]


def test_denoise_options_reject_unknown_denoise_mode() -> None:
    with pytest.raises(ValueError, match="denoise_mode"):
        DenoiseRequestOptions(denoise_mode="custom")  # type: ignore[arg-type]


def test_diffusion_encoded_batch_preserves_shared_values_and_expands_samples() -> None:
    from vrl.generation.execution.sample_batches import GenerationSampleBatch

    executor = GenericDenoiseBatchExecutor(SimpleNamespace(), family="test", task="t2i")
    executor.batch_passthrough_keys = ("text_ids",)
    already_sized = torch.ones(3, 2)
    shared_ids = torch.ones(5, 3)
    scalar = torch.tensor(2)
    metadata = ["shared"]
    arguments = dict(generation_request=_request(), batch=GenerationSampleBatch(0, 0, 3))
    result = executor.expand_conditioning_to_batch(
        encoded=dict(
            single=torch.ones(1, 2),
            sized=already_sized,
            text_ids=shared_ids,
            scalar=scalar,
            metadata=metadata,
        ),
        **arguments,
    )
    assert torch.equal(result["single"], torch.ones(3, 2))
    assert result["sized"] is already_sized
    assert result["text_ids"] is shared_ids
    assert result["scalar"] is scalar
    assert result["metadata"] is metadata
    with pytest.raises(ValueError, match="cannot broadcast tensor batch=2"):
        executor.expand_conditioning_to_batch(encoded={"single": torch.ones(2, 2)}, **arguments)


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_extra"),
    [
        (None, {}),
        (123, {"max_sequence_length": 123}),
    ],
)
def test_diffusion_executor_only_projects_real_text_length(
    max_sequence_length: int | None,
    expected_extra: dict[str, int],
) -> None:
    """An absent family/request value stays absent at both model boundaries."""

    model = SimpleNamespace()
    executor = GenericDenoiseBatchExecutor(
        model,
        family="test",
        task="t2i",
        max_sequence_length=max_sequence_length,
    )
    params = executor.parse_sampling_params(_request())
    assert params.max_sequence_length == max_sequence_length
    assert params.text_encode_kwargs() == {
        "guidance_scale": 4.5,
        **expected_extra,
    }


def test_sde_window_is_resolved_once_at_parse_time() -> None:
    """The RESOLVED window lives on the params, so every sample batch of a
    request — and therefore every sample of a prompt group — shares one window
    (Flash-GRPO's iso-temporal grouping)."""

    params = _layout().parse_sampling_params(
        _request(
            {"seed": 7},
            denoise=DenoiseRequestOptions(sde_window_size=1, sde_window_range=(0, 10)),
        ),
    )
    assert params.sde_window is not None
    lo, hi = params.sde_window
    assert hi - lo == 1
    assert 0 <= lo < 10

    no_window = _layout().parse_sampling_params(_request())
    assert no_window.sde_window is None


def test_seeded_sde_window_is_deterministic_per_request() -> None:
    """Same request seed -> same window on every parse (multi-rank engines and
    re-parses agree without relying on the worker RNG sync); the seed does
    actually steer the draw."""

    denoise = DenoiseRequestOptions(sde_window_size=1, sde_window_range=(0, 10))
    first = _layout().parse_sampling_params(_request({"seed": 1234}, denoise=denoise))
    second = _layout().parse_sampling_params(_request({"seed": 1234}, denoise=denoise))
    assert first.sde_window == second.sde_window

    windows = {
        _layout().parse_sampling_params(_request({"seed": seed}, denoise=denoise)).sde_window
        for seed in range(30)
    }
    assert len(windows) > 1, "30 distinct seeds all drew the same window"


def _layout() -> DenoiseRequestLayout:
    """A layout with explicit fallbacks (the executor is the real source)."""
    return DenoiseRequestLayout(
        default_num_frames=1,
        default_fps=None,
        default_max_sequence_length=512,
        sde_type="flow_grpo",
    )


def _request(
    extra_sampling: dict[str, object] | None = None,
    *,
    denoise: DenoiseRequestOptions | None = None,
) -> GenerationRequest:
    sampling = {
        "num_steps": 20,
        "guidance_scale": 4.5,
        "height": 64,
        "width": 64,
        **(extra_sampling or {}),
    }
    return GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0"],
        samples_per_prompt=1,
        sampling=sampling,
        denoise=denoise,
    )


@pytest.mark.parametrize("size", [0.5, True, "2"])
def test_denoise_options_reject_noninteger_window_size(size: object) -> None:
    with pytest.raises(ValueError, match="window_size"):
        DenoiseRequestOptions(sde_window_size=size)


@pytest.mark.parametrize(
    "field", ["num_steps", "width", "height", "num_frames", "fps", "max_sequence_length"]
)
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_diffusion_layout_rejects_coerced_or_nonpositive_dimensions(field, value) -> None:
    with pytest.raises(ValueError, match="frame_count" if field == "num_frames" else field):
        _layout().parse_sampling_params(_request({field: value}))


@pytest.mark.parametrize("seed", [True, 1.5, "2"])
def test_diffusion_layout_rejects_coerced_seed(seed) -> None:
    with pytest.raises(ValueError, match=r"DenoiseRequest.seed"):
        _layout().parse_sampling_params(_request({"seed": seed}))


@pytest.mark.parametrize("field", ["width", "height", "frame_count", "num_steps", "fps"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_direct_denoise_request_validates_geometry(field, value) -> None:
    values = dict(width=8, height=8, frame_count=1, num_steps=2, guidance_scale=1.0)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        DenoiseRequest(**values)


@pytest.mark.parametrize("field", ["num_frames", "fps", "max_sequence_length"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_generic_executor_preserves_invalid_defaults_for_request_validation(field, value) -> None:
    executor = GenericDenoiseBatchExecutor(
        model=object(),
        family="test",
        task="t2i",
        **{field: value},
    )
    request = _request()
    request.sampling.pop(field, None)
    with pytest.raises(ValueError, match="frame_count" if field == "num_frames" else field):
        executor.parse_sampling_params(request)


@pytest.mark.parametrize("negative", [None, torch.ones(1, 2)])
def test_cosmos_encoded_batch_reuses_text_expansion(negative):
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.models.families.cosmos.predict2.runtime import CosmosBatchExecutor

    executor = object.__new__(CosmosBatchExecutor)
    reference = object()
    executor._reference_image_for_batch = lambda request, batch: reference
    result = executor.expand_conditioning_to_batch(
        encoded={"prompt_embeds": torch.ones(1, 2), "negative_prompt_embeds": negative},
        generation_request=_request(),
        batch=GenerationSampleBatch(0, 0, 3),
    )
    assert torch.equal(result["prompt_embeds"], torch.ones(3, 2))
    assert result["reference_image"] is reference
    if negative is None:
        assert result["negative_prompt_embeds"] is None
    else:
        assert torch.equal(result["negative_prompt_embeds"], torch.ones(3, 2))


@pytest.mark.parametrize("family", ["cosmos3", "minimax_h3"])
def test_single_sample_families_preserve_encoded_values(family):
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.models.families.cosmos.cosmos3.runtime import Cosmos3BatchExecutor
    from vrl.models.families.minimax_h3.runtime import MiniMaxH3BatchExecutor

    executor_cls = Cosmos3BatchExecutor if family == "cosmos3" else MiniMaxH3BatchExecutor
    encoded = (
        {"cond_input_ids": [1, 2], "uncond_input_ids": [0], "guidance_scale": 7.0}
        if family == "cosmos3"
        else {"prompt_embeds": torch.ones(1, 4, 8), "max_text_tokens": 4}
    )
    executor = executor_cls(SimpleNamespace())
    result = executor.expand_conditioning_to_batch(
        encoded=encoded,
        generation_request=_request(),
        batch=GenerationSampleBatch(0, 0, 1),
    )
    assert result is not encoded
    assert result.keys() == encoded.keys()
    assert all(result[key] is value for key, value in encoded.items())


def test_unseeded_window_survives_serialized_batch_split_retry(monkeypatch):
    import cloudpickle

    from vrl.generation.execution.sample_batches import (
        GenerationSampleBatch,
        execute_generation_batches,
    )
    from vrl.generation.execution.types import GenerationBatchEnvelope

    monkeypatch.setattr("vrl.generation.execution.sample_batches.empty_cuda_cache", lambda: None)
    request = _request(denoise=DenoiseRequestOptions(sde_window_size=2, sde_window_range=(0, 10)))
    windows = []

    def execute(batch):
        # Each dispatch reconstructs its own copy, as remote workers do.
        envelope = cloudpickle.loads(cloudpickle.dumps(GenerationBatchEnvelope(request, batch)))
        params = _layout().parse_sampling_params(envelope.request)
        windows.append(params.sde_window)
        if batch.sample_count > 1:
            raise RuntimeError("CUDA out of memory")
        return params.sde_window

    result = execute_generation_batches([GenerationSampleBatch(0, 0, 2)], execute)
    assert len(windows) == 3
    assert windows[0] is not None
    assert windows == [windows[0]] * 3
    assert result == [windows[0]] * 2
    assert request.sampling.get("seed") is None


def test_batch_broadcast_preserves_view_and_materialized_storage_contracts():
    from vrl.utils.tensors import expand_tensor_to_batch

    source = torch.tensor([[1.0, 2.0]], requires_grad=True)
    view = expand_tensor_to_batch(source, 3)
    copied = expand_tensor_to_batch(source, 3, materialize=True)
    assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert copied.is_contiguous()
    assert torch.equal(view, copied)
    copied.sum().backward()
    assert torch.equal(source.grad, torch.tensor([[3.0, 3.0]]))
    copied.detach()[0, 0] = 9
    assert copied[1, 0] == 1
    assert source[0, 0] == 1
    assert expand_tensor_to_batch(source, 1, materialize=True) is source
    with pytest.raises(ValueError, match="cannot broadcast tensor batch=2"):
        expand_tensor_to_batch(torch.ones(2, 3), 4)

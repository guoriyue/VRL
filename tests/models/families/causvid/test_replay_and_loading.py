"""CausVid grouped replay and fail-closed loading contracts."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseGatherer,
)
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.models.families.causvid.model import (
    CAUSVID_SOURCE_REVISION,
    CausVidModel,
    CausVidReplayModel,
    CausVidResolvedArtifacts,
    _load_generator_state_dict,
    _OfficialCausVidBackend,
    _require_causvid_flash_attention,
    _require_noncommercial_license,
    _require_pinned_source_import,
    _resolve_artifacts,
    _resolve_base_model,
    _resolve_checkpoint,
    _resolve_source_root,
    _verified_source_revision,
)
from vrl.models.families.causvid.runner import (
    OFFICIAL_CAUSVID_SCHEDULE,
    CausVidCausalRunner,
    CausVidGeometry,
)
from vrl.models.families.causvid.runtime import CausVidChunkExecutor

REPLAY_GEOMETRY = CausVidGeometry(
    latent_frames=4,
    latent_channels=1,
    latent_height=2,
    latent_width=2,
    frames_per_chunk=2,
    pixel_frames=13,
    pixel_height=16,
    pixel_width=16,
)


def test_flash_attention_preflight_accepts_either_supported_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def fake_import(module_name: str) -> object:
        imported.append(module_name)
        if module_name == "flash_attn_interface":
            raise ModuleNotFoundError(module_name)
        return object()

    monkeypatch.setattr(
        "vrl.models.families.causvid.model.importlib.import_module",
        fake_import,
    )

    _require_causvid_flash_attention()

    assert imported == ["flash_attn_interface", "flash_attn"]


def test_flash_attention_preflight_rejects_missing_or_broken_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(module_name: str) -> object:
        if module_name == "flash_attn_interface":
            raise RuntimeError("CUDA ABI mismatch")
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(
        "vrl.models.families.causvid.model.importlib.import_module",
        fake_import,
    )

    with pytest.raises(
        ImportError,
        match=r"requires FlashAttention 2 or 3.*CUDA ABI mismatch",
    ):
        _require_causvid_flash_attention()


class _ScalarTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))


class _ReplayParityBackend:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.transformer = _ScalarTransformer()
        self.allocated_caches: list[dict[str, Any]] = []
        self.cached_prefix_lengths: list[int] = []
        self.full_prefix_lengths: list[int] = []

    def new_causal_cache(
        self,
        *,
        batch_size: int,
        geometry: CausVidGeometry,
    ) -> dict[str, Any]:
        assert geometry is REPLAY_GEOMETRY
        cache: dict[str, Any] = {"batch_size": batch_size, "finalized": []}
        self.allocated_caches.append(cache)
        return cache

    def predict_x0_cached(
        self,
        noisy_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        timestep: int,
        cache: dict[str, Any],
        chunk_index: int,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        del prompt_embeds, chunk_index, geometry
        prefix = cache["finalized"]
        prefix_frames = sum(int(value.shape[1]) for value in prefix)
        self.cached_prefix_lengths.append(prefix_frames)
        prefix_term = self._prefix_term(prefix, noisy_chunk)
        prediction = (
            noisy_chunk * self.transformer.weight + prefix_term + float(timestep) / 10_000.0
        )
        if timestep == OFFICIAL_CAUSVID_SCHEDULE.cache_timestep:
            cache["finalized"].append(noisy_chunk.detach().clone())
        return prediction

    def predict_x0_full_prefix(
        self,
        prefix_and_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        per_frame_timesteps: torch.Tensor,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        del prompt_embeds
        self.full_prefix_lengths.append(int(prefix_and_chunk.shape[1]))
        output = torch.empty_like(prefix_and_chunk)
        frames_per_chunk = geometry.frames_per_chunk
        for block_start in range(0, int(prefix_and_chunk.shape[1]), frames_per_chunk):
            block_end = block_start + frames_per_chunk
            block = prefix_and_chunk[:, block_start:block_end]
            earlier = prefix_and_chunk[:, :block_start]
            prefix_term = self._prefix_term(
                [] if block_start == 0 else [earlier],
                block,
            )
            timestep_term = per_frame_timesteps[:, block_start:block_end].view(
                int(block.shape[0]),
                int(block.shape[1]),
                1,
                1,
                1,
            )
            output[:, block_start:block_end] = (
                block * self.transformer.weight + prefix_term + timestep_term / 10_000.0
            )
        return output

    @staticmethod
    def _prefix_term(
        prefix: list[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if not prefix:
            return like.new_zeros((int(like.shape[0]), 1, 1, 1, 1))
        joined = torch.cat(prefix, dim=1)
        return joined.mean(dim=(1, 2, 3, 4), keepdim=True)


def _run_behavior(backend: _ReplayParityBackend) -> Any:
    generators = (
        torch.Generator(device="cpu").manual_seed(10),
        torch.Generator(device="cpu").manual_seed(20),
    )
    initial_noise = torch.linspace(
        -1,
        1,
        2 * REPLAY_GEOMETRY.latent_frames * 4,
    ).reshape(2, *REPLAY_GEOMETRY.latent_shape)
    return CausVidCausalRunner(
        backend,
        geometry=REPLAY_GEOMETRY,
        schedule=OFFICIAL_CAUSVID_SCHEDULE,
    ).run(
        initial_noise,
        prompt_embeds=torch.ones(2, 3, 4),
        generators=generators,
    )


def test_full_prefix_replay_matches_rollout_and_backpropagates() -> None:
    backend = _ReplayParityBackend()
    behavior = _run_behavior(backend)
    replay_model = CausVidReplayModel(
        backend=backend,
        geometry=REPLAY_GEOMETRY,
        schedule=OFFICIAL_CAUSVID_SCHEDULE,
    )

    current_log_probs = replay_model.replay_log_probs(
        observations=behavior.observations,
        actions=behavior.actions,
        timesteps=behavior.timesteps,
        next_sigmas=behavior.next_sigmas,
        finalized_chunk_latents=behavior.finalized_chunk_latents,
        prompt_embeds=behavior.prompt_embeds,
    )

    assert current_log_probs.shape == (2, 2, 2)
    torch.testing.assert_close(current_log_probs, behavior.old_log_prob)
    # Two transitions for each current prefix. Growing 2 -> 4 proves replay did
    # not accidentally reuse only the first chunk's causal-attention extent.
    assert backend.full_prefix_lengths == [2, 2, 4, 4]
    (-current_log_probs.sum()).backward()
    gradient = backend.transformer.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


def test_each_rollout_allocates_a_fresh_empty_cache() -> None:
    backend = _ReplayParityBackend()

    _run_behavior(backend)
    first_call_count = len(backend.cached_prefix_lengths)
    _run_behavior(backend)

    assert len(backend.allocated_caches) == 2
    assert backend.allocated_caches[0] is not backend.allocated_caches[1]
    assert backend.cached_prefix_lengths[0] == 0
    assert backend.cached_prefix_lengths[first_call_count] == 0
    assert len(backend.allocated_caches[0]["finalized"]) == 2
    assert len(backend.allocated_caches[1]["finalized"]) == 2


def test_noncommercial_license_must_be_explicitly_accepted() -> None:
    with pytest.raises(ValueError, match=r"CC BY-NC-SA 4\.0"):
        _require_noncommercial_license({})
    with pytest.raises(ValueError, match="non-commercial"):
        _require_noncommercial_license({"accept_noncommercial_license": 1})

    _require_noncommercial_license({"accept_noncommercial_license": True})


def test_generator_checkpoint_requires_the_official_single_model_prefix(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.pt"
    torch.save({"generator": {"model.weight": torch.tensor(3.0)}}, good)
    assert _load_generator_state_dict(good) == {"weight": torch.tensor(3.0)}

    bad = tmp_path / "bad.pt"
    torch.save({"generator": {"weight": torch.tensor(3.0)}}, bad)
    with pytest.raises(ValueError, match=r"single 'model\.' prefix"):
        _load_generator_state_dict(bad)


def test_checkpoint_sha_and_source_revision_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"pinned test checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    build = SimpleNamespace(
        model_name_or_path=str(checkpoint),
        revision=None,
        model_config={"checkpoint_sha256": digest},
    )

    resolved = _resolve_checkpoint(build)

    assert resolved == checkpoint
    build.model_config["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _resolve_checkpoint(build)

    source_root = tmp_path / "CausVid"
    (source_root / "causvid").mkdir(parents=True)
    (source_root / ".git").touch()
    with pytest.raises(ValueError, match="source revision mismatch"):
        _resolve_source_root(
            {
                "causvid_source_path": str(source_root),
                "causvid_source_revision": "0" * 40,
            },
        )

    import_probe = "vrl.models.families.causvid.model.importlib.util.find_spec"
    monkeypatch.setattr(import_probe, lambda name: None)
    with pytest.raises(ImportError, match="pinned CausVid checkout is not importable"):
        _require_pinned_source_import(source_root)

    monkeypatch.setattr(
        import_probe,
        lambda name: SimpleNamespace(
            submodule_search_locations=[str(source_root / "causvid")],
        ),
    )
    _require_pinned_source_import(source_root)


def test_resolved_artifacts_retain_only_runtime_paths() -> None:
    assert [field.name for field in fields(CausVidResolvedArtifacts)] == [
        "base_model_dir",
        "checkpoint_file",
    ]


def test_source_root_returns_verified_path_and_rejects_wrong_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "CausVid"
    (source_root / "causvid").mkdir(parents=True)
    revision_probe = "vrl.models.families.causvid.model._verified_source_revision"
    monkeypatch.setattr(revision_probe, lambda path: CAUSVID_SOURCE_REVISION)

    resolved = _resolve_source_root(
        {
            "causvid_source_path": str(source_root),
            "causvid_source_revision": CAUSVID_SOURCE_REVISION,
        },
    )

    assert resolved == source_root

    monkeypatch.setattr(revision_probe, lambda path: "0" * 40)
    with pytest.raises(ValueError, match="source revision mismatch"):
        _resolve_source_root(
            {
                "causvid_source_path": str(source_root),
                "causvid_source_revision": CAUSVID_SOURCE_REVISION,
            },
        )


def test_base_model_resolution_preserves_remote_revision_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    assert _resolve_base_model({"base_model_path": str(local_model)}) == local_model

    with pytest.raises(ValueError, match=r"requires immutable model\.base_model_revision"):
        _resolve_base_model({"base_model_path": "example/base-model"})

    downloaded = tmp_path / "downloaded-model"
    downloaded.mkdir()
    calls: list[dict[str, str]] = []

    def fake_snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append({"repo_id": repo_id, "revision": revision})
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert (
        _resolve_base_model(
            {
                "base_model_path": "example/base-model",
                "base_model_revision": "base-revision",
            },
        )
        == downloaded
    )
    assert calls == [{"repo_id": "example/base-model", "revision": "base-revision"}]


def test_checkpoint_resolution_preserves_remote_revision_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = SimpleNamespace(
        model_name_or_path="example/checkpoint",
        revision=None,
        model_config={"checkpoint_file": "weights/model.pt"},
    )
    with pytest.raises(ValueError, match=r"requires immutable model\.revision"):
        _resolve_checkpoint(build)

    downloaded = tmp_path / "model.pt"
    downloaded.write_bytes(b"remote checkpoint")
    digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    calls: list[dict[str, str]] = []

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
            },
        )
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    build.revision = "checkpoint-revision"
    build.model_config["checkpoint_sha256"] = digest

    assert _resolve_checkpoint(build) == downloaded
    assert calls == [
        {
            "repo_id": "example/checkpoint",
            "filename": "weights/model.pt",
            "revision": "checkpoint-revision",
        },
    ]

    build.model_config["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _resolve_checkpoint(build)


@pytest.mark.parametrize("checkpoint_file", ["../outside.pt", "/tmp/outside.pt"])
def test_checkpoint_member_cannot_escape_local_source(
    tmp_path: Path,
    checkpoint_file: str,
) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    build = SimpleNamespace(
        model_name_or_path=str(root),
        revision=None,
        model_config={"checkpoint_file": checkpoint_file},
    )

    with pytest.raises(ValueError, match="stay within its checkpoint source"):
        _resolve_checkpoint(build)


def test_artifact_resolution_keeps_source_import_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    base_model_dir = tmp_path / "base"
    checkpoint_file = tmp_path / "model.pt"
    gated_paths: list[Path] = []
    monkeypatch.setattr(
        "vrl.models.families.causvid.model._resolve_source_root",
        lambda config: source_root,
    )
    monkeypatch.setattr(
        "vrl.models.families.causvid.model._require_pinned_source_import",
        gated_paths.append,
    )
    monkeypatch.setattr(
        "vrl.models.families.causvid.model._require_causvid_flash_attention",
        lambda: None,
    )
    monkeypatch.setattr(
        "vrl.models.families.causvid.model._resolve_base_model",
        lambda config: base_model_dir,
    )
    monkeypatch.setattr(
        "vrl.models.families.causvid.model._resolve_checkpoint",
        lambda build: checkpoint_file,
    )
    build = SimpleNamespace(model_config={"accept_noncommercial_license": True})

    resolved = _resolve_artifacts(build)

    assert gated_paths == [source_root]
    assert resolved == CausVidResolvedArtifacts(
        base_model_dir=base_model_dir,
        checkpoint_file=checkpoint_file,
    )


def test_packaged_source_without_git_uses_audited_runtime_digest(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[4] / "third_party" / "CausVid"
    packaged = tmp_path / "CausVid"
    shutil.copytree(
        source_root / "causvid",
        packaged / "causvid",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    assert _verified_source_revision(packaged) == CAUSVID_SOURCE_REVISION

    import_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from vrl.models.families.causvid.model import "
                "_require_pinned_source_import; "
                "root=Path(sys.argv[1]); _require_pinned_source_import(root); "
                "import causvid; "
                "assert list(causvid.__path__) == [str((root/'causvid').resolve())]"
            ),
            str(packaged),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_probe.returncode == 0, import_probe.stderr

    causal_model = packaged / "causvid" / "models" / "wan" / "causal_model.py"
    causal_model.write_text(
        causal_model.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="packaged runtime source digest mismatch"):
        _verified_source_revision(packaged)


class _DecodeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.tensor(0.0))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.num_heads = 1
        self.dim = 1
        self.patch_size = (1, 1, 1)
        self.text_len = 2


class _IdentityVideoVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.tensor(0.0))

    def decode(self, value: torch.Tensor, scale: Any) -> torch.Tensor:
        del scale
        return value


def test_official_backend_decode_returns_channel_first_video() -> None:
    backend = _OfficialCausVidBackend(
        transformer=_DecodeTransformer(),
        parameter_dtype=torch.float32,
        device="cpu",
        vae=_IdentityVideoVae(),
    )
    latents = torch.zeros(2, 5, 16, 3, 4)

    video = backend.decode_latents(latents)

    assert video.shape == (2, 16, 5, 3, 4)


def test_x0_conversion_matches_upstream_float64_reference() -> None:
    backend = _OfficialCausVidBackend(
        transformer=_DecodeTransformer(),
        parameter_dtype=torch.bfloat16,
        device="cpu",
    )
    noisy = torch.tensor(
        [[[[[0.1001, -0.3333]]], [[[1.2345, -2.3456]]]]],
        dtype=torch.bfloat16,
    )
    flow = torch.tensor(
        [[[[[0.7777, -1.1111]]], [[[0.2222, 3.1415]]]]],
        dtype=torch.bfloat16,
    )
    timesteps = torch.tensor([[757, 522]], dtype=torch.long)

    actual = backend._flow_to_x0(flow, noisy, timesteps)

    schedule = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
    sigmas = 8.0 * schedule / (1.0 + 7.0 * schedule)
    table_timesteps = sigmas * 1000.0
    indices = torch.argmin(
        (table_timesteps.double().view(1, 1, -1) - timesteps.double().unsqueeze(-1)).abs(),
        dim=-1,
    )
    sigma = sigmas[indices].double().view(1, 2, 1, 1, 1)
    expected = (noisy.double() - sigma * flow.double()).to(flow.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transition_rng_is_drawn_in_checkpoint_dtype() -> None:
    prediction = torch.zeros(1, 2, 1, 2, 2, dtype=torch.bfloat16)
    behavior_generator = torch.Generator(device="cpu").manual_seed(42)
    reference_generator = torch.Generator(device="cpu").manual_seed(42)

    actual = CausVidCausalRunner._transition_noise(
        prediction,
        generators=(behavior_generator,),
    )
    expected = torch.randn(
        prediction.shape[1:],
        generator=reference_generator,
        dtype=prediction.dtype,
    ).unsqueeze(0)

    assert actual is not None
    assert actual.dtype == prediction.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _GenerationBackend(_ReplayParityBackend):
    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        return torch.ones(len(prompts), 3, 4)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(latents.shape[0]),
            3,
            REPLAY_GEOMETRY.pixel_frames,
            REPLAY_GEOMETRY.pixel_height,
            REPLAY_GEOMETRY.pixel_width,
        )


def test_generation_executor_builds_trainable_chunk_trajectory() -> None:
    backend = _GenerationBackend()
    model = CausVidModel(
        backend=backend,
        geometry=REPLAY_GEOMETRY,
        schedule=OFFICIAL_CAUSVID_SCHEDULE,
    )
    executor = CausVidChunkExecutor(
        model,
        gatherer=ChunkAutoregressiveDenoiseGatherer(),
    )
    request = GenerationRequest(
        request_id="causvid-test",
        family="causvid",
        task="t2v",
        inputs=["a tiny causal video"],
        samples_per_prompt=1,
        sampling={
            "width": REPLAY_GEOMETRY.pixel_width,
            "height": REPLAY_GEOMETRY.pixel_height,
            "num_frames": REPLAY_GEOMETRY.pixel_frames,
            "fps": REPLAY_GEOMETRY.fps,
            "num_steps": 3,
            "guidance_scale": 1.0,
            "negative_prompt": "",
            "seed": 123,
        },
    )
    rows = build_sample_rows(request)

    output = executor.forward_plan(request, rows, executor.plan(request))

    assert output.output.shape == (
        1,
        3,
        REPLAY_GEOMETRY.pixel_frames,
        REPLAY_GEOMETRY.pixel_height,
        REPLAY_GEOMETRY.pixel_width,
    )
    assert output.output.dtype == torch.uint8
    assert output.trajectory is not None
    trajectory = output.trajectory
    assert trajectory.axes["temporal_chunk"].length == 2
    assert trajectory.axes["denoise_transition"].length == 2
    segment = trajectory.segments["denoise"]
    assert segment.trainable is True
    assert segment.distribution == "gaussian"
    assert segment.tensors["next_sigmas"].axes == (
        "sample",
        "temporal_chunk",
        "denoise_transition",
    )
    assert segment.tensors["prompt_embeds"].axes == ("sample",)
    assert trajectory.context == {}


def test_runtime_import_does_not_import_upstream_causal_model() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import vrl.models.families.causvid.runtime; "
                "assert 'causvid.models.wan.causal_model' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

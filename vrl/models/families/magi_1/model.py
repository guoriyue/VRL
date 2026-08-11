"""Generation-only adapter for the official MAGI-1 4.5B runtime.

MAGI-1's released code is a top-level ``inference`` package pinned to its own
PyTorch/CUDA environment.  Importing it into a VRL worker would let that package
shadow unrelated modules and would couple two incompatible dependency stacks.
This adapter therefore invokes the official ``inference/pipeline/entry.py`` in
an isolated subprocess, one sample at a time.

The official CLI writes only a decoded MP4.  It does not expose stochastic
transition likelihoods, denoise states, or an autograd-capable replay forward.
The model consequently returns a generation-only chunk result and rejects every
replay/training surface instead of fabricating policy facts.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseResult,
)
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationInput, GenerationRequest
from vrl.models.interfaces.replay import ReplayRequest, ReplayResult
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.source_integrity import runtime_source_tree_sha256
from vrl.utils.media import read_video_frames

# This adapter is implemented against the official CLI/config contract at this
# immutable source commit.  A different checkout must be audited before this
# value moves; accepting it silently could reinterpret config or output fields.
MAGI_1_SUPPORTED_SOURCE_REVISION = "0fcefdef8ce2df37a3b8890979433c06eb003328"
MAGI_1_RUNTIME_SOURCE_SHA256 = "e3ec1712437d99934776516e22960e12e9be70bd3c4d13f8f345eb3b1918d461"
_MAGI_1_RUNTIME_SOURCE_GLOBS = (
    "inference/**/*.py",
    "example/4.5B/4.5B_base_config.json",
    "example/assets/special_tokens.npz",
)
_MAGI_1_BASE_CONFIG_RELATIVE_PATH = Path("example/4.5B/4.5B_base_config.json")
_MAGI_1_SAMPLING_RUNTIME_KEYS = (
    ("num_frames", "num_frames"),
    ("height", "video_size_h"),
    ("width", "video_size_w"),
    ("num_steps", "num_steps"),
    ("fps", "fps"),
)


@dataclass(frozen=True, slots=True)
class Magi1SubprocessConfig:
    """Paths and process policy for an external MAGI-1 checkout."""

    source_path: Path
    source_revision: str
    config_path: Path
    python_executable: str = sys.executable
    timeout_seconds: float = 7200.0
    checkpoint_path: Path | None = None
    t5_pretrained_path: Path | None = None
    vae_pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        source_path = Path(self.source_path).expanduser().resolve()
        config_path = _resolve_from_source(source_path, self.config_path)
        expected_config_path = (source_path / _MAGI_1_BASE_CONFIG_RELATIVE_PATH).resolve()
        if config_path != expected_config_path:
            raise ValueError(
                "MAGI-1 adapter supports only the audited 4.5B base config at "
                f"{_MAGI_1_BASE_CONFIG_RELATIVE_PATH}; got {config_path}",
            )
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "config_path", config_path)
        source_revision = str(self.source_revision).strip()
        if source_revision != MAGI_1_SUPPORTED_SOURCE_REVISION:
            raise ValueError(
                "MAGI-1 source_revision must match the audited official commit "
                f"{MAGI_1_SUPPORTED_SOURCE_REVISION}; got {source_revision!r}",
            )
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(
            self,
            "checkpoint_path",
            _optional_source_path(source_path, self.checkpoint_path),
        )
        object.__setattr__(
            self,
            "t5_pretrained_path",
            _optional_source_path(source_path, self.t5_pretrained_path),
        )
        object.__setattr__(
            self,
            "vae_pretrained_path",
            _optional_source_path(source_path, self.vae_pretrained_path),
        )
        python_executable = str(self.python_executable).strip()
        if not python_executable:
            raise ValueError("MAGI-1 python_executable must be non-empty")
        # A path is normalized before the subprocess changes cwd to the MAGI
        # source checkout. Bare command names (for example ``python3``) remain
        # PATH-resolved by shutil/subprocess.
        if os.sep in python_executable or (os.altsep and os.altsep in python_executable):
            python_executable = str(Path(python_executable).expanduser().resolve())
        object.__setattr__(self, "python_executable", python_executable)
        if self.timeout_seconds <= 0:
            raise ValueError("MAGI-1 timeout_seconds must be > 0")

    @property
    def entry_path(self) -> Path:
        return self.source_path / "inference" / "pipeline" / "entry.py"

    @classmethod
    def from_build(cls, build: ModelBuild) -> Magi1SubprocessConfig:
        config = build.model_config or {}
        source_path = config.get("source_path")
        source_revision = config.get("source_revision")
        config_path = config.get("config_path")
        python_executable = config.get("python_executable")
        if not source_path:
            raise ValueError(
                "MAGI-1 requires model.source_path pointing to an official "
                "SandAI-org/MAGI-1 checkout",
            )
        if not source_revision:
            raise ValueError(
                "MAGI-1 requires model.source_revision pinned to the audited "
                f"official commit {MAGI_1_SUPPORTED_SOURCE_REVISION}",
            )
        if not config_path:
            raise ValueError(
                "MAGI-1 requires model.config_path (for example "
                "example/4.5B/4.5B_base_config.json)",
            )
        if not python_executable:
            raise ValueError(
                "MAGI-1 requires model.python_executable from its dedicated "
                "official environment (for example "
                "third_party/MAGI-1/.venv/bin/python)",
            )

        # Check the cheap, local compatibility boundary before a missing weight
        # component can trigger a multi-gigabyte Hub download.
        preflight = cls(
            source_path=Path(str(source_path)),
            source_revision=str(source_revision),
            config_path=Path(str(config_path)),
            python_executable=str(python_executable),
            timeout_seconds=float(config.get("timeout_seconds", 7200.0)),
        )
        preflight_payload = _preflight_local_installation(preflight)
        _validate_magi_sampling_contract(
            preflight_payload,
            sampling=build.sampling_config or {},
        )
        checkpoint_path, t5_path, vae_path = _resolve_weight_components(build)
        return cls(
            source_path=Path(str(source_path)),
            source_revision=str(source_revision),
            config_path=Path(str(config_path)),
            python_executable=str(python_executable),
            timeout_seconds=float(config.get("timeout_seconds", 7200.0)),
            checkpoint_path=checkpoint_path,
            t5_pretrained_path=t5_path,
            vae_pretrained_path=vae_path,
        )


def normalize_magi_1_model_build(build: ModelBuild) -> ModelBuild:
    """Resolve the external Python on the driver before Ray serialization.

    Ray packages source code but deliberately excludes virtual environments.
    A path-like interpreter therefore remains an external deployment input and
    must be absolute in the worker payload. Multi-node deployments must provide
    that same path on each rollout node (normally through a shared mount or
    container image); the worker preflight verifies it before weight download.
    """

    if build.family != "magi_1":
        raise ValueError(f"MAGI-1 build normalizer received family {build.family!r}")
    config = dict(build.model_config or {})
    raw_executable = config.get("python_executable")
    if raw_executable:
        executable = str(raw_executable).strip()
        if os.sep in executable or (os.altsep and os.altsep in executable):
            path = Path(executable).expanduser()
            if not path.is_absolute():
                repo_root = Path(__file__).resolve().parents[4]
                path = repo_root / path
            config["python_executable"] = str(path.resolve())
    build.model_config = config
    return build


class Magi1SubprocessModel(torch.nn.Module):
    """RuntimeModel facade over the official generation-only MAGI-1 CLI."""

    supports_versioned_trainable_state = False

    def __init__(
        self,
        config: Magi1SubprocessConfig,
        *,
        device: Any = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self._device = torch.device(device or "cpu")
        self._base_config = prepare_magi_runtime_config(
            _preflight_local_installation(config),
            config=config,
            sampling={},
            sample_index=0,
        )
        self.precision: Any = None

    @classmethod
    def from_build(cls, build: ModelBuild) -> Magi1SubprocessModel:
        build.require_rollout()
        return cls(Magi1SubprocessConfig.from_build(build), device=build.device)

    @property
    def device(self) -> torch.device:
        """Logical worker device; persistent weights live only in subprocesses."""

        return self._device

    @property
    def trainable_modules(self) -> dict[str, Any]:
        """MAGI's released inference process exposes no trainable module."""

        return {}

    @property
    def raw_handle(self) -> Magi1SubprocessModel:
        return self

    @property
    def scheduler(self) -> None:
        return None

    def to(self, device: Any, *args: Any, **kwargs: Any) -> Magi1SubprocessModel:
        """Participate in worker parking without claiming subprocess ownership."""

        del args, kwargs
        self._device = torch.device(device)
        return self

    def move_frozen_components(self, device: Any) -> None:
        """No-op lifecycle hook: each official process exits after one sample."""

        self._device = torch.device(device)

    def generate_chunk_autoregressive(
        self,
        *,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ChunkAutoregressiveDenoiseResult:
        """Generate each requested sample through the official MAGI CLI."""

        input_value = request.inputs[chunk.prompt_index]
        mode, conditioning_path = _magi_mode(input_value)
        videos: list[torch.Tensor] = []
        temporal_chunk_counts: list[int] = []
        for sample_offset in range(chunk.sample_count):
            sample_index = chunk.sample_start + sample_offset
            flat_sample_index = chunk.prompt_index * request.samples_per_prompt + sample_index
            video, temporal_chunk_count = self._generate_one(
                prompt=input_value.prompt,
                mode=mode,
                conditioning_path=conditioning_path,
                sampling=request.sampling,
                sample_index=flat_sample_index,
            )
            videos.append(video)
            temporal_chunk_counts.append(temporal_chunk_count)

        if len(set(temporal_chunk_counts)) != 1:
            raise RuntimeError(
                "MAGI-1 returned inconsistent temporal chunk counts within one sample chunk",
            )
        return ChunkAutoregressiveDenoiseResult(
            chunk=chunk,
            output=torch.cat(videos, dim=0),
            temporal_chunk_count=temporal_chunk_counts[0],
            context={
                "runtime": "official_magi_1_subprocess",
                "upstream_entry": "inference/pipeline/entry.py",
                "source_revision": self.config.source_revision,
                "generation_only": True,
            },
        )

    def _generate_one(
        self,
        *,
        prompt: str,
        mode: str,
        conditioning_path: str | None,
        sampling: Mapping[str, Any],
        sample_index: int,
    ) -> tuple[torch.Tensor, int]:
        if conditioning_path is not None:
            resolved_conditioning = Path(conditioning_path).expanduser().resolve()
            if not resolved_conditioning.is_file():
                raise FileNotFoundError(
                    f"MAGI-1 conditioning input does not exist: {resolved_conditioning}",
                )
            conditioning_path = str(resolved_conditioning)
        prepared = prepare_magi_runtime_config(
            self._base_config,
            config=self.config,
            sampling=sampling,
            sample_index=sample_index,
        )
        runtime_config = prepared["runtime_config"]
        with tempfile.TemporaryDirectory(prefix="vrl-magi-1-") as temp_dir:
            work_path = Path(temp_dir)
            config_path = work_path / "config.json"
            output_path = work_path / "output.mp4"
            config_path.write_text(
                json.dumps(prepared, indent=2),
                encoding="utf-8",
            )
            command = build_magi_command(
                config=self.config,
                prepared_config_path=config_path,
                mode=mode,
                prompt=prompt,
                output_path=output_path,
                conditioning_path=conditioning_path,
            )
            self._run_command(command)
            if not output_path.is_file():
                raise RuntimeError(
                    "MAGI-1 official process exited successfully without writing "
                    f"the requested output: {output_path}",
                )
            frames = read_video_frames(output_path)

        # media utility returns [T,H,W,C]; generation output is [B,C,T,H,W].
        video = frames.permute(3, 0, 1, 2).contiguous().unsqueeze(0)
        frames_per_chunk = int(runtime_config["chunk_width"]) * int(
            runtime_config["temporal_downsample_factor"],
        )
        temporal_chunk_count = math.ceil(int(video.shape[2]) / frames_per_chunk)
        return video, temporal_chunk_count

    def _run_command(self, command: list[str]) -> None:
        env = magi_subprocess_environment(self.config.source_path)
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.source_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "MAGI-1 official inference timed out after "
                f"{self.config.timeout_seconds:g} seconds",
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
            raise RuntimeError(
                "MAGI-1 official inference failed with return code "
                f"{completed.returncode}: {detail or 'no subprocess output'}",
            )

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        raise RuntimeError(_generation_only_error())

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """MAGI's official inference process has no VRL-managed adapter."""

        yield

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise RuntimeError(_generation_only_error())


def prepare_magi_runtime_config(
    base_config: Mapping[str, Any],
    *,
    config: Magi1SubprocessConfig,
    sampling: Mapping[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    """Copy and specialize one official JSON config for one VRL sample."""

    if sample_index < 0:
        raise ValueError("MAGI-1 sample_index must be >= 0")
    prepared = copy.deepcopy(dict(base_config))
    runtime = prepared.get("runtime_config")
    engine = prepared.get("engine_config")
    if not isinstance(runtime, dict) or not isinstance(engine, dict):
        raise ValueError(
            "MAGI-1 config must contain object-valued runtime_config and engine_config",
        )

    if config.checkpoint_path is not None:
        runtime["load"] = str(config.checkpoint_path)
    if config.t5_pretrained_path is not None:
        runtime["t5_pretrained"] = str(config.t5_pretrained_path)
    if config.vae_pretrained_path is not None:
        runtime["vae_pretrained"] = str(config.vae_pretrained_path)

    for request_key, runtime_key in _MAGI_1_SAMPLING_RUNTIME_KEYS:
        if request_key in sampling and sampling[request_key] is not None:
            value = int(sampling[request_key])
            if value < 1:
                raise ValueError(f"MAGI-1 sampling.{request_key} must be >= 1")
            runtime[runtime_key] = value

    base_seed = sampling.get("seed", runtime.get("seed", 1234))
    if base_seed is None:
        base_seed = runtime.get("seed", 1234)
    runtime["seed"] = int(base_seed) + sample_index
    _validate_single_process_config(prepared)
    _validate_magi_sampling_contract(prepared, sampling=sampling)
    _validate_runtime_paths(prepared, source_path=config.source_path)
    return prepared


def build_magi_command(
    *,
    config: Magi1SubprocessConfig,
    prepared_config_path: Path,
    mode: str,
    prompt: str,
    output_path: Path,
    conditioning_path: str | None,
) -> list[str]:
    """Build argv for the official ``inference/pipeline/entry.py`` CLI."""

    if mode not in {"t2v", "i2v", "v2v"}:
        raise ValueError(f"unsupported MAGI-1 mode: {mode!r}")
    command = [
        config.python_executable,
        str(config.entry_path),
        "--config_file",
        str(prepared_config_path),
        "--mode",
        mode,
        "--prompt",
        prompt,
        "--output_path",
        str(output_path),
    ]
    if mode == "i2v":
        if not conditioning_path:
            raise ValueError("MAGI-1 i2v mode requires a reference image")
        command.extend(("--image_path", conditioning_path))
    elif mode == "v2v":
        if not conditioning_path:
            raise ValueError("MAGI-1 v2v mode requires a reference video")
        command.extend(("--prefix_video_path", conditioning_path))
    elif conditioning_path is not None:
        raise ValueError("MAGI-1 t2v mode does not accept a conditioning path")
    return command


def magi_subprocess_environment(source_path: Path) -> dict[str, str]:
    """Mirror the official single-GPU 4.5B launcher without overriding Ray GPU scope."""

    env = dict(os.environ)
    # The subprocess uses its dedicated interpreter plus this one verified
    # upstream package root. Inheriting a trainer PYTHONPATH could shadow either
    # boundary with an unrelated checkout or dependency.
    env["PYTHONPATH"] = str(source_path)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    # Do not inherit trainer/distributed rank variables: this adapter supports
    # the official single-process 4.5B launcher only.
    # Prompt-token and near-clean conditioning variables are part of the
    # audited official 4.5B behavior, not ambient operator overrides.
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_unused_local_port()),
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "GPUS_PER_NODE": "1",
            "NNODES": "1",
            "SPECIAL_TOKEN_PATH": str(
                (source_path / "example" / "assets" / "special_tokens.npz").resolve(),
            ),
            "PAD_HQ": "1",
            "PAD_DURATION": "1",
            "PAD_STATIC": "0",
            "PAD_DYNAMIC": "0",
            "PAD_BORDERNESS": "0",
            "PAD_THREE_D_MODEL": "0",
            "PAD_TWO_D_ANIME": "0",
            "NEG_PROMPT": "0",
            "SKIP_LOAD_MODEL": "0",
            "prev_chunks_scale": "0.7",
            "OFFLOAD_T5_CACHE": "true",
            "OFFLOAD_VAE_CACHE": "true",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        },
    )
    return env


def _preflight_local_installation(
    config: Magi1SubprocessConfig,
) -> dict[str, Any]:
    """Validate the local source/runtime boundary without resolving weights."""

    if not config.source_path.is_dir():
        raise FileNotFoundError(
            f"MAGI-1 source_path is not a directory: {config.source_path}",
        )
    if not config.entry_path.is_file():
        raise FileNotFoundError(
            f"MAGI-1 official entry point is missing: {config.entry_path}",
        )
    actual_revision = _source_head_revision(config.source_path)
    if actual_revision != config.source_revision:
        raise RuntimeError(
            "MAGI-1 source checkout revision mismatch: expected "
            f"{config.source_revision}, got {actual_revision}",
        )
    if not config.config_path.is_file():
        raise FileNotFoundError(
            f"MAGI-1 config_path does not exist: {config.config_path}",
        )
    executable = config.python_executable
    executable_path = shutil.which(executable)
    if executable_path is None and not (
        Path(executable).is_file() and os.access(executable, os.X_OK)
    ):
        raise FileNotFoundError(
            f"MAGI-1 python_executable was not found: {executable!r}",
        )
    _probe_runtime_environment(config)
    try:
        payload = json.loads(config.config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"failed to read MAGI-1 JSON config {config.config_path}: {error}",
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("MAGI-1 JSON config must contain an object")
    model = payload.get("model_config")
    runtime = payload.get("runtime_config")
    engine = payload.get("engine_config")
    if (
        not isinstance(model, dict)
        or not isinstance(runtime, dict)
        or not isinstance(engine, dict)
    ):
        raise ValueError(
            "MAGI-1 config must contain object-valued model_config, runtime_config, "
            "and engine_config",
        )
    if model.get("params_dtype") != "torch.bfloat16":
        raise ValueError(
            "MAGI-1 4.5B model_config.params_dtype must be 'torch.bfloat16'",
        )
    _validate_single_process_config(payload)
    _validate_magi_sampling_contract(payload, sampling={})
    return payload


def _probe_runtime_environment(config: Magi1SubprocessConfig) -> None:
    """Import the official CLI in its dedicated Python before weight download."""

    probe_code = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='vrl_magi_1_preflight')"
    try:
        completed = subprocess.run(
            [
                config.python_executable,
                "-c",
                probe_code,
                str(config.entry_path),
            ],
            cwd=config.source_path,
            env=magi_subprocess_environment(config.source_path),
            capture_output=True,
            text=True,
            timeout=min(config.timeout_seconds, 120.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"MAGI-1 dedicated environment preflight could not import the official CLI: {error}",
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise RuntimeError(
            "MAGI-1 dedicated environment cannot import the official CLI; "
            "install the pinned upstream requirements before resolving weights: "
            f"{detail or 'no subprocess output'}",
        )


def _validate_single_process_config(config: Mapping[str, Any]) -> None:
    engine = config["engine_config"]
    pp_size = int(engine.get("pp_size", 1))
    cp_size = int(engine.get("cp_size", 1))
    if pp_size != 1 or cp_size != 1:
        raise ValueError(
            "VRL's MAGI-1 4.5B subprocess adapter is single-process only; "
            f"official config has pp_size={pp_size}, cp_size={cp_size}",
        )


def _validate_magi_sampling_contract(
    config: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
) -> None:
    """Fail before model launch on shapes/schedules the official CLI rounds."""

    if sampling.get("guidance_scale") is not None:
        raise ValueError(
            "MAGI-1 does not map sampling.guidance_scale: the official base "
            "config owns a multi-stage text_scales schedule",
        )
    if sampling.get("negative_prompt") not in (None, ""):
        raise ValueError("MAGI-1 official 4.5B CLI does not expose negative_prompt")

    runtime = dict(config["runtime_config"])
    # During build preflight, overrides have not yet been applied to the JSON
    # payload. During request preparation they have; applying them again is
    # idempotent and keeps both paths on one validation contract.
    for request_key, runtime_key in _MAGI_1_SAMPLING_RUNTIME_KEYS:
        value = sampling.get(request_key)
        if value is not None:
            runtime[runtime_key] = value
    chunk_width = int(runtime.get("chunk_width", 0))
    temporal_factor = int(runtime.get("temporal_downsample_factor", 0))
    if chunk_width < 1 or temporal_factor < 1:
        raise ValueError(
            "MAGI-1 runtime_config.chunk_width and temporal_downsample_factor must be >= 1",
        )
    decoded_chunk_frames = chunk_width * temporal_factor
    num_frames = int(runtime.get("num_frames", 0))
    if num_frames < 1 or num_frames % decoded_chunk_frames:
        raise ValueError(
            "MAGI-1 sampling.num_frames must be a positive multiple of the "
            f"official decoded chunk size {decoded_chunk_frames}; got {num_frames}",
        )

    for key in ("video_size_h", "video_size_w"):
        value = int(runtime.get(key, 0))
        if value < 16 or value % 16:
            public_name = "height" if key.endswith("_h") else "width"
            raise ValueError(
                f"MAGI-1 sampling.{public_name} must be a positive multiple of 16; got {value}",
            )

    fps = int(runtime.get("fps", 0))
    if fps < 2:
        raise ValueError(
            "MAGI-1 sampling.fps must be >= 2 because the official VAE tiled "
            f"decode uses fps // 2 as its minimum tile length; got {fps}",
        )

    num_steps = int(runtime.get("num_steps", 0))
    window_size = int(runtime.get("window_size", 0))
    if window_size < 1 or num_steps < 1 or num_steps % window_size:
        raise ValueError(
            "MAGI-1 sampling.num_steps must be positive and divisible by "
            f"runtime_config.window_size={window_size}; got {num_steps}",
        )
    noise_ranges = runtime.get("noise2clean_kvrange", ())
    if not isinstance(noise_ranges, (list, tuple)):
        raise ValueError("MAGI-1 runtime_config.noise2clean_kvrange must be a sequence")
    if noise_ranges and num_steps % len(noise_ranges):
        raise ValueError(
            "MAGI-1 sampling.num_steps must be divisible by the number of "
            f"noise2clean_kvrange stages ({len(noise_ranges)}); got {num_steps}",
        )


def _validate_runtime_paths(
    config: Mapping[str, Any],
    *,
    source_path: Path,
) -> None:
    runtime = config["runtime_config"]
    for key in ("load", "t5_pretrained", "vae_pretrained"):
        value = runtime.get(key)
        if not value:
            raise ValueError(f"MAGI-1 runtime_config.{key} must be configured")
        path = _resolve_from_source(source_path, Path(str(value)))
        if not path.exists():
            raise FileNotFoundError(
                f"MAGI-1 runtime_config.{key} does not exist: {path}",
            )
        runtime[key] = str(path)

    for key in ("chunk_width", "temporal_downsample_factor"):
        value = int(runtime.get(key, 0))
        if value < 1:
            raise ValueError(f"MAGI-1 runtime_config.{key} must be >= 1")


def _resolve_weight_components(
    build: ModelBuild,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve official 4.5B/T5/VAE folders at one immutable HF revision."""

    model_config = build.model_config or {}
    explicit_checkpoint = _optional_path(model_config.get("checkpoint_path"))
    explicit_t5 = _optional_path(model_config.get("t5_pretrained_path"))
    explicit_vae = _optional_path(model_config.get("vae_pretrained_path"))
    if all(value is not None for value in (explicit_checkpoint, explicit_t5, explicit_vae)):
        return explicit_checkpoint, explicit_t5, explicit_vae

    repo_id = str(build.model_name_or_path or "").strip()
    revision = str(build.revision or "").strip()
    if not repo_id or repo_id == "None":
        raise ValueError(
            "MAGI-1 missing model.path: set an official Hugging Face repository "
            "or configure checkpoint_path, t5_pretrained_path, and vae_pretrained_path",
        )
    if not revision:
        raise ValueError(
            "MAGI-1 model.revision is required when resolving official weights from model.path",
        )

    from huggingface_hub import snapshot_download

    patterns: list[str] = []
    if explicit_checkpoint is None:
        patterns.append("ckpt/magi/4.5B_base/**")
    if explicit_t5 is None:
        patterns.append("ckpt/t5/**")
    if explicit_vae is None:
        patterns.append("ckpt/vae/**")
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=patterns,
        ),
    )
    checkpoint = explicit_checkpoint or snapshot / "ckpt" / "magi" / "4.5B_base"
    t5 = explicit_t5 or snapshot / "ckpt" / "t5"
    vae = explicit_vae or snapshot / "ckpt" / "vae"
    return checkpoint, t5, vae


def _source_head_revision(source_path: Path) -> str:
    if not (source_path / ".git").exists():
        actual_digest = runtime_source_tree_sha256(
            source_path,
            include_globs=_MAGI_1_RUNTIME_SOURCE_GLOBS,
        )
        if actual_digest != MAGI_1_RUNTIME_SOURCE_SHA256:
            raise RuntimeError(
                "MAGI-1 packaged runtime source digest mismatch: expected "
                f"{MAGI_1_RUNTIME_SOURCE_SHA256}, got {actual_digest} at {source_path}",
            )
        return MAGI_1_SUPPORTED_SOURCE_REVISION

    try:
        completed = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"could not verify MAGI-1 source revision at {source_path}: {error}",
        ) from error
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision:
        detail = completed.stderr.strip() or "not a Git checkout"
        raise RuntimeError(
            f"could not verify MAGI-1 source revision at {source_path}: {detail}",
        )
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(source_path),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"could not verify MAGI-1 source cleanliness at {source_path}: {error}",
        ) from error
    if status.returncode != 0 or status.stdout.strip():
        detail = status.stdout.strip() or status.stderr.strip()
        raise RuntimeError(
            f"MAGI-1 source checkout at {source_path} has tracked modifications: {detail}",
        )
    return revision


def _magi_mode(input_value: GenerationInput) -> tuple[str, str | None]:
    if input_value.reference_image and input_value.reference_video:
        raise ValueError("MAGI-1 accepts either reference_image or reference_video, not both")

    task_type = input_value.task_type
    if task_type in (None, ""):
        if input_value.reference_image:
            return "i2v", input_value.reference_image
        if input_value.reference_video:
            return "v2v", input_value.reference_video
        return "t2v", None

    # Canonical task_type long forms only (semantics.py owns this vocabulary):
    # video2world is image-conditioned in this repo — every v2w dataset preset
    # declares conditioning: reference_image — so it routes to i2v, not v2v.
    aliases = {
        "text_to_video": "t2v",
        "image_to_video": "i2v",
        "video2world": "i2v",
    }
    try:
        mode = aliases[task_type]
    except KeyError as error:
        raise ValueError(f"unsupported MAGI-1 task_type: {task_type!r}") from error
    conditioning_path = (
        input_value.reference_image if mode == "i2v" else input_value.reference_video
    )
    if mode != "t2v" and not conditioning_path:
        raise ValueError(f"MAGI-1 {mode} mode requires its reference input")
    if mode == "t2v" and (input_value.reference_image or input_value.reference_video):
        raise ValueError("MAGI-1 t2v task_type cannot carry a reference input")
    return mode, conditioning_path


def _generation_only_error() -> str:
    return (
        "MAGI-1 replay/training is unavailable: the official 4.5B release is a "
        "deterministic inference subprocess that exposes only a final MP4, not "
        "autograd, denoise transition densities, or replayable likelihoods"
    )


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _resolve_from_source(source_path: Path, value: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_path / path
    return path.resolve()


def _optional_source_path(source_path: Path, value: Path | None) -> Path | None:
    return None if value is None else _resolve_from_source(source_path, value)


def _optional_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(str(value))


# Canonical family surface used by the registry.  The explicit subprocess name
# remains available for callers/tests that need to distinguish this integration
# from an in-process MAGI implementation.
Magi1Model = Magi1SubprocessModel


__all__ = [
    "MAGI_1_RUNTIME_SOURCE_SHA256",
    "MAGI_1_SUPPORTED_SOURCE_REVISION",
    "Magi1Model",
    "Magi1SubprocessConfig",
    "Magi1SubprocessModel",
    "build_magi_command",
    "magi_subprocess_environment",
    "normalize_magi_1_model_build",
    "prepare_magi_runtime_config",
]

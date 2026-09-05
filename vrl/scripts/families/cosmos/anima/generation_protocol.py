"""Typed persistence boundary shared by Anima generation and evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from PIL import Image

from vrl.models.checkpoint_identity import MODEL_IDENTITY_SCHEMA
from vrl.scripts.eval.denoise_generation import GeneratorRuntimeIdentity, ImageSampling
from vrl.utils.artifacts import sha256_file

ANIMA_GENERATION_SCHEMA = "vrl.anima-generation/v1"
ANIMA_ANCHOR_MANIFEST_SCHEMA = "vrl.anima-anchor-manifest/v1"


@dataclass(frozen=True, slots=True)
class AnimaGenerationCell:
    """One content-bound image cell from a completed Anima generation grid."""

    prompt_index: int
    sample_index: int
    seed: int
    prompt: str
    prompt_metadata: dict[str, Any]
    reward_metadata: dict[str, Any]
    image_path: Path
    image_sha256: str

    @property
    def key(self) -> tuple[int, int]:
        return self.prompt_index, self.sample_index


@dataclass(frozen=True, slots=True)
class AnimaGenerationProtocol:
    """Typed projection of one generator-owned ``run_config.json``."""

    prompt_count: int
    samples_per_prompt: int
    base_seed: int
    sampling: ImageSampling
    negative_prompt: str
    # The persisted ``execution`` mapping has exactly these two keys; the
    # device string keeps its ordinal so causal audits can name the card.
    execution_device: str
    execution_dtype: str
    generator_runtime: GeneratorRuntimeIdentity
    use_lora: bool
    # Schema-derived records compared whole, never probed by key here:
    # ``model_identity`` is ``resolve_checkpoint_model_identity``'s output
    # (fields come from the family's model section) and ``generation_policy``
    # is the generator's projection of its resolved build/precision.
    model_identity: dict[str, Any]
    generation_policy: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        directory: Path,
    ) -> AnimaGenerationProtocol:
        """Parse the generator contract without applying evaluator policy."""

        if value.get("schema") != ANIMA_GENERATION_SCHEMA:
            raise ValueError(
                f"unsupported Anima generation schema in {directory}: {value.get('schema')!r}",
            )
        sampling = ImageSampling.from_mapping(
            value.get("sampling"),
            what=f"Anima generation sampling in {directory}",
        )
        negative_prompt = value.get("negative_prompt")
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"Anima generation negative_prompt in {directory} must be a string",
            )

        execution = _require_mapping(value, "execution", directory)
        for key in ("device", "dtype"):
            if not isinstance(execution.get(key), str) or not execution[key]:
                raise ValueError(
                    f"Anima generation execution.{key} in {directory} must be a non-empty string",
                )

        generator_runtime = GeneratorRuntimeIdentity.from_mapping(
            value.get("generator_runtime"),
            what=f"Anima generator runtime in {directory}",
        )

        generation_policy = _require_mapping(value, "generation_policy", directory)
        if not generation_policy:
            raise ValueError(f"Anima generation policy in {directory} must not be empty")
        model = _require_mapping(value, "model", directory)
        use_lora = model.get("use_lora")
        if not isinstance(use_lora, bool):
            raise TypeError(f"Anima generation model.use_lora in {directory} must be a bool")

        model_identity = _require_mapping(value, "model_identity", directory)
        identity_schema = model_identity.get("schema")
        sources = model_identity.get("sources")
        build = model_identity.get("build")
        if identity_schema != MODEL_IDENTITY_SCHEMA:
            raise ValueError(
                f"Anima model identity schema in {directory} must be {MODEL_IDENTITY_SCHEMA!r}",
            )
        if not isinstance(sources, Mapping) or not sources or not isinstance(build, Mapping):
            raise ValueError(
                f"Anima model identity in {directory} requires non-empty sources and build",
            )
        if build.get("use_lora") is not use_lora:
            raise ValueError(
                f"Anima model identity and model.use_lora disagree in {directory}",
            )

        return cls(
            prompt_count=_run_config_int(value, "prompt_count", directory, positive=True),
            samples_per_prompt=_run_config_int(
                value,
                "samples_per_prompt",
                directory,
                positive=True,
            ),
            base_seed=_run_config_int(value, "base_seed", directory, positive=False),
            sampling=sampling,
            negative_prompt=negative_prompt,
            execution_device=execution["device"],
            execution_dtype=execution["dtype"],
            generator_runtime=generator_runtime,
            use_lora=use_lora,
            model_identity={
                "schema": identity_schema,
                "sources": dict(sources),
                "build": dict(build),
            },
            generation_policy=dict(generation_policy),
        )

    @property
    def base_model_identity(self) -> dict[str, Any]:
        """Project the shared base identity while leaving LoRA as treatment."""

        build = self.model_identity["build"]
        return {
            "schema": self.model_identity["schema"],
            "sources": dict(self.model_identity["sources"]),
            "build": {
                key: value for key, value in build.items() if key not in {"use_lora", "lora"}
            },
        }


@dataclass(frozen=True, slots=True)
class AnimaPixelPairingProtocol:
    """Pixel-producing inputs that must match across paired generations."""

    prompt_count: int
    samples_per_prompt: int
    base_seed: int
    sampling: ImageSampling
    negative_prompt: str
    execution_device_type: str
    execution_dtype: str
    generator_python: str
    generator_packages: dict[str, str | None]
    generation_policy: dict[str, Any]
    base_model_identity: dict[str, Any]

    @classmethod
    def from_generation_protocol(
        cls,
        protocol: AnimaGenerationProtocol,
    ) -> AnimaPixelPairingProtocol:
        """Exclude placement and broad provenance that cannot change pixels."""

        runtime = protocol.generator_runtime
        # The broad tree hash includes reward and evaluator files that never
        # participate in generation. Causal audits bind it separately; paired
        # image evaluation uses the narrower pixel-producing runtime identity.
        return cls(
            prompt_count=protocol.prompt_count,
            samples_per_prompt=protocol.samples_per_prompt,
            base_seed=protocol.base_seed,
            sampling=protocol.sampling,
            negative_prompt=protocol.negative_prompt,
            # Device ordinals are placement, but the backend changes generation
            # numerics and therefore remains part of the paired-pixel protocol.
            execution_device_type=protocol.execution_device.partition(":")[0].lower(),
            execution_dtype=protocol.execution_dtype,
            generator_python=runtime.python,
            generator_packages=dict(runtime.packages),
            generation_policy=dict(protocol.generation_policy),
            base_model_identity=protocol.base_model_identity,
        )


@dataclass(frozen=True, slots=True)
class AnimaGenerationArchive:
    """Strictly verified output of one completed repo-native Anima generation."""

    directory: Path
    run_config: dict[str, Any]
    protocol: AnimaGenerationProtocol
    cells: tuple[AnimaGenerationCell, ...]
    metadata_sha256: str
    run_config_sha256: str
    anchor_manifest_sha256: str

    @classmethod
    def load(cls, directory: Path) -> AnimaGenerationArchive:
        """Load only a complete, internally consistent, content-bound archive."""

        directory = directory.expanduser().resolve()
        metadata_path = directory / "metadata.jsonl"
        run_config_path = directory / "run_config.json"
        anchor_path = directory / "anchor_manifest.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"generation metadata does not exist: {metadata_path}")
        if not anchor_path.is_file():
            raise FileNotFoundError(f"generation anchor manifest does not exist: {anchor_path}")

        run_config = _read_json_mapping(run_config_path)
        protocol = AnimaGenerationProtocol.from_mapping(run_config, directory)
        cells = _load_cells(metadata_path, directory, protocol.sampling)
        _validate_complete_grid(cells, protocol, directory)
        _validate_anchor_manifest(
            cells,
            anchor_path=anchor_path,
            directory=directory,
            expected_source=(
                "anima_lora_synthetic" if protocol.use_lora else "anima_base_synthetic"
            ),
        )
        return cls(
            directory=directory,
            run_config=run_config,
            protocol=protocol,
            cells=tuple(cells),
            metadata_sha256=sha256_file(metadata_path),
            run_config_sha256=sha256_file(run_config_path),
            anchor_manifest_sha256=sha256_file(anchor_path),
        )


def validate_paired_generation_archives(
    base_archive: AnimaGenerationArchive,
    checkpoint_archive: AnimaGenerationArchive,
) -> None:
    """Require two archives to differ only in treatment and generated pixels."""

    base_protocol = AnimaPixelPairingProtocol.from_generation_protocol(
        base_archive.protocol,
    )
    checkpoint_protocol = AnimaPixelPairingProtocol.from_generation_protocol(
        checkpoint_archive.protocol,
    )
    protocol_mismatches = [
        field.name
        for field in fields(AnimaPixelPairingProtocol)
        if getattr(base_protocol, field.name) != getattr(checkpoint_protocol, field.name)
    ]
    if protocol_mismatches:
        raise ValueError(
            f"checkpoint generation protocol differs from base: fields={protocol_mismatches}",
        )

    base_cells = {cell.key: cell for cell in base_archive.cells}
    checkpoint_cells = {cell.key: cell for cell in checkpoint_archive.cells}
    if len(base_cells) != len(base_archive.cells):
        raise ValueError(f"base archive contains duplicate cell keys: {base_archive.directory}")
    if len(checkpoint_cells) != len(checkpoint_archive.cells):
        raise ValueError(
            f"checkpoint archive contains duplicate cell keys: {checkpoint_archive.directory}",
        )
    if set(base_cells) != set(checkpoint_cells):
        missing = sorted(set(base_cells) - set(checkpoint_cells))
        extra = sorted(set(checkpoint_cells) - set(base_cells))
        raise ValueError(
            f"checkpoint cell grid differs from base: missing={missing} extra={extra}",
        )
    for key in sorted(base_cells):
        base_cell = base_cells[key]
        checkpoint_cell = checkpoint_cells[key]
        cell_mismatches = [
            name
            for name in ("prompt", "seed", "prompt_metadata", "reward_metadata")
            if getattr(base_cell, name) != getattr(checkpoint_cell, name)
        ]
        if cell_mismatches:
            raise ValueError(
                f"paired generation cell differs at {key}: fields={cell_mismatches}",
            )


def _load_cells(
    metadata_path: Path,
    directory: Path,
    sampling: ImageSampling,
) -> list[AnimaGenerationCell]:
    persisted_fields = {field.name for field in fields(AnimaGenerationCell)} - {
        "image_sha256",
    }
    cells: list[AnimaGenerationCell] = []
    seen_keys: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{metadata_path}:{line_number} must contain a JSON object")
        missing = sorted(persisted_fields - set(raw))
        unknown = sorted(set(raw) - persisted_fields)
        if missing or unknown:
            raise ValueError(
                f"invalid generation row at {metadata_path}:{line_number}: "
                f"missing={missing} unknown={unknown}",
            )
        prompt_index = _nonnegative_int(raw, "prompt_index", metadata_path, line_number)
        sample_index = _nonnegative_int(raw, "sample_index", metadata_path, line_number)
        seed = _nonnegative_int(raw, "seed", metadata_path, line_number)
        key = (prompt_index, sample_index)
        if key in seen_keys:
            raise ValueError(f"duplicate generation cell {key} in {metadata_path}")
        seen_keys.add(key)

        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{metadata_path}:{line_number} has no prompt")
        prompt_metadata = raw.get("prompt_metadata")
        reward_metadata = raw.get("reward_metadata")
        if not isinstance(prompt_metadata, Mapping):
            raise TypeError(f"{metadata_path}:{line_number} prompt_metadata must be a mapping")
        if not isinstance(reward_metadata, Mapping):
            raise TypeError(f"{metadata_path}:{line_number} reward_metadata must be a mapping")

        image_path = _resolve_archive_file(
            directory,
            raw.get("image_path"),
            metadata_path,
            line_number,
            field_name="image_path",
            relative_only=False,
        )
        if image_path in seen_paths:
            raise ValueError(f"duplicate generated image path in {metadata_path}: {image_path}")
        seen_paths.add(image_path)
        with Image.open(image_path) as image:
            image.load()
            expected_size = (sampling.width, sampling.height)
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(
                    f"{metadata_path}:{line_number} image must be an RGB PNG; "
                    f"format={image.format!r} mode={image.mode!r}",
                )
            if image.size != expected_size:
                raise ValueError(
                    f"{metadata_path}:{line_number} image dimensions differ from "
                    f"run_config.json: expected={expected_size} actual={image.size}",
                )
        cells.append(
            AnimaGenerationCell(
                prompt_index=prompt_index,
                sample_index=sample_index,
                seed=seed,
                prompt=prompt,
                prompt_metadata=dict(prompt_metadata),
                reward_metadata=dict(reward_metadata),
                image_path=image_path,
                image_sha256=sha256_file(image_path),
            ),
        )
    if not cells:
        raise ValueError(f"generation metadata is empty: {metadata_path}")
    return sorted(cells, key=lambda cell: cell.key)


def _validate_complete_grid(
    cells: Sequence[AnimaGenerationCell],
    protocol: AnimaGenerationProtocol,
    directory: Path,
) -> None:
    expected_keys = {
        (prompt_index, sample_index)
        for prompt_index in range(protocol.prompt_count)
        for sample_index in range(protocol.samples_per_prompt)
    }
    actual_keys = {cell.key for cell in cells}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"generation grid does not match run_config.json in {directory}: "
            f"missing={missing} extra={extra}",
        )

    grouped: dict[int, list[AnimaGenerationCell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.prompt_index].append(cell)
    for prompt_index, prompt_cells in sorted(grouped.items()):
        first = prompt_cells[0]
        expected_seed = protocol.base_seed + prompt_index * protocol.samples_per_prompt
        for cell in prompt_cells:
            if (
                cell.prompt != first.prompt
                or cell.prompt_metadata != first.prompt_metadata
                or cell.reward_metadata != first.reward_metadata
            ):
                raise ValueError(
                    f"generation prompt metadata differs within prompt_index="
                    f"{prompt_index} in {directory}",
                )
            if cell.seed != expected_seed:
                raise ValueError(
                    f"generation seed does not follow the registered batch formula in "
                    f"{directory}: prompt_index={prompt_index} "
                    f"seed={cell.seed} expected={expected_seed}",
                )


def _validate_anchor_manifest(
    cells: Sequence[AnimaGenerationCell],
    *,
    anchor_path: Path,
    directory: Path,
    expected_source: str,
) -> None:
    cells_by_path = {cell.image_path: cell for cell in cells}
    anchors_by_path: dict[Path, tuple[str, int, int, dict[str, Any]]] = {}
    required_fields = {"prompt", "target_image", "metadata"}
    anchor_fields = {"anchor_sample_index", "anchor_seed", "anchor_source"}
    for line_number, line in enumerate(anchor_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{anchor_path}:{line_number} must contain a JSON object")
        missing = sorted(required_fields - set(raw))
        unknown = sorted(set(raw) - required_fields)
        if missing or unknown:
            raise ValueError(
                f"invalid {ANIMA_ANCHOR_MANIFEST_SCHEMA} row at "
                f"{anchor_path}:{line_number}: missing={missing} unknown={unknown}",
            )
        prompt = raw.get("prompt")
        metadata = raw.get("metadata")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{anchor_path}:{line_number} has no prompt")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{anchor_path}:{line_number} metadata must be a mapping")
        sample_index = _nonnegative_int(
            metadata,
            "anchor_sample_index",
            anchor_path,
            line_number,
        )
        seed = _nonnegative_int(metadata, "anchor_seed", anchor_path, line_number)
        if metadata.get("anchor_source") != expected_source:
            raise ValueError(
                f"{anchor_path}:{line_number} metadata.anchor_source must be {expected_source!r}",
            )
        image_path = _resolve_archive_file(
            directory,
            raw.get("target_image"),
            anchor_path,
            line_number,
            field_name="target_image",
            relative_only=True,
        )
        if image_path in anchors_by_path:
            raise ValueError(f"duplicate anchor image path in {anchor_path}: {image_path}")
        prompt_metadata = {
            key: value for key, value in metadata.items() if key not in anchor_fields
        }
        anchors_by_path[image_path] = (prompt, sample_index, seed, prompt_metadata)

    if set(cells_by_path) != set(anchors_by_path):
        missing = sorted(str(path) for path in set(cells_by_path) - set(anchors_by_path))
        extra = sorted(str(path) for path in set(anchors_by_path) - set(cells_by_path))
        raise ValueError(
            f"anchor_manifest.jsonl does not match the generation grid in {directory}: "
            f"missing={missing} extra={extra}",
        )
    for image_path, cell in cells_by_path.items():
        prompt, sample_index, seed, prompt_metadata = anchors_by_path[image_path]
        if (
            prompt != cell.prompt
            or sample_index != cell.sample_index
            or seed != cell.seed
            or prompt_metadata != cell.prompt_metadata
        ):
            raise ValueError(
                f"anchor_manifest.jsonl differs from metadata.jsonl for {image_path}",
            )


def _resolve_archive_file(
    directory: Path,
    raw_path: Any,
    source_path: Path,
    line_number: int,
    *,
    field_name: str,
    relative_only: bool,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{source_path}:{line_number} has no {field_name}")
    candidate = Path(raw_path).expanduser()
    if relative_only and candidate.is_absolute():
        raise ValueError(f"{source_path}:{line_number} {field_name} must be a relative path")
    if not candidate.is_absolute():
        candidate = directory / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(directory):
        raise ValueError(
            f"{source_path}:{line_number} {field_name} escapes generation directory: {candidate}",
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"generated image does not exist: {candidate}")
    return candidate


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
    directory: Path,
) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"Anima generation {key} in {directory} must be a mapping")
    return result


def _nonnegative_int(
    row: Mapping[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{path}:{line_number} {key} must be a non-negative integer")
    return value


def _run_config_int(
    value: Mapping[str, Any],
    key: str,
    directory: Path,
    *,
    positive: bool,
) -> int:
    result = value.get(key)
    minimum = 1 if positive else 0
    if type(result) is not int or result < minimum:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(
            f"run_config.json {key} must be a {requirement} integer in {directory}",
        )
    return result


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


__all__ = [
    "ANIMA_ANCHOR_MANIFEST_SCHEMA",
    "ANIMA_GENERATION_SCHEMA",
    "AnimaGenerationArchive",
    "AnimaGenerationCell",
    "AnimaGenerationProtocol",
    "AnimaPixelPairingProtocol",
    "validate_paired_generation_archives",
]

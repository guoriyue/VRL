"""Deterministic exact-person-count reward backed by pinned CountGD.

CountGD is kept in an isolated reward-service environment because its upstream
source is not packaged and imports generic top-level modules such as ``models``
and ``util``.  This adapter follows the project's official single-image,
text-only protocol exactly: the ``datasets_inference`` transform, no visual
exemplars, caption ``"person ."``, and confidence threshold 0.23.

The reward target comes only from ``artifact.metadata["expected_people"]``.
Prompt parsing would make wording an accidental second source of truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.media import decode_artifact_frames
from vrl.utils.artifacts import default_data_root, sha256_file
from vrl.utils.media import to_pil_image

# These values define the qualified external-model protocol. They are real
# source/checkpoint boundaries, not a duplicate business vocabulary.
COUNTGD_SOURCE_REVISION = "b6f362b3f5cd20db4a171faa410dfed8f2f466d8"
COUNTGD_SPACE_REVISION = "6e82e59569a84ee5c6aafa35d396f2d2bee57be2"
COUNTGD_CHECKPOINT_SHA256 = "c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126"
COUNTGD_RUNTIME_TREE_SHA256 = "e41c4fd64148a0a55a4d5bda3e0f5f8da6297811d3f7648506761742ac04b450"
COUNTGD_INSTALL_SCHEMA = "vrl.countgd-install/v1"

_COUNT_CAPTION = "person ."
_CONFIDENCE_THRESHOLD = 0.23
_EXPECTED_PEOPLE_METADATA_KEY = "expected_people"
COUNTGD_PERSON_COUNT_SCORE_KEY = "countgd_person_count"
_PROTOCOL_SCHEMA = "vrl.countgd-person-count-protocol/v1"
_RESIZE_SHORT_SIDE = 800
_RESIZE_MAX_SIDE = 1333
_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
_NORMALIZATION_STD = (0.229, 0.224, 0.225)
_VISUAL_EXEMPLAR_COUNT = 0
_EXEMPLAR_POINT_INDICES = (0,)
_MODEL_INIT_SEED = 42
_MAX_COUNT_ERROR = 0


def countgd_model_protocol() -> dict[str, Any]:
    """Return the canonical semantic scoring protocol for manifests."""

    return {
        "schema": _PROTOCOL_SCHEMA,
        "caption": _COUNT_CAPTION,
        "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
        "confidence_threshold": _CONFIDENCE_THRESHOLD,
        "count_decision": "sigmoid(pred_logits).max(-1) > confidence_threshold",
        "exemplar_point_indices": list(_EXEMPLAR_POINT_INDICES),
        "expected_people_metadata_key": _EXPECTED_PEOPLE_METADATA_KEY,
        "max_count_error": _MAX_COUNT_ERROR,
        "model_init_seed": _MODEL_INIT_SEED,
        "normalization_mean": list(_NORMALIZATION_MEAN),
        "normalization_std": list(_NORMALIZATION_STD),
        "resize_max_side": _RESIZE_MAX_SIDE,
        "resize_short_side": _RESIZE_SHORT_SIDE,
        "runtime_tree_sha256": COUNTGD_RUNTIME_TREE_SHA256,
        "score_key": COUNTGD_PERSON_COUNT_SCORE_KEY,
        "source_revision": COUNTGD_SOURCE_REVISION,
        "space_revision": COUNTGD_SPACE_REVISION,
        "visual_exemplar_count": _VISUAL_EXEMPLAR_COUNT,
    }


COUNTGD_MODEL_VERSION = (
    f"CountGD@{COUNTGD_SOURCE_REVISION}+protocol-"
    + hashlib.sha256(
        json.dumps(
            countgd_model_protocol(),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    ).hexdigest()
)
_RUNTIME_TREE_SCHEMA = "vrl.countgd-runtime-tree/v1"
_RUNTIME_TREE_ALGORITHM = "sha256-length-framed-path-content-v1"
_RUNTIME_TREE_HASH_HEADER = _RUNTIME_TREE_SCHEMA.encode("ascii") + b"\0"
_RUNTIME_TREE_EXCLUDED_DIRS = frozenset({"__pycache__", ".cache", "cache", "outputs"})


@dataclass(frozen=True, slots=True)
class CountGDPersonCountConfig:
    """Resolved model-owned paths and device for one CountGD service."""

    source_dir: Path
    device: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CountGDPersonCountConfig:
        advertised_version = str(value.get("reward_model_version", "")).strip()
        if advertised_version != COUNTGD_MODEL_VERSION:
            raise ValueError(
                "CountGD reward_model_version does not match the executable protocol: "
                f"expected={COUNTGD_MODEL_VERSION!r}, actual={advertised_version!r}",
            )
        source_value = str(value.get("source_dir", "")).strip()
        source_dir = (
            Path(source_value).expanduser().resolve()
            if source_value
            else (default_data_root() / "countgd" / "source").resolve()
        )
        device = str(value.get("device", "cpu")).strip().lower()
        if not device:
            raise ValueError("CountGD person-count device must be non-empty")
        return cls(source_dir=source_dir, device=device)


@dataclass(frozen=True, slots=True)
class CountGDPersonDetection:
    """One protocol-filtered CountGD person detection."""

    bbox_cxcywh: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        bbox = tuple(float(value) for value in self.bbox_cxcywh)
        confidence = float(self.confidence)
        if len(bbox) != 4 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in bbox
        ):
            raise ValueError("CountGD bbox_cxcywh must contain four finite normalized values")
        if not math.isfinite(confidence) or not _CONFIDENCE_THRESHOLD < confidence <= 1.0:
            raise ValueError("CountGD detection confidence must pass the protocol threshold")
        object.__setattr__(self, "bbox_cxcywh", bbox)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class CountGDPersonCountObservation:
    """Typed output from one qualified CountGD image observation."""

    observed_people: int

    def __post_init__(self) -> None:
        if type(self.observed_people) is not int or self.observed_people < 0:
            raise ValueError("CountGD observed_people must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CountGDPersonCountResult:
    """One observation evaluated against the artifact's typed target."""

    expected_people: int
    observation: CountGDPersonCountObservation

    def __post_init__(self) -> None:
        if type(self.expected_people) is not int or self.expected_people < 1:
            raise ValueError("CountGD expected_people must be a positive integer")
        if not isinstance(self.observation, CountGDPersonCountObservation):
            raise TypeError("CountGD observation must be CountGDPersonCountObservation")

    @property
    def observed_people(self) -> int:
        return self.observation.observed_people

    @property
    def exact_match(self) -> bool:
        return abs(self.observed_people - self.expected_people) <= _MAX_COUNT_ERROR

    @property
    def reward(self) -> float:
        return float(self.exact_match)

    def to_scores(self) -> dict[str, float]:
        """Serialize the production reward-service score boundary."""

        return {COUNTGD_PERSON_COUNT_SCORE_KEY: self.reward}


@dataclass(slots=True)
class _CountGDRuntime:
    """Loaded upstream objects that share one lifecycle."""

    model: Any
    transform: Any
    torch: Any


@dataclass(frozen=True, slots=True)
class _RuntimeTreeDigest:
    """Content identity for the qualified executable source and local BERT."""

    sha256: str
    file_count: int


class CountGDPersonCountModel:
    """Return one iff CountGD observes exactly the typed requested person count."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.config = CountGDPersonCountConfig.from_mapping(worker_config)
        self._runtime: _CountGDRuntime | None = None

    def prepare_for_inference(self) -> None:
        """Load and verify the pinned model once inside the service process."""

        if self._runtime is None:
            self._runtime = self._load_runtime()

    def score_batch(
        self,
        artifacts: Sequence[RewardInferenceArtifact],
    ) -> list[dict[str, float]]:
        self.prepare_for_inference()
        return [self(artifact) for artifact in artifacts]

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        return self.evaluate(artifact).to_scores()

    def observe(self, artifact: RewardInferenceArtifact) -> CountGDPersonCountObservation:
        """Return the typed CountGD observation used by every scorer."""

        return CountGDPersonCountObservation(observed_people=self.count_people(artifact))

    def evaluate(self, artifact: RewardInferenceArtifact) -> CountGDPersonCountResult:
        """Apply the model-owned exact-count decision to one typed target."""

        expected_people = _expected_people(artifact)
        return CountGDPersonCountResult(
            expected_people=expected_people,
            observation=self.observe(artifact),
        )

    def count_people(self, artifact: RewardInferenceArtifact) -> int:
        """Return the count derived from the protocol-filtered detections."""

        return len(self.detect_people(artifact))

    def detect_people(
        self,
        artifact: RewardInferenceArtifact,
    ) -> tuple[CountGDPersonDetection, ...]:
        """Run the qualified protocol and retain its typed person detections."""

        self.prepare_for_inference()
        runtime = self._runtime
        assert runtime is not None

        frame = decode_artifact_frames(artifact, num_frames=1)[0]
        image = to_pil_image(frame)
        empty_exemplars = runtime.torch.empty((_VISUAL_EXEMPLAR_COUNT,))
        input_image, target = runtime.transform(
            image,
            {"exemplars": empty_exemplars},
        )
        device = self.config.device
        with runtime.torch.inference_mode():
            output = runtime.model(
                input_image.unsqueeze(0).to(device),
                [target["exemplars"].to(device)],
                [runtime.torch.tensor(_EXEMPLAR_POINT_INDICES, device=device)],
                captions=[_COUNT_CAPTION],
            )
        confidence = output["pred_logits"][0].sigmoid().max(dim=-1).values
        keep = confidence > _CONFIDENCE_THRESHOLD
        kept_boxes = output["pred_boxes"][0][keep].detach().cpu().tolist()
        kept_confidence = confidence[keep].detach().cpu().tolist()
        return tuple(
            CountGDPersonDetection(
                bbox_cxcywh=tuple(box),
                confidence=score,
            )
            for box, score in zip(kept_boxes, kept_confidence, strict=True)
        )

    def _load_runtime(self) -> _CountGDRuntime:
        source_dir = self.config.source_dir
        _verify_install(source_dir)
        _expose_isolated_upstream(source_dir)

        import datasets_inference.transforms as transforms
        import numpy as np
        import torch
        from models.registry import MODULE_BUILD_FUNCS
        from util.slconfig import SLConfig

        args = Namespace(device=self.config.device, finetune_ignore=None)
        cfg = SLConfig.fromfile(str(source_dir / "cfg_app.py"))
        cfg.merge_from_dict(
            {"text_encoder_type": str(source_dir / "checkpoints" / "bert-base-uncased")},
        )
        for key, item in cfg._cfg_dict.to_dict().items():
            setattr(args, key, item)

        torch.manual_seed(_MODEL_INIT_SEED)
        np.random.seed(_MODEL_INIT_SEED)
        random.seed(_MODEL_INIT_SEED)
        build = MODULE_BUILD_FUNCS.get(args.modelname)
        if build is None:
            raise RuntimeError(f"CountGD model builder is not registered: {args.modelname!r}")
        model, _, _ = build(args)

        checkpoint_path = source_dir / "checkpoint_best_regular.pth"
        # CountGD's trusted author checkpoint contains argparse.Namespace, so
        # weights_only=False is required. Hash verification happens first to
        # keep that pickle boundary pinned to the qualified file.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )["model"]
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "CountGD checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}",
            )
        model.to(self.config.device).eval()

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    _NORMALIZATION_MEAN,
                    _NORMALIZATION_STD,
                ),
            ],
        )
        transform = transforms.Compose(
            [
                transforms.RandomResize(
                    [_RESIZE_SHORT_SIDE],
                    max_size=_RESIZE_MAX_SIDE,
                ),
                normalize,
            ],
        )
        return _CountGDRuntime(model=model, transform=transform, torch=torch)


def _expected_people(artifact: RewardInferenceArtifact) -> int:
    value = artifact.metadata.get(_EXPECTED_PEOPLE_METADATA_KEY)
    if type(value) is not int or value < 1:
        raise ValueError(
            f"CountGD artifact {artifact.artifact_id!r} requires a positive integer "
            f"metadata[{_EXPECTED_PEOPLE_METADATA_KEY!r}]",
        )
    return value


def _verify_install(source_dir: Path) -> None:
    checkpoint_path = source_dir / "checkpoint_best_regular.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CountGD checkpoint is missing: {checkpoint_path}")
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != COUNTGD_CHECKPOINT_SHA256:
        raise ValueError(
            f"CountGD checkpoint hash mismatch for {checkpoint_path}: "
            f"expected={COUNTGD_CHECKPOINT_SHA256}, actual={actual_checkpoint_sha256}",
        )

    manifest_path = source_dir.parent / "install_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CountGD installation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError(f"CountGD installation manifest must be a mapping: {manifest_path}")
    expected_manifest = {
        "schema": COUNTGD_INSTALL_SCHEMA,
        "source_revision": COUNTGD_SOURCE_REVISION,
        "space_revision": COUNTGD_SPACE_REVISION,
        "checkpoint_sha256": COUNTGD_CHECKPOINT_SHA256,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"CountGD installation manifest mismatch: {mismatches}")

    runtime_tree = manifest.get("runtime_tree")
    if not isinstance(runtime_tree, Mapping):
        raise ValueError("CountGD installation manifest requires runtime_tree metadata")
    expected_runtime_tree = {
        "schema": _RUNTIME_TREE_SCHEMA,
        "algorithm": _RUNTIME_TREE_ALGORITHM,
        "sha256": COUNTGD_RUNTIME_TREE_SHA256,
    }
    runtime_tree_mismatches = {
        key: {"expected": expected, "actual": runtime_tree.get(key)}
        for key, expected in expected_runtime_tree.items()
        if runtime_tree.get(key) != expected
    }
    if runtime_tree_mismatches:
        raise ValueError(
            f"CountGD runtime-tree protocol mismatch: {runtime_tree_mismatches}",
        )
    expected_file_count = runtime_tree.get("file_count")
    if type(expected_file_count) is not int or expected_file_count < 1:
        raise ValueError("CountGD runtime_tree.file_count must be a positive integer")
    actual_runtime_tree = _runtime_tree_digest(source_dir)
    if (
        actual_runtime_tree.file_count != expected_file_count
        or actual_runtime_tree.sha256 != COUNTGD_RUNTIME_TREE_SHA256
    ):
        raise ValueError(
            "CountGD qualified runtime tree mismatch: "
            f"expected_files={expected_file_count}, "
            f"actual_files={actual_runtime_tree.file_count}, "
            f"expected_sha256={COUNTGD_RUNTIME_TREE_SHA256}, "
            f"actual_sha256={actual_runtime_tree.sha256}",
        )


def _runtime_tree_digest(source_dir: Path) -> _RuntimeTreeDigest:
    """Hash all Python source plus the complete local BERT runtime directory.

    Paths are relative POSIX strings sorted bytewise. Each entry is framed as
    an unsigned eight-byte path length, path bytes, unsigned eight-byte content
    length, then content bytes. Cache/output directories are excluded. The main
    CountGD checkpoint stays outside this tree because it has its own pinned
    hash and trusted-pickle boundary.
    """

    source_dir = source_dir.resolve()
    bert_dir = source_dir / "checkpoints" / "bert-base-uncased"
    if not bert_dir.is_dir():
        raise FileNotFoundError(f"CountGD local BERT directory is missing: {bert_dir}")
    if bert_dir.is_symlink():
        raise ValueError(f"CountGD local BERT directory cannot be a symlink: {bert_dir}")

    selected: dict[str, Path] = {}
    for path in source_dir.rglob("*.py"):
        _select_runtime_tree_file(source_dir, path, selected)
    for path in bert_dir.rglob("*"):
        if path.is_file():
            _select_runtime_tree_file(source_dir, path, selected)
    if not selected:
        raise ValueError(f"CountGD qualified runtime tree is empty: {source_dir}")

    digest = hashlib.sha256(_RUNTIME_TREE_HASH_HEADER)
    for relative_path in sorted(selected):
        path = selected[relative_path]
        path_bytes = relative_path.encode("utf-8")
        size_bytes = path.stat().st_size
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(size_bytes.to_bytes(8, "big"))
        bytes_read = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size_bytes:
            raise RuntimeError(
                f"CountGD runtime file changed while hashing: {path}",
            )
    return _RuntimeTreeDigest(sha256=digest.hexdigest(), file_count=len(selected))


def _select_runtime_tree_file(
    source_dir: Path,
    path: Path,
    selected: dict[str, Path],
) -> None:
    relative = path.relative_to(source_dir)
    if any(part in _RUNTIME_TREE_EXCLUDED_DIRS for part in relative.parts[:-1]):
        return
    if path.is_symlink():
        raise ValueError(f"CountGD qualified runtime files cannot be symlinks: {path}")
    if not path.is_file():
        return
    relative_text = relative.as_posix()
    existing = selected.get(relative_text)
    if existing is not None and existing != path:
        raise ValueError(f"duplicate CountGD runtime-tree path: {relative_text}")
    selected[relative_text] = path


def _expose_isolated_upstream(source_dir: Path) -> None:
    """Expose CountGD's un-packaged top-level imports inside its service only."""

    for module_name in ("datasets_inference", "groundingdino", "models", "util"):
        module = sys.modules.get(module_name)
        if module is not None and not _module_is_under(module, source_dir):
            raise RuntimeError(
                f"CountGD requires an isolated service process; top-level module "
                f"{module_name!r} is already loaded from {getattr(module, '__file__', None)!r}",
            )
    source_text = str(source_dir.resolve())
    # CountGD itself imports generic top-level packages. Merely finding this
    # path later in sys.path is insufficient: an earlier unverified ``models``
    # or ``util`` package would win before either appears in sys.modules.
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)


def _module_is_under(module: ModuleType, source_dir: Path) -> bool:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        return False
    try:
        return Path(raw_path).resolve().is_relative_to(source_dir)
    except (OSError, ValueError):
        return False


__all__ = [
    "COUNTGD_CHECKPOINT_SHA256",
    "COUNTGD_INSTALL_SCHEMA",
    "COUNTGD_MODEL_VERSION",
    "COUNTGD_PERSON_COUNT_SCORE_KEY",
    "COUNTGD_RUNTIME_TREE_SHA256",
    "COUNTGD_SOURCE_REVISION",
    "COUNTGD_SPACE_REVISION",
    "CountGDPersonCountConfig",
    "CountGDPersonCountModel",
    "CountGDPersonCountObservation",
    "CountGDPersonCountResult",
    "CountGDPersonDetection",
    "countgd_model_protocol",
]

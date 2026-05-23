"""Artifact resolution helpers for Anima's split single-file checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ANIMA_HF_REPO = "circlestone-labs/Anima"
ANIMA_TRANSFORMER_FILE = "diffusion_models/anima-preview3-base.safetensors"
ANIMA_TEXT_ENCODER_FILE = "text_encoders/qwen_3_06b_base.safetensors"
ANIMA_VAE_FILE = "vae/qwen_image_vae.safetensors"
DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen2.5-0.5B"
DEFAULT_T5_TOKENIZER = "google-t5/t5-base"
HF_ARTIFACT_PREFIX = "hf://"


def resolve_anima_paths(model_cfg: Any) -> dict[str, str]:
    """Resolve full generation artifacts from explicit component sources."""

    return {
        "transformer_path": resolve_anima_artifact_ref(
            model_cfg,
            "transformer_path",
            ANIMA_TRANSFORMER_FILE,
        ),
        "text_encoder_path": resolve_anima_artifact_ref(
            model_cfg,
            "text_encoder_path",
            ANIMA_TEXT_ENCODER_FILE,
        ),
        "vae_path": resolve_anima_artifact_ref(
            model_cfg,
            "vae_path",
            ANIMA_VAE_FILE,
        ),
        "qwen_tokenizer_path": resolve_tokenizer_source(
            model_cfg,
            "qwen_tokenizer_path",
            DEFAULT_QWEN_TOKENIZER,
        ),
        "t5_tokenizer_path": resolve_tokenizer_source(
            model_cfg,
            "t5_tokenizer_path",
            DEFAULT_T5_TOKENIZER,
        ),
    }


def resolve_anima_replay_paths(model_cfg: Any) -> dict[str, str]:
    """Resolve only the transformer required by trainer replay."""

    return {
        "transformer_path": resolve_anima_artifact_ref(
            model_cfg,
            "transformer_path",
            ANIMA_TRANSFORMER_FILE,
        ),
    }


def resolve_anima_artifact_ref(
    model_cfg: Any,
    field_name: str,
    relative_path: str,
) -> str:
    explicit = _cfg_text(model_cfg, field_name)
    if explicit:
        return _explicit_artifact_ref(explicit)

    base_ref = _cfg_text(model_cfg, "path") or ANIMA_HF_REPO
    if base_ref.startswith(HF_ARTIFACT_PREFIX):
        base_ref = base_ref[len(HF_ARTIFACT_PREFIX) :]

    base_path = Path(base_ref).expanduser()
    if _looks_like_filesystem_source(base_ref) or base_path.exists():
        return _local_file_ref(base_path / relative_path)

    return f"{HF_ARTIFACT_PREFIX}{base_ref.rstrip('/')}/{relative_path}"


def resolve_tokenizer_source(model_cfg: Any, field_name: str, default: str) -> str:
    source = _cfg_text(model_cfg, field_name) or default
    source_path = Path(source).expanduser()
    if _looks_like_filesystem_source(source) or source_path.exists():
        if not source_path.exists():
            raise FileNotFoundError(f"Anima tokenizer source does not exist: {source}")
        return str(source_path)
    return source


def materialize_anima_artifact(ref: str) -> str:
    text = str(ref).strip()
    if not text:
        raise ValueError("Anima artifact reference is empty")
    if not text.startswith(HF_ARTIFACT_PREFIX):
        return str(Path(text).expanduser())

    repo_id, filename = _split_hf_artifact_ref(text)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def is_local_tokenizer_source(source: str) -> bool:
    source_path = Path(str(source)).expanduser()
    return source_path.exists() or _looks_like_filesystem_source(str(source))


def _explicit_artifact_ref(value: str) -> str:
    if value.startswith(HF_ARTIFACT_PREFIX):
        return value
    path = Path(value).expanduser()
    if _looks_like_filesystem_source(value) or path.exists():
        return _local_file_ref(path)
    return value


def _local_file_ref(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Anima artifact does not exist: {path}")
    return str(path)


def _split_hf_artifact_ref(ref: str) -> tuple[str, str]:
    payload = ref[len(HF_ARTIFACT_PREFIX) :]
    parts = payload.split("/", 2)
    if len(parts) != 3:
        raise ValueError(
            "Anima Hugging Face artifact refs must be "
            f"{HF_ARTIFACT_PREFIX}<namespace>/<repo>/<file>",
        )
    return f"{parts[0]}/{parts[1]}", parts[2]


def _cfg_text(cfg: Any, name: str) -> str:
    value = getattr(cfg, name, None)
    return str(value or "").strip()


def _looks_like_filesystem_source(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~")) or value.endswith(".safetensors")

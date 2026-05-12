"""Remote video reward client compatible with cosmos-rl reward service."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import torch


@dataclass(slots=True)
class RemoteVideoRewardConfig:
    enqueue_url: str
    fetch_url: str
    reward_name: str
    score_key: str = "overall_reward"
    token: str = ""
    media_type: str = "video"
    timeout_s: float = 60.0
    poll_interval_s: float = 1.0
    max_wait_s: float = 600.0
    debug_dir: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class RemoteVideoRewardClient:
    """Async client for the cosmos-rl reward service."""

    def __init__(self, config: RemoteVideoRewardConfig) -> None:
        if not config.enqueue_url:
            raise ValueError("remote video reward requires enqueue_url")
        if not config.fetch_url:
            raise ValueError("remote video reward requires fetch_url")
        if not config.reward_name:
            raise ValueError("remote video reward requires reward_name")
        if not config.score_key:
            raise ValueError("remote video reward requires score_key")
        if config.media_type not in {"video", "image"}:
            raise ValueError("remote video reward media_type must be 'video' or 'image'")
        self.config = config

    async def score_batch(
        self,
        *,
        media: Any,
        prompts: list[str],
        metadata: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        batch_size = _batch_size(media)
        if batch_size != len(prompts):
            raise ValueError(
                "remote video reward prompt/media batch mismatch: "
                f"media={batch_size}, prompts={len(prompts)}",
            )

        payload_meta = self._metadata(prompts, media, metadata)
        body = _json_plus_npy(payload_meta, _media_to_numpy(media, self.config.media_type))
        timeout = aiohttp.ClientTimeout(total=float(self.config.timeout_s))
        headers = {
            "Content-Type": "application/octet-stream",
        }
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        async with aiohttp.ClientSession(timeout=timeout) as session:
            enqueue_data = await self._enqueue(session, body, headers)
            uuid = enqueue_data.get("uuid")
            if not isinstance(uuid, str) or not uuid:
                raise RuntimeError(f"remote video reward enqueue missing uuid: {enqueue_data}")
            replica_id = enqueue_data.get("replica_id")
            pull_data = await self._poll(session, uuid, replica_id, headers)

        scores = extract_scores(pull_data, self.config.score_key)
        if len(scores) != batch_size:
            raise RuntimeError(
                "remote video reward score batch mismatch: "
                f"scores={len(scores)}, expected={batch_size}",
            )
        _ensure_finite(scores, self.config.score_key)
        self._write_debug(
            {
                "enqueue": enqueue_data,
                "pull": pull_data,
                "reward_name": self.config.reward_name,
                "score_key": self.config.score_key,
                "scores": scores,
            },
        )
        return scores, pull_data

    def _metadata(
        self,
        prompts: list[str],
        media: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        out = {
            "prompts": prompts,
            "reward_fn": {self.config.reward_name: 1.0},
            "media_type": self.config.media_type,
        }
        out.update(self.config.extra_metadata)
        if self.config.media_type == "video":
            video_infos = metadata.get("video_infos")
            if video_infos is None:
                fps = metadata.get("video_fps", metadata.get("fps", 16.0))
                video_infos = [{"video_fps": float(fps)} for _ in range(_batch_size(media))]
            out["video_infos"] = video_infos
        return out

    async def _enqueue(
        self,
        session: aiohttp.ClientSession,
        body: bytes,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        async with session.post(
            self.config.enqueue_url,
            data=body,
            headers=headers,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    "remote video reward enqueue failed: "
                    f"status={resp.status}, body={text}",
                )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"remote video reward enqueue returned non-JSON: {text}") from exc

    async def _poll(
        self,
        session: aiohttp.ClientSession,
        uuid: str,
        replica_id: Any,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        poll_headers = dict(headers)
        poll_headers.pop("Content-Type", None)
        if replica_id is not None:
            poll_headers["X-Lepton-Replica-Target"] = str(replica_id)
        deadline = time.monotonic() + float(self.config.max_wait_s)
        last_text = ""
        while time.monotonic() < deadline:
            async with session.post(
                self.config.fetch_url,
                data={"uuid": uuid, "type": self.config.reward_name},
                headers=poll_headers,
            ) as resp:
                last_text = await resp.text()
                if resp.status == 200:
                    try:
                        return json.loads(last_text)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"remote video reward pull returned non-JSON: {last_text}",
                        ) from exc
                if resp.status >= 500:
                    raise RuntimeError(
                        "remote video reward pull failed: "
                        f"status={resp.status}, body={last_text}",
                    )
            await _async_sleep(float(self.config.poll_interval_s))
        raise TimeoutError(
            "remote video reward timed out waiting for result: "
            f"uuid={uuid}, reward={self.config.reward_name}, last_body={last_text}",
        )

    def _write_debug(self, row: dict[str, Any]) -> None:
        if not self.config.debug_dir:
            return
        path = Path(self.config.debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "remote_video_reward.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def extract_scores(response: dict[str, Any], score_key: str) -> list[float]:
    scores = response.get("scores")
    if not isinstance(scores, dict):
        raise KeyError(
            "remote video reward response missing dict field 'scores'; "
            f"got {type(scores).__name__}",
        )
    if score_key in scores:
        return _score_list(scores[score_key])
    keys = [key.strip() for key in score_key.split("+") if key.strip()]
    if not keys:
        raise ValueError(f"invalid remote video reward score_key: {score_key!r}")
    missing = [key for key in keys if key not in scores]
    if missing:
        raise KeyError(
            "remote video reward response missing score keys: "
            f"missing={missing}, requested={score_key!r}, available={sorted(scores)}",
        )
    parts = [_score_list(scores[key]) for key in keys]
    length = len(parts[0])
    if any(len(part) != length for part in parts):
        raise RuntimeError(f"composite score_key={score_key!r} has mismatched lengths")
    return [float(sum(part[i] for part in parts)) for i in range(length)]


def _score_list(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(v) for v in value]
    raise TypeError(f"score values must be number or list, got {type(value).__name__}")


def _ensure_finite(values: list[float], key: str) -> None:
    import math

    bad = [value for value in values if not math.isfinite(float(value))]
    if bad:
        raise ValueError(f"remote video reward score_key={key!r} returned non-finite scores: {bad}")


def _media_to_numpy(media: Any, media_type: str) -> np.ndarray:
    tensor = media if isinstance(media, torch.Tensor) else torch.as_tensor(media)
    tensor = tensor.detach().cpu()
    if media_type == "video":
        if tensor.ndim != 5:
            raise ValueError(
                "video reward expects batched video tensor [B,C,T,H,W] or [B,T,C,H,W], "
                f"got shape={tuple(tensor.shape)}",
            )
        if tensor.shape[1] not in {1, 3} and tensor.shape[2] in {1, 3}:
            tensor = tensor.permute(0, 2, 1, 3, 4)
        return (tensor.float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    if tensor.ndim != 4:
        raise ValueError(
            "image reward expects batched image tensor [B,C,H,W] or [B,H,W,C], "
            f"got shape={tuple(tensor.shape)}",
        )
    if tensor.shape[1] in {1, 3}:
        tensor = tensor.permute(0, 2, 3, 1)
    return (tensor.float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def _json_plus_npy(meta: dict[str, Any], array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return json.dumps(meta).encode("utf-8") + b"\n" + buffer.getvalue()


def _batch_size(media: Any) -> int:
    shape = getattr(media, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(media)


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


__all__ = [
    "RemoteVideoRewardClient",
    "RemoteVideoRewardConfig",
    "extract_scores",
]

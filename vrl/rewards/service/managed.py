"""Driver-launched reward service: ``reward.inference.<name>.kind=service``.

The reward function hands this scorer the same ``worker_config`` it would have
built its in-process model from. On first use the scorer writes a
``RewardServiceConfig`` YAML, spawns ``python -m vrl.rewards.service.server``
as a child process on a loopback port, waits for ``/ready``, and from then on
is an ordinary :class:`HttpRewardScorer`. ``shutdown`` terminates the child.

Why a subprocess and not a thread: the point is isolation from the trainer's
Python interpreter. A CPU reward scoring inside the driver competes with the
launch-bound replay for the GIL (measured: +61 s per epoch on SD3.5 under
continuous scheduling); a separate process has its own interpreter and event
loop, and the driver only awaits an HTTP response.

GPU time-sharing: when the registry hands over ``worker_config.sleep_offload``
(shared reward GPU topology), the child builds its model in a CuMem pool and
takes the phase lease over HTTP — the driver's ``activate``/``park_memory``
become ``POST /wake`` / ``POST /park``. The child inherits this process's
environment, so under torchrun's rank-local ``CUDA_VISIBLE_DEVICES`` the
resolved ``cuda:0`` names the same physical card in both processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.service.client import HttpRewardScorer

logger = logging.getLogger(__name__)

_STOP_GRACE_S = 30.0
_READY_POLL_S = 1.0


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _service_identity(worker_config: Mapping[str, Any], component_name: str) -> tuple[str, str]:
    """(model_name, model_version) the child advertises and the client expects.

    The public model id wins as the name (the same precedence the service CLI
    applies), the factory path is the fallback, and the component name ties a
    generic factory to the reward it serves.
    """

    factory = str(worker_config.get("model_factory", "")).strip()
    model_name = str(worker_config.get("reward_model_name", "")).strip() or factory
    if not model_name:
        raise ValueError("a managed reward service needs worker_config.model_factory")
    version = str(worker_config.get("reward_model_version", "")).strip() or model_name
    return f"{component_name}:{model_name}" if component_name else model_name, version


class ManagedRewardScorer(HttpRewardScorer):
    """An :class:`HttpRewardScorer` that owns the service process it talks to."""

    def __init__(
        self,
        deployment: RewardInferenceConfig,
        *,
        worker_config: Mapping[str, Any],
        artifact_dir: str | Path,
        component_name: str = "",
        python: str | None = None,
    ) -> None:
        if deployment.kind != "service":
            raise ValueError("ManagedRewardScorer requires inference.kind=service")
        worker_config = dict(worker_config)
        model_name, model_version = _service_identity(worker_config, component_name)
        # The in-service scorer stamps results with worker_config's version, so
        # the launched identity and the stamped version are one value.
        worker_config.setdefault("reward_model_version", model_version)
        endpoint = deployment.endpoint or f"http://127.0.0.1:{_free_loopback_port()}"
        resolved = replace(
            deployment,
            endpoint=endpoint,
            expected_model=deployment.expected_model or model_name,
            expected_model_version=deployment.expected_model_version or model_version,
        )
        super().__init__(resolved)
        self.deployment = resolved
        self.worker_config = worker_config
        self.component_name = component_name
        self.model_name = model_name
        self.model_version = model_version
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        # Config and log live beside the artifacts, i.e. in the run's output
        # tree, so a failed launch leaves its evidence with the run.
        self.state_dir = self.artifact_dir.parent
        self.python = python or sys.executable
        self._process: subprocess.Popen[bytes] | None = None
        self._start_lock = asyncio.Lock()

    # -- config ----------------------------------------------------------------

    @property
    def host_port(self) -> tuple[str, int]:
        from urllib.parse import urlparse

        parsed = urlparse(self.deployment.endpoint)
        return str(parsed.hostname), int(parsed.port or 80)

    def service_config(self) -> dict[str, Any]:
        """The ``RewardServiceConfig`` mapping the child is launched with."""

        host, port = self.host_port
        return {
            "host": host,
            "port": port,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_roots": [str(self.artifact_dir)],
            "max_concurrency": 1,
            "max_pending_requests": 16,
            "max_cached_requests": 1024,
            # Never claimed: a parking service shares its GPU by definition, a
            # dedicated-GPU one is not proven disjoint, and the CPU case is
            # inferred safe by the server itself.
            "generation_overlap_safe": False,
            "worker_config": self.worker_config,
        }

    @property
    def config_path(self) -> Path:
        return self.state_dir / f"reward_service.{self._file_tag}.yaml"

    @property
    def log_path(self) -> Path:
        return self.state_dir / f"reward_service.{self._file_tag}.log"

    @property
    def _file_tag(self) -> str:
        return self.component_name or "reward"

    # -- lifecycle -------------------------------------------------------------

    async def ensure_ready(self) -> None:
        await self._ensure_started()
        await super().ensure_ready()

    async def score_batch(self, request: Any) -> Any:
        await self._ensure_started()
        return await super().score_batch(request)

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            process = self._process
            if process is not None:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"managed reward service {self.model_name!r} exited with "
                        f"code {process.returncode}; see {self.log_path}",
                    )
                return
            await asyncio.to_thread(self._spawn)
            await self._wait_ready()

    def _spawn(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(self.service_config(), sort_keys=False))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        log = open(self.log_path, "ab")  # noqa: SIM115 - handed to the child
        try:
            self._process = subprocess.Popen(
                [
                    self.python,
                    "-m",
                    "vrl.rewards.service.server",
                    "--config",
                    str(self.config_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                # Its own session: a Ctrl-C aimed at the trainer must not kill
                # the service before the trainer's shutdown drains scoring.
                start_new_session=True,
            )
        finally:
            log.close()
        logger.info(
            "managed reward service %s: pid=%d endpoint=%s config=%s log=%s",
            self.model_name,
            self._process.pid,
            self.deployment.endpoint,
            self.config_path,
            self.log_path,
        )

    async def _wait_ready(self) -> None:
        process = self._process
        assert process is not None
        deadline = time.monotonic() + float(self.deployment.timeout_s)
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"managed reward service {self.model_name!r} exited during startup "
                    f"with code {process.returncode}; see {self.log_path}",
                )
            try:
                if await self.ready():
                    return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                self._terminate()
                raise TimeoutError(
                    f"managed reward service {self.model_name!r} was not ready within "
                    f"{self.deployment.timeout_s:.0f}s; see {self.log_path}",
                )
            await asyncio.sleep(_READY_POLL_S)

    async def shutdown(self) -> None:
        try:
            await super().shutdown()
        finally:
            await asyncio.to_thread(self._terminate)

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "managed reward service %s ignored SIGTERM for %.0fs; killing",
                    self.model_name,
                    _STOP_GRACE_S,
                )
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=_STOP_GRACE_S)
        self._process = None

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid


__all__ = ["ManagedRewardScorer"]

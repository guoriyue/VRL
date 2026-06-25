"""PhyMotion human-dynamics reward model (optional external backend).

PhyMotion recovers an SMPL human mesh from a video, retargets the motion onto a
MuJoCo humanoid, and scores three axes of physical realism (the key names match
upstream ``astrolabe.rewards.smpl_physics_score``):

* ``kinematic`` - joint angles / poses are anatomically plausible.
* ``contact``   - foot contact and center-of-mass balance hold.
* ``dynamic``   - the motion is dynamically achievable (torque-feasible).

The full pipeline needs SMPL body models and a MuJoCo runtime, which are heavy
and license-encumbered, so PhyMotion is deliberately NOT a base dependency and
NOT loaded in-process. The upstream code is vendored at
``third_party/PhyMotion`` (submodule), and ``vrl/scripts/eval/phymotion_score.py``
bridges it to a one-video -> JSON contract. This wrapper shells out to that
bridge (or any external scorer): ``worker_config.phymotion_cmd`` is a command
template with ``{video}`` and ``{output}`` slots; we run it in the PhyMotion env
and read the axes back from the JSON written to ``{output}``. This keeps
PhyMotion an explicit, opt-in integration point rather than shipping an
unvalidated SMPL+MuJoCo stack inside the base VRL env.

Public score keys: ``kinematic`` / ``contact`` / ``dynamic`` / ``overall``
(upstream's ``overall`` if present, else the mean of the three axes).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.ray.model import RewardModel
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_AXES = ("kinematic", "contact", "dynamic")
_DEFAULT_TIMEOUT_S = 1200


class PhyMotionModel(RewardModel):
    """Score one video by delegating to an external PhyMotion command."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.phymotion_cmd = str(self.worker_config.get("phymotion_cmd", "")).strip()
        if not self.phymotion_cmd:
            raise ValueError(
                "PhyMotion requires worker_config.phymotion_cmd - a command template "
                "with {video} and {output} slots that runs the external PhyMotion "
                "environment (SMPL + MuJoCo) and writes a JSON of the three axes. "
                "PhyMotion is not a base VRL dependency; stand up its env separately.",
            )
        if "{video}" not in self.phymotion_cmd or "{output}" not in self.phymotion_cmd:
            raise ValueError(
                "phymotion_cmd must contain both {video} and {output} slots",
            )
        self.timeout_s = int(self.worker_config.get("timeout_s", _DEFAULT_TIMEOUT_S))
        self.cwd = str(self.worker_config.get("cwd", "")).strip() or None
        logger.info("PhyMotion external wrapper %s", kv(cmd=self.phymotion_cmd, cwd=self.cwd))

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        del request
        video_path = str(Path(artifact.as_path()).expanduser().resolve())
        with tempfile.TemporaryDirectory(prefix="phymotion-") as tmp:
            output_path = str(Path(tmp) / "scores.json")
            command = self.phymotion_cmd.format(video=video_path, output=output_path)
            logger.info("running PhyMotion: %s", command)
            completed = subprocess.run(  # operator-supplied trusted command
                shlex.split(command),
                cwd=self.cwd,
                timeout=self.timeout_s,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"PhyMotion command failed (exit {completed.returncode}): "
                    f"{completed.stderr[-500:]}",
                )
            return _read_phymotion_scores(output_path)


def preflight_phymotion_backend() -> None:
    """No importable backend - PhyMotion runs as an external process."""

    return None


def _read_phymotion_scores(output_path: str) -> dict[str, float]:
    """Read and validate the three axes from PhyMotion's JSON output (fail-fast)."""

    path = Path(output_path)
    if not path.exists():
        raise RuntimeError(f"PhyMotion produced no output JSON at {output_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("PhyMotion output JSON must be a mapping of axis -> score")
    missing = [axis for axis in _AXES if axis not in payload]
    if missing:
        raise ValueError(
            f"PhyMotion output JSON missing axes {missing}; got keys {sorted(payload)}",
        )
    scores = {axis: float(payload[axis]) for axis in _AXES}
    # Prefer upstream's own 'overall'; fall back to the mean of the three axes.
    scores["overall"] = (
        float(payload["overall"])
        if "overall" in payload
        else sum(scores[axis] for axis in _AXES) / len(_AXES)
    )
    return scores


__all__ = ["PhyMotionModel", "preflight_phymotion_backend"]

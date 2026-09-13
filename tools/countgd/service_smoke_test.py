"""Start the CountGD reward service on the Bazel runtime tree and score through HTTP.

This is the acceptance the retired installer ran after every install: the
service advertises the executable protocol version, and a request round-trips
with a `countgd` score. Model loading takes about a minute on CPU.
"""

import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import aiohttp
from PIL import Image
from python.runfiles import runfiles

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.countgd import COUNTGD_MODEL_VERSION
from vrl.rewards.service.server import RewardService
from vrl.rewards.service.wire import request_to_wire, score_response_from_wire


class CountGDServiceSmokeTest(unittest.TestCase):
    def test_service_scores_on_the_assembled_runtime(self):
        os.environ["VRL_DATA_ROOT"] = runfiles.Create().Rlocation(
            "_main/third_party/countgd/runtime"
        )
        asyncio.run(self._round_trip())

    async def _round_trip(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as raw_root:
            root = Path(raw_root)
            image_path = root / "blank.png"
            Image.new("RGB", (64, 64), color=(127, 127, 127)).save(image_path)
            payload = image_path.read_bytes()
            config = root / "service.yaml"
            config.write_text(
                "\n".join(
                    [
                        "host: 127.0.0.1",
                        "port: 0",
                        "model_name: countgd-bazel-smoke",
                        f"model_version: {COUNTGD_MODEL_VERSION}",
                        "artifact_roots:",
                        f"  - {root}",
                        "max_concurrency: 1",
                        "worker_config:",
                        "  model_factory: vrl.rewards.models.countgd:CountGDModel",
                        f"  reward_model_version: {COUNTGD_MODEL_VERSION}",
                        "  device: cpu",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            service = RewardService.from_yaml(config)
            await service.start()
            try:
                host, port = service.address
                request = RewardInferenceRequest(
                    request_id="bazel-smoke",
                    artifacts=(
                        RewardInferenceArtifact(
                            artifact_id="blank",
                            sample_id="blank",
                            path=str(image_path),
                            prompt="one person",
                            size_bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(),
                            metadata={"object_class": "person", "expected_count": 1},
                        ),
                    ),
                )
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as session:
                    async with session.get(f"http://{host}:{port}/info") as response:
                        info = await response.json()
                        self.assertEqual(response.status, 200, info)
                        self.assertEqual(info["info"]["model_version"], COUNTGD_MODEL_VERSION)
                    async with session.post(
                        f"http://{host}:{port}/score", json=request_to_wire(request)
                    ) as response:
                        body = await response.json()
                        self.assertEqual(response.status, 200, body)
                results = score_response_from_wire(body, expected_request_id=request.request_id)
                self.assertEqual(len(results), 1)
                self.assertIn("countgd", results[0].scores)
                # A flat grey image contains no person: the exact-count reward is 0.
                self.assertEqual(results[0].scores["countgd"], 0.0)
            finally:
                await service.shutdown_async()


if __name__ == "__main__":
    unittest.main()

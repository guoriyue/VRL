"""Forward-process replay retains the exact decoded latent through decode and gather."""

from types import SimpleNamespace

import torch

from vrl.generation.bindings.full_sequence.executor import DenoiseBatchExecutorBase
from vrl.generation.bindings.full_sequence.gather import DenoiseBatchGatherer
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseLoopConfig, DenoiseSDEParams
from vrl.generation.types import GenerationRequest
from vrl.trajectory.reader import TrajectoryReader


def test_decoded_fp32_latent_survives_lower_precision_path_buffer_and_gather() -> None:
    class Scheduler:
        sigmas = torch.tensor([1.0, 0.0])

        def step(self, prediction, timestep, sample, *, return_dict):
            return (sample.float() - prediction,)

    class Model:
        def forward_step(self, state, step_index):
            return {"noise_pred": torch.full_like(state.latents, 0.123456, dtype=torch.float32)}

        def decode_latents(self, latents):
            self.decoded_latent = latents.clone()
            return latents.view(-1, 1, 1, 1).expand(-1, 3, 4, 4)

        def export_replay_tensors(self, state):
            return {
                "prompt_embeds": torch.zeros(state.latents.shape[0], 1, 4),
                "latents_clean": state.latents.detach(),
            }

        def export_batch_context(self, state):
            return {"height": 4, "width": 4}

    model = Model()
    executor = DenoiseBatchExecutorBase(model)
    state = SimpleNamespace(
        latents=torch.ones(2, 1, dtype=torch.bfloat16),
        timesteps=torch.tensor([1.0]),
        scheduler=Scheduler(),
    )
    config = DenoiseLoopConfig(
        sample_start=0,
        sample_count=2,
        seed=7,
        sde=DenoiseSDEParams(noise_level=0.7, sde_type="flow_grpo"),
        sde_window=None,
        denoise_mode="native",
    )
    loop = executor.run_denoise_steps(state=state, config=config)
    batch = executor.decode_denoise_result(
        batch=GenerationSampleBatch(0, 0, 2), config=config, denoise_result=loop
    )
    request = GenerationRequest("clean-replay", "test", "t2i", ["edit"], 2)
    output = DenoiseBatchGatherer().merge_generation_batches(request, [batch])
    reader = TrajectoryReader(output.trajectory)
    replay = reader.forward_process_replay("denoise", 0)
    torch.testing.assert_close(replay.latents_clean, model.decoded_latent, rtol=0, atol=0)
    assert replay.latents_clean.dtype == torch.float32
    final_action = reader.tensor_value("denoise", "actions")[:, -1]
    assert final_action.dtype == torch.bfloat16
    assert (final_action.float() - replay.latents_clean).abs().max() > 0

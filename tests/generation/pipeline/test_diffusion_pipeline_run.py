"""run_chunk_through_pipeline threads the executor stages identically to the
monolithic forward_chunk_plan (the T2 behavior-identical foundation)."""

from __future__ import annotations

from vrl.generation.diffusion.pipeline import run_chunk_through_pipeline


class _FakeExecutor:
    """Records stage calls + threads a nested sentinel so we can assert the
    pipeline calls the SAME methods in the SAME order with the SAME data flow as
    forward_chunk_plan (executor.py:479-495)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def build_prompt_stage_input(self, request, chunk):
        self.calls.append("build_prompt_stage_input")
        return ("prompt_input", request, chunk)

    def run_prompt_encode_stage(self, prompt_input, *, stage_durations, record_function):
        self.calls.append("run_prompt_encode_stage")
        assert callable(record_function)
        assert isinstance(stage_durations, dict)
        return ("encoded", prompt_input)

    def run_prepare_stage(self, prompt_output, *, stage_durations):
        self.calls.append("run_prepare_stage")
        return ("prepared", prompt_output)

    def run_denoise_stage(self, prepared, *, stage_durations):
        self.calls.append("run_denoise_stage")
        return ("denoised", prepared)

    def run_decode_stage(self, denoised):
        self.calls.append("run_decode_stage")
        return ("decoded", denoised)

    def apply_wire_storage_policy(self, request, chunk_result):
        self.calls.append("apply_wire_storage_policy")
        return ("wired", chunk_result)


def _monolithic(executor, request, chunk):
    """The exact forward_chunk_plan chain (executor.py:479-495), for comparison."""
    prompt_input = executor.build_prompt_stage_input(request, chunk)
    prompt_output = executor.run_prompt_encode_stage(
        prompt_input, stage_durations={}, record_function=lambda *_: None,
    )
    prepared = executor.run_prepare_stage(prompt_output, stage_durations={})
    denoised = executor.run_denoise_stage(prepared, stage_durations={})
    return executor.apply_wire_storage_policy(request, executor.run_decode_stage(denoised))


def test_pipeline_calls_the_same_stages_in_the_same_order() -> None:
    fake = _FakeExecutor()
    run_chunk_through_pipeline(fake, request="req", chunk="chunk")

    assert fake.calls == [
        "build_prompt_stage_input",
        "run_prompt_encode_stage",
        "run_prepare_stage",
        "run_denoise_stage",
        "run_decode_stage",
        "apply_wire_storage_policy",
    ]


def test_pipeline_output_is_identical_to_the_monolithic_chain() -> None:
    """Same nested sentinel out of both paths => identical stage-to-stage threading
    => the pipeline re-expression is behavior-identical (bit-exact by construction;
    the real torch.equal check is the 1-GPU integration step)."""
    pipe_out = run_chunk_through_pipeline(_FakeExecutor(), request="req", chunk="chunk")
    mono_out = _monolithic(_FakeExecutor(), request="req", chunk="chunk")

    assert pipe_out == mono_out
    # Spot-check the full thread: wire(decode(denoise(prepare(encode(prompt(req,chunk)))))).
    assert pipe_out == (
        "wired",
        ("decoded", ("denoised", ("prepared", ("encoded", ("prompt_input", "req", "chunk"))))),
    )

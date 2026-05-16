"""Tests for the shared AR decode loop driver."""

from __future__ import annotations

from typing import Any

from vrl.engine.ar import ARDecodeLoop, ARStepResult
from vrl.engine.core.types import GenerationRequest, GenerationSampleRow


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="fake_ar",
        task="ar_t2i",
        prompts=["p0"],
        samples_per_prompt=3,
    )


def _sample_row(index: int) -> GenerationSampleRow:
    return GenerationSampleRow(
        prompt_index=0,
        sample_index=index,
        prompt="p0",
        prompt_id="prompt-0",
        group_id="group-0",
        sample_id=f"sample-{index}",
        trajectory_id=f"traj-{index}",
        seed=None,
        metadata={"existing": index},
    )


class _FakeARModel:
    def __init__(self) -> None:
        self.init_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.step_calls: list[list[tuple[int, int]]] = []
        self.finalized = False

    def init_ar(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.init_calls.append((args, kwargs))
        return {"steps": []}

    def step_ar(
        self,
        state: dict[str, Any],
        sequences: list[Any],
        *,
        accepted: bool = False,
    ) -> ARStepResult:
        assert accepted is True
        rows_and_positions = [
            (int(seq.metadata["row_index"]), int(seq.position))
            for seq in sequences
        ]
        self.step_calls.append(rows_and_positions)
        state["steps"].append(rows_and_positions)
        return ARStepResult(
            sequence_ids=[str(seq.sample_id) for seq in sequences],
            positions=[int(seq.position) for seq in sequences],
            token=[row for row, _position in rows_and_positions],
            log_prob=[0.0 for _ in rows_and_positions],
            debug_counters={"last_batch_size": len(sequences)},
        )

    def finalize_ar(self, state: dict[str, Any]) -> dict[str, Any]:
        self.finalized = True
        return state


def test_ar_decode_loop_drives_family_hooks_and_batches_by_position() -> None:
    model = _FakeARModel()
    request = _request()
    rows = [_sample_row(index) for index in range(3)]

    result = ARDecodeLoop(
        request=request,
        sample_rows=rows,
        model=model,
        max_new_tokens=2,
        tokenizer_key="fake-tokenizer",
        dtype="float32",
        scheduler_batch_size=2,
        init_args=("embeds",),
        init_kwargs={"unused": "kept-for-init"},
        step_kwargs={"accepted": True, "ignored": "filtered"},
    ).run()

    assert model.init_calls == [(("embeds",), {"unused": "kept-for-init"})]
    assert model.step_calls == [
        [(0, 0), (1, 0)],
        [(2, 0)],
        [(0, 1), (1, 1)],
        [(2, 1)],
    ]
    assert model.finalized is True
    assert result.finalized == {"steps": model.step_calls}
    assert result.scheduler_batches == 4
    assert result.engine_counters["ar_decode_loop_enabled"] is True
    assert result.engine_counters["ar_scheduler_batches"] == 4
    assert result.engine_counters["last_batch_size"] == 1


def test_ar_decode_loop_requires_family_hooks() -> None:
    try:
        ARDecodeLoop(
            request=_request(),
            sample_rows=[_sample_row(0)],
            model=object(),
            max_new_tokens=1,
            tokenizer_key="fake-tokenizer",
            dtype="float32",
            scheduler_batch_size=None,
        ).run()
    except TypeError as exc:
        assert "init_ar" in str(exc)
        assert "step_ar" in str(exc)
        assert "finalize_ar" in str(exc)
    else:
        raise AssertionError("ARDecodeLoop should reject models without AR hooks")

"""Emu3 structural-constraint schedule + decode-loop tests.

The grid contract (official ``prefix_allowed_tokens_fn`` from the Emu3-Gen
model card, 0-indexed): column ``width`` of every row forces EOL, then the
EOF/EOI/EOS tail — everything else is image-vocab-only. These tests pin the
precomputed schedule/mask AND the end-to-end decode loop on a tiny real
Emu3 model (CPU, no weights).
"""

from __future__ import annotations

import pytest
import torch

from tests.models.families.emu3.fixtures import (
    TINY_EOF_IDX,
    TINY_EOI_IDX,
    TINY_EOL_IDX,
    TINY_EOS_IDX,
    TINY_IMAGE_VOCAB,
    build_tiny_emu3_model,
)
from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.emu3.model import (
    emu3_allowed_token_mask,
    emu3_forced_token_schedule,
    emu3_grid_token_num,
)
from vrl.models.families.emu3.runner import Emu3TokenRunner
from vrl.nn.modules.ar_attention_backends import build_torch_native_backend

HEIGHT, WIDTH = 2, 3
TOTAL = emu3_grid_token_num(HEIGHT, WIDTH)  # 2*(3+1) + 3 = 11


def test_grid_token_num_counts_eol_columns_and_tail() -> None:
    assert TOTAL == 11
    assert emu3_grid_token_num(64, 64) == 64 * 65 + 3
    with pytest.raises(ValueError, match="grid dims"):
        emu3_grid_token_num(0, 3)


def test_forced_token_schedule_matches_official_prefix_fn() -> None:
    forced = emu3_forced_token_schedule(HEIGHT, WIDTH, TINY_IMAGE_VOCAB)
    assert forced.shape == (TOTAL,)
    # EOL forced at column `width` of each row (positions 3 and 7).
    assert forced[3].item() == TINY_EOL_IDX
    assert forced[7].item() == TINY_EOL_IDX
    # Tail: EOF, EOI, EOS.
    assert forced[8].item() == TINY_EOF_IDX
    assert forced[9].item() == TINY_EOI_IDX
    assert forced[10].item() == TINY_EOS_IDX
    # Everything else is free.
    free_positions = [0, 1, 2, 4, 5, 6]
    assert all(forced[j].item() == -1 for j in free_positions)


def test_allowed_token_mask_free_vs_forced_rows() -> None:
    forced = emu3_forced_token_schedule(HEIGHT, WIDTH, TINY_IMAGE_VOCAB)
    allowed = emu3_allowed_token_mask(forced, TINY_IMAGE_VOCAB)
    assert allowed.shape == (TOTAL, TINY_IMAGE_VOCAB + 4)
    # Free rows: exactly the image columns.
    assert allowed[0, :TINY_IMAGE_VOCAB].all()
    assert not allowed[0, TINY_IMAGE_VOCAB:].any()
    # Forced rows: exactly one legal column.
    for position, forced_idx in ((3, TINY_EOL_IDX), (8, TINY_EOF_IDX), (10, TINY_EOS_IDX)):
        row = allowed[position]
        assert row[forced_idx]
        assert row.sum().item() == 1


def _run_tiny_decode_loop(model, batch_size: int = 2):
    request = GenerationRequest(
        request_id="test-emu3-decode",
        family="emu3",
        task="ar_t2i",
        inputs=["test prompt"],
        samples_per_prompt=batch_size,
    )
    rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="test prompt",
            group_id="group-0",
            sample_id=f"sample-{index}",
            trajectory_id=f"trajectory-{index}",
            seed=None,
            metadata={},
        )
        for index in range(batch_size)
    ]
    embed = model.language_model.get_input_embeddings()
    cond_ids = torch.tensor([[1, 5, 6, 51]] * batch_size)
    uncond_ids = torch.tensor([[1, 0, 0, 51]] * batch_size)
    return TokenAutoregressiveLoop(
        request=request,
        sample_rows=rows,
        runner=Emu3TokenRunner(
            model,
            attention_backend=build_torch_native_backend(model, family="emu3"),
        ),
        max_new_tokens=TOTAL,
        tokenizer_key="emu3",
        dtype="float32",
        scheduler_batch_size=batch_size,
        init_args=(
            embed(cond_ids),
            embed(uncond_ids),
            torch.ones(batch_size, 4, dtype=torch.long),
            torch.ones(batch_size, 4, dtype=torch.long),
        ),
        init_kwargs={
            "height": HEIGHT,
            "width": WIDTH,
            "guidance_scale": 1.5,
            "temperature": 1.0,
        },
    ).run()


def test_decode_loop_enforces_structural_schedule_end_to_end() -> None:
    """The full scheduled loop emits EOL at column w, then EOF/EOI/EOS."""

    torch.manual_seed(0)
    model = build_tiny_emu3_model()

    token_ids, logprobs = _run_tiny_decode_loop(model).finalized

    assert token_ids.shape == (2, TOTAL)
    assert logprobs.shape == (2, TOTAL)
    forced_positions = {
        3: TINY_EOL_IDX,
        7: TINY_EOL_IDX,
        8: TINY_EOF_IDX,
        9: TINY_EOI_IDX,
        10: TINY_EOS_IDX,
    }
    for position, expected in forced_positions.items():
        assert (token_ids[:, position] == expected).all(), position
        # Forced positions renormalize to a single legal token -> lp == 0.
        assert torch.allclose(
            logprobs[:, position],
            torch.zeros(2),
            atol=1e-6,
        ), position
    free_positions = [0, 1, 2, 4, 5, 6]
    assert (token_ids[:, free_positions] < TINY_IMAGE_VOCAB).all()
    assert (token_ids[:, free_positions] >= 0).all()
    assert (logprobs[:, free_positions] < 0).all()


def test_decode_loop_output_decodes_through_tiny_vq() -> None:
    torch.manual_seed(0)
    model = build_tiny_emu3_model()
    token_ids, _ = _run_tiny_decode_loop(model).finalized

    image = model.decode_image_tokens(token_ids, height=HEIGHT, width=WIDTH)

    # Tiny VQ has spatial factor 2 (channel_multiplier length 2).
    assert image.shape == (2, 3, HEIGHT * 2, WIDTH * 2)
    assert image.min().item() >= -1.0
    assert image.max().item() <= 1.0


def test_decode_image_tokens_rejects_truncated_sequences() -> None:
    model = build_tiny_emu3_model()
    with pytest.raises(ValueError, match="full structural sequence"):
        model.decode_image_tokens(
            torch.zeros(1, TOTAL - 3, dtype=torch.long),
            height=HEIGHT,
            width=WIDTH,
        )

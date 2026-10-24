# SPRINT: PT26-01 — Correct reward frame selection, then indexed decoding

Status: **planned — P0; semantic fix can start now; decoder adoption is conditional on measured decode cost.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/rewards/models/media.py::artifact_middle_frame_image` calls `decode_artifact_frames(artifact, 1)`. `vrl/utils/media.py::sample_frames` uses a single-point linspace, selecting frame zero; taking the middle of that one-frame stack still returns frame zero. `read_video_frames` decodes the full file before sampling. These are separate correctness and performance tasks.

## Deliverables

- [ ] Trace the helper's producers, image-domain reward consumers, tensor layouts and file/PIL paths. Make this helper select the actual middle frame; define the even-length choice as upper-middle `T // 2` unless a documented consumer contract requires otherwise.
- [ ] Preserve the generic evenly spaced sampler contract, including callers intentionally requesting the first frame. Commit the semantic correction independently of the decoding experiment.
- [ ] Add identifiable odd/even frame-sequence regression cases, one-frame/PIL cases and path/Tensor parity, with explicit treatment of lossy video encoding tolerances and empty/corrupt inputs.
- [ ] Measure artifact resolution, CPU decode, conversion, reward service and full collection wall time with representative materialized videos. Separate storage, codec and frame-selection costs.
- [ ] If decode is material, prototype TorchCodec indexed access with a tested imageio fallback. Check frame-count/index/timestamp semantics, variable-frame-rate inputs, seek behavior, FFmpeg/CUDA availability, color/layout and resource cleanup before selecting a backend.

## Acceptance

The semantic fix must select the intended frame for every supported artifact representation. The optimized decoder must match a full-decode **middle-frame** reference, with declared codec tolerances, preserve error behavior and improve measured p50/p95 decode plus end-to-end collection performance without higher failure rates or unacceptable memory. Reward changes from correcting first-frame selection are expected semantic changes and must be assessed separately from decoder equivalence.

## Boundaries

Touch `vrl/rewards/models/media.py`, `vrl/utils/media.py`, `tests/rewards/test_media_layout.py` and `tests/utils/test_media.py` as needed. Keep media references and reward lifecycle ownership unchanged. Do not switch every one-frame consumer, install a new backend globally or promise GPU decode is faster before measuring.

Reference: [TorchCodec video decoding](https://github.com/meta-pytorch/torchcodec#video-decoding).

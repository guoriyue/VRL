# VideoCon real GPU scoring acceptance

Date: 2026-09-12 UTC. Scope: real single-video reward inference, not combined
reward acceptance, learning quality, multi-GPU scaling or deployment throughput.

## Runtime and input

- Candidate: `/home/ubuntu/VRL-mgpu-integration`, based on `75d69be2`, plus
  vendor-class-only `get_head_mask` compatibility fix.
- Torch 2.12.0+cu130; isolated Transformers 5.13.0 overlay. Shared environment
  and dirty VideoPhy vendor source were not edited.
- Pinned model: `videophysics/videocon_physics@2b908dfc044350a1441efc785235c0dc110f14e1`.
- One L40S, BF16, default 32-frame input, no quantization or reduced frame count.
- Input: existing generated Wan video, prompt index 2/sample 0 from
  `outputs/wan_hpsv3_flash_grpo/eval_final/final/generated.jsonl`.
  Size and SHA256 were checked before scoring; this is not a new I2V sample.

## Results

Initial full forward failed in the visual abstractor: Transformers 5 removed
`get_head_mask`. Loading alone did not expose this. The compatibility fix
preserves legacy no-mask, per-head, per-layer, dtype and chunked broadcasting
semantics and only installs a missing method on the vendor base class.

The rerun exited 0 with finite probability scores:

| Measurement | Result |
| --- | --- |
| Physical commonsense | 0.439453125 |
| Semantic adherence | 0.8828125 |
| Overall | 0.6611328125 |
| Load time | 5.992183 s |
| Both scoring forwards | 3.200657 s |
| Peak allocated memory | 14,615,734,784 bytes |
| Peak reserved memory | 14,816,378,880 bytes |

Six focused compatibility/loading tests passed under Transformers 5.13.
The process terminated and fresh GPU process inventory was empty.
These scores are observations, not correctness labels or policy-quality claims.

## Artifacts

Root: `/mnt/nvme/outputs/wan22_i2v_cache`.

- `videocon_gpu_probe.py`: reproducible probe with input integrity assertions.
- `videocon_gpu_probe.log`: preserved initial forward failure.
- `videocon_gpu_probe_head_mask_fix.log`: successful run.
- `videocon_gpu_probe.json`: scores, timing, memory and input provenance.

Launch from the candidate worktree:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/nvme/hf/huggingface \
  PYTHONPATH=/mnt/nvme/venvs/transformers-5.13-overlay:/home/ubuntu/VRL-mgpu-integration \
  /home/ubuntu/VRL/.venv/bin/python \
  /mnt/nvme/outputs/wan22_i2v_cache/videocon_gpu_probe.py
```

Full Kling + VideoCon reward integration, scheduling alongside the large policy,
and the original physics-training/quality objective remain open.

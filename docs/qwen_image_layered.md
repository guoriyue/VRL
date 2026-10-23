# Qwen Image Layered

The `qwen_image_layered` family decomposes one source image into `layers` RGBA
layers with the Qwen-Image-Layered checkpoint. It is the `qwen_image` MMDiT and
four-channel video VAE loaded through the checkpoint's own
`QwenImageLayeredPipeline`; the family code lives beside `qwen_image`
(`vrl/models/families/qwen_image/layered.py`) and reuses its prompt encoding,
CFG runner and decode machinery.

Use `model/qwen_image_layered/base`. The preset pins the Hub revision, keeps
the frozen Qwen2.5-VL encoder on the host for a 32 GB card, and names the
LoRA targets of the layered release. Sampling adds `layers` (default 4, at most
8) and `input_mode` (`rgb` replaces a source PNG's alpha with opaque, `rgba`
keeps it). Every prompt row supplies exactly one `reference_images` entry, the
source image; it is resized to the request geometry.

## Token layout

The source is VAE-encoded into packed condition tokens appended after the
generated tokens; the transformer output is sliced back to the generated
prefix, so the condition is conditioning, never an action. The generated
sequence holds `layers + 1` frames of one `h x w` grid: frame 0 is the model's
copy of the source (dropped at decode, as upstream does), frames `1..layers`
are RGBA layers. Timesteps are `linspace(1, 0)` sigmas shifted by
`mu = sqrt(condition_tokens / 256)` with the checkpoint's linear time shift.

`decode_latents` returns `[B, layers, 4, H, W]` in `[0, 1]`; rollout media
and PNG artifacts keep all four channels. `encode_video_to_latents` is the
exact inverse on a `[B, C, layers + 1, H, W]` clip (frame 0 composite, then
layers), so known complete layers can be encoded as `data.sft_latents`
targets for the GRPO diffusion-loss regularizer.

## Memory

The ~20B transformer in bf16 does not fit a 32 GB card. Rollout builds fit it
with `precision.rollout.quantization.format=fp8` (or `nvfp4` on Blackwell):
the swap runs on the host before the device move and drops the bf16 masters
when base weights are never synced. A 640x640, 4-layer, 20-step decomposition
peaks at about 28 GB. Training on one such card streams the frozen transformer
blocks from the host: `actor.offload: true` (the rollout's counterpart is
`model.offload: true`) makes each block's call run over device copies of its
frozen weights
(`vrl/nn/offload/block_streaming.py`). The copies are prefetched one block ahead
on a CUDA copy stream, so the PCIe transfer of block `i + 1` overlaps the
compute of block `i` in the forward pass and in the checkpoint recompute of
the backward pass. The frozen weights are pinned as DMA sources: the loader's
private file mappings cannot be registered without the kernel copying them per
process, so they are copied once into one content-keyed file under
`/dev/shm/vrl-host-streaming/`, which the trainer and the rollout worker both
map, register and read from (one pinned physical copy, reused by later runs;
`vrl/nn/offload/host_pinning.py`). Pinning locks host RAM equal to the frozen
weights; `offload.pin_host: false` keeps them pageable instead, as does a host
that cannot pin them (a non-tmpfs or too small `offload.share_directory`, a
driver refusal), and each copy is then bounded by the host memcpy at about
17 GB/s instead of the link at about 50 GB/s. Before
prefetching, with each block copied synchronously in front of its own compute,
a 640x640 update took about 12.5 s on an RTX 5090 against ~3 s with the blocks
resident on a 48 GB card.

## Supervised flow matching

`experiment/qwen_image_layered/offline_sft_layered_scenes` trains the LoRA on
rendered scenes with full per-layer truth (`algorithm.kind: diffusion_sft`, the
offline `OfflineSFTTrainer`). `vrl/scripts/denoise/encode_layered_scenes.py`
caches each scene's condition tokens, prompt embeds, packed target frames and
token weights with the family model; `vrl/scripts/eval/layered_decompose_eval.py`
decomposes cached scenes with one adapter and scores the known-layer axes. It
takes the same `--config` and overrides as training, plus `--adapter` (omit it
for the base model), `--seeds` and `--output`; pass `data.split=heldout` to
evaluate the held-out scenes. The
experiments and the data layout are in `archive/main-20261016:docs/research/layering_scene_family/`.

## Smoke

```bash
# loading / conditioning / decomposition / replay on the real checkpoint
# (fresh adapter); there is no online recipe, so the model preset is composed
# onto the generic online one
HF_HUB_OFFLINE=1 .venv/bin/python -m vrl.scripts.generation.full_sequence_probe \
  --config recipe/online/flow_matching_grpo +model=qwen_image_layered/base \
  precision.training.dtype=bf16 precision.float32_precision=ieee \
  precision.rollout.quantization.format=fp8 rollout.n_samples_per_prompt=1 \
  sampling.width=640 sampling.height=640 sampling.num_steps=20 \
  sampling.guidance_scale=1.0 sampling.layers=4 \
  actor.optim.lr=0 trainer.output_dir=outputs/probes/qwen_image_layered \
  --reference-image <source.png> --artifact-dir outputs/probes/qwen_image_layered/probe
```

The probe prepares the batch through `QwenImageLayeredBatchExecutor`'s own
hooks, so `--reference-image` is the RGBA source it decomposes. CPU tests in
`tests/models/families/qwen_image/` cover the token layout, replay parity, the
codec round trip and the executor.

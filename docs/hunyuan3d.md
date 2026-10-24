# Hunyuan3D 2.1 (shape stage)

The `hunyuan3d` family trains the shape stage of Tencent's Hunyuan3D 2.1 (released
2025-06): a 3.3B flow-matching DiT with six MoE blocks that generates a set of
4096 VecSet latent tokens from one image, decoded by the ShapeVAE into an
occupancy field. The texture (paint) stage is not part of the family. Task token
`i23d`: one RGBA cutout in, a sequence of rendered views out, scored like video
frames.

Use `model/hunyuan3d/shape_v2_1` (`tencent/Hunyuan3D-2.1`, pinned revision; the
repo is not gated). The weights are under the Tencent Hunyuan 3D 2.1 Community
License, whose territory excludes the European Union, the United Kingdom and
South Korea; `vrl/models/families/hunyuan3d/NOTICE` carries the required notice.

## What the family does differently

- Nothing is in diffusers. The DiT, the ShapeVAE decoder, the DINOv2 conditioner
  and the image preprocessing are vendored in `upstream.py` (the modifications are
  listed in its docstring); `from_build` reads the released `config.yaml` +
  `model.fp16.ckpt` layout directly, so neither class has a `transformer_classname`.
  The ShapeVAE's point-cloud encoder is not vendored: RL never encodes a mesh.
- The latent is a token set `[B, 4096, 64]` with no spatial size. `sampling.height`
  / `width` size the rendered views and `sampling.num_frames` is their count.
- Conditioning is one RGBA cutout (the alpha channel is the object mask upstream
  recenters on), encoded by DINOv2-L into 1370 tokens. The unconditional branch is
  the all-zero embedding, so replay stores only the conditional tokens.
- Reversed clock: upstream's DiT runs `t = 0` noise to `t = 1` data and predicts
  `data - noise`. `forward_step` feeds `1 - sigma` and negates the output, which is
  the shared convention exactly; `Hunyuan3DFlowScheduler` is upstream's uniform
  grid mirrored (`sampling.num_steps` real steps ending at 0).
- Decode: the ShapeVAE answers occupancy on a dense 128-cell grid and `render.py`
  marches orthographic rays through it, shading the first zero crossing (misses are
  white). No mesh library; the frames are what the reward sees. Mesh export
  (marching cubes) is not provided.
- `torch.compile` is off: MoE routing changes per batch.
- The ShapeVAE exposes no `vae_decode` memory target (tiling and slicing are
  image-VAE knobs), so a config for this family leaves `model.memory.vae_decode`
  unset.

## Bundled experiments

- `experiment/hunyuan3d/online_grpo_shape_dino_smoke_single_gpu`: GRPO on the
  released 108 example cutouts (`dataset/hunyuan3d_example_objects`; copy
  `assets/example_images` of the upstream repository into the data root under
  `hunyuan3d/example_images`), four 256 px views per shape scored by
  `reward/target_dino_similarity` against the input image with the temporal term
  off. A shaded silhouette against an RGB photo is a coarse shape anchor, not a
  quality judge; a silhouette-IoU or multi-view reward is the natural next step.

## Tests

CPU tests in `tests/models/families/hunyuan3d/` run the real tiny vendored DiT
through `forward_step` (clock, sign, batched CFG), the replay restore, the
scheduler grid against upstream's, the renderer on a synthetic sphere, and the
released checkpoint layout end to end (tiny yaml + ckpt -> `from_build` ->
encode -> prepare -> forward -> decode); the registry matrices, the scheduler
log-prob parity fixture and the config sweep cover the family automatically.

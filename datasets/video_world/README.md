# Video World Artifact Manifests

Production manifests must come from a source-backed importer and should contain
prompt metadata plus relative reference paths only. Reference frames and source
videos live outside git under `VRL_DATA_ROOT` or the default local data root:

```text
data/external/video_world/references/
data/external/video_world/source_videos/
```

For a tiny local smoke dataset:

```bash
python -m vrl.scripts.data.populate video-world-tiny
```

Default output:

```text
data/external/video_world/manifests/tiny_train.jsonl
data/external/video_world/manifests/tiny_eval.jsonl
data/external/video_world/references/tiny_train_ref.ppm
data/external/video_world/references/tiny_eval_ref.ppm
```

If you use a custom data root:

```bash
export VRL_DATA_ROOT=/path/to/external/data
python -m vrl.scripts.data.populate video-world-tiny
```

Example production row:

```json
{"prompt":"The robot arm reaches toward the cup.","reference_image":"video_world/references/bridge_episode_000001_first.png","metadata":{"source":"bridge","source_episode":"bridge_episode_000001","conditioning":"first_frame"}}
```

Do not commit production images or videos here.

# Video World Artifact Manifests

Production manifests must come from a source-backed importer and should contain
prompt metadata plus relative reference paths only. Reference frames and target
clips live outside git under `VRL_DATA_ROOT` or the default local data root:

```text
data/external/video_world/references/
data/external/video_world/targets/
data/external/video_world/manifests/
```

`vrl/scripts/data/video_world.py` imports them from public LeRobot v3.0 DROID
datasets on the HF Hub. Each manifest set is one name; its source repo, camera,
episode count and eval split are constants in that module:

```bash
python -m vrl.scripts.data.video_world robot               # first frames, lerobot/droid_100
python -m vrl.scripts.data.video_world droid_targets       # + 33-frame target clips, lerobot/droid_100
python -m vrl.scripts.data.video_world droid_full_targets  # + target clips, lerobot/droid_1.0.1
```

`--limit N` takes fewer episodes for a smoke build; `--data-root DIR` (or
`VRL_DATA_ROOT`) moves the output. For first frames, captions come from
`meta/tasks.parquet`, per-episode first frames from `data/*.parquet`, and pixels
are decoded from the per-camera mp4; only the first data/video file is read, so
`--limit` caps episodes to that file. Target clips are located through
`meta/episodes/*.parquet`. `droid_full_targets` also writes the text-only
`droid_full_targets_t2v_{train,eval}.jsonl` view for Wan T2V.

Output of `robot`:

```text
data/external/video_world/manifests/robot_train.jsonl
data/external/video_world/manifests/robot_eval.jsonl
data/external/video_world/references/droid_<episode>_first.png
data/external/video_world/robot_report.json
```

Example produced row:

```json
{"prompt":"Put the marker in the pot","reference_image":"video_world/references/droid_000000_first.png","task_type":"video2world","metadata":{"source":"droid","source_repo":"lerobot/droid_100","source_split":"main","source_episode":"000000","source_video":"videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4","source_frame_index":0,"decode_method":"pyav_http_first_frame","conditioning":"first_frame"}}
```

Needs `pyarrow`, `av`, `huggingface_hub`, `pillow` (and `imageio` for target
clips); no `lerobot` dependency. Do not commit production images or videos here.

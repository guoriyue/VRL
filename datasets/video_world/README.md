# Video World Artifact Manifests

Production manifests must come from a source-backed importer and should contain
prompt metadata plus relative reference paths only. Reference frames live outside
git under `VRL_DATA_ROOT` or the default local data root:

```text
data/external/video_world/references/
data/external/video_world/manifests/
```

Import real first frames from a LeRobot v2.1 robotics dataset (DROID / BridgeData
V2 on the HF Hub). Captions come from `meta/tasks.parquet`, per-episode first
frames from `data/*.parquet`, and pixels are decoded from the per-camera mp4:

```bash
# DROID-100 (small, default, verified)
python -m vrl.scripts.data.populate video-world-bridge --repo-id lerobot/droid_100 --limit 50

# BridgeData V2 mirror
python -m vrl.scripts.data.populate video-world-bridge \
  --repo-id IPEC-COMMUNITY/bridge_orig_lerobot --source bridge --name bridge --limit 200
```

Pick a specific camera with `--camera observation.images.<name>` (default: first
camera in `meta/info.json`).

Default output:

```text
data/external/video_world/manifests/robot_train.jsonl
data/external/video_world/manifests/robot_eval.jsonl
data/external/video_world/references/droid_<episode>_first.png
data/external/video_world/robot_report.json
```

If you use a custom data root:

```bash
export VRL_DATA_ROOT=/path/to/external/data
python -m vrl.scripts.data.populate video-world-bridge --repo-id lerobot/droid_100
```

Example produced row:

```json
{"prompt":"Put the marker in the pot","reference_image":"video_world/references/droid_000000_first.png","task_type":"video2world","metadata":{"source":"droid","source_episode":"000000","conditioning":"first_frame"}}
```

Needs `pyarrow`, `av`, `huggingface_hub`, `pillow` (no `lerobot` dependency). v1
reads the first data/video file; `--limit` caps episodes to that file. Do not
commit production images or videos here.

# Video World Artifact Manifests

Production manifests must come from a source-backed importer and should contain
prompt metadata plus relative reference paths only. Reference frames and source
videos live outside git under `VRL_DATA_ROOT` or the default local data root:

```text
data/external/video_world/references/
data/external/video_world/source_videos/
```

Import real first frames from a LeRobot/HF robotics dataset (BridgeData V2 /
DROID style):

```bash
python -m vrl.scripts.data.populate video-world-bridge \
  --repo-id lerobot/bridge_orig \
  --limit 200
```

For DROID or another mirror, point `--repo-id`/`--source` at it; if the schema is
non-standard, set `--image-column`/`--language-column`.

Default output:

```text
data/external/video_world/manifests/bridge_train.jsonl
data/external/video_world/manifests/bridge_eval.jsonl
data/external/video_world/references/bridge_<episode>_first.png
data/external/video_world/bridge_report.json
```

If you use a custom data root:

```bash
export VRL_DATA_ROOT=/path/to/external/data
python -m vrl.scripts.data.populate video-world-bridge --repo-id lerobot/bridge_orig
```

Example production row:

```json
{"prompt":"The robot arm reaches toward the cup.","reference_image":"video_world/references/bridge_000001_first.png","metadata":{"source":"bridge","source_episode":"000001","conditioning":"first_frame"}}
```

Do not commit production images or videos here.

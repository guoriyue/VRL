# Video World Artifact Manifests

Production manifests under this directory must come from a source-backed
importer and should contain prompt metadata plus relative artifact paths only.
Reference frames and source videos live under `VRL_DATA_ROOT`, normally outside
the repository:

```text
${VRL_DATA_ROOT}/video_world/references/
${VRL_DATA_ROOT}/video_world/source_videos/
```

Example row:

```json
{"prompt":"The robot arm reaches toward the cup.","reference_image":"video_world/references/bridge_episode_000001_first.png","metadata":{"source":"bridge","source_episode":"bridge_episode_000001","conditioning":"first_frame"}}
```

Do not commit production images or videos here. Tiny test artifacts belong under
`tests/fixtures/**`.

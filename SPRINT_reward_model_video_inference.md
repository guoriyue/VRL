# SPRINT：Reward-model video inference infrastructure

## 结论

这个 sprint 的主体不是 Cosmos-Predict2.5，也不是 DiffusionNFT。

真正要补的是一层可复用的 reward-model inference infrastructure：

```text
rollout output image/video
  -> stable reward artifact
  -> reward-model backend inference
  -> normalized reward result
  -> RL reward score
  -> debug/audit records
```

Cosmos-Predict2.5 + DiffusionNFT 只作为第一条真实验收 consumer。不要继续把这件事写成 Cosmos model architecture sprint，否则会让人误解成 Cosmos runtime 或 DiffusionNFT objective 还没有接好。

## 当前事实

已经存在：

```text
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

但当前 `VideoReward` 还是 reward entrypoint + backend selection + media stack + remote client glue 混在一起：

```text
Rollout -> VideoReward -> RemoteVideoRewardClient -> float score
```

这对短期 plumbing 可以，但不适合作为长期 reward-model inference 层，原因是：

- 没有 first-class artifact contract。
- 没有统一 `RewardModelRequest` / `RewardModelResult`。
- local reward model inference 还没有 runtime 边界。
- remote reward debug 只记录 raw service response，不够表达 artifact、model version、latency、score breakdown。
- Cosmos recipe 容易被误写成“真实 video reward 已验证”，但当前 recipe 可以退回 OCR/simple reward。

## 目标

建立一个独立 reward-model inference 层，让 image/video reward model 能被所有 visual RL recipe 复用。

目标接口：

```text
RewardArtifact
RewardArtifactStore
RewardModelRequest
RewardModelResult
RewardModelBackend
VideoReward adapter
```

目标能力：

- reward backend 能看见稳定 artifact，而不是临时 tensor shape。
- remote service、local model inference、stub test backend 共享同一套 request/result schema。
- raw request、raw response、score breakdown、latency、artifact path 都能落盘。
- trainer 仍然只通过现有 `RewardFunction.score_batch(...)` 消费 float reward。
- reward infra 支持 image 和 video，不写死 Cosmos。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为真实 video reward 验收，但不是这个 infra 的唯一目标。

## 不做的事

- 不把 `dance_grpo` / `cosmos_reason1` 写死进 Cosmos trainer。
- 不把 reward model 加载进训练 policy module。
- 不把 remote reward service 当成本地必需依赖。
- 不用 `backend: stub` 证明真实 reward model 已完成。
- 不把 OCR/aesthetic-only 当成 future video reward-model 验收。
- 不做 supervised V2W / SFT / reconstruction loss。
- 不把 generated video artifact 只放在 transient tensor 里。

## 设计

### 1. RewardArtifact

新增 artifact contract，表达 reward model 实际看到的媒体：

```text
vrl/rewards/model_inference/types.py
```

建议字段：

```text
artifact_id: str
media_type: "image" | "video"
prompt: str
sample_id: str
seed: int | None
path: str | None
tensor_ref: str | None
shape: tuple[int, ...] | None
fps: float | None
num_frames: int | None
model_family: str
task: str
metadata: dict[str, object]
```

原则：

- `artifact_id` 是 reward request、debug、trainer metric 的 join key。
- `path` 优先用于 audit 和 remote/local worker 解耦。
- `tensor_ref` 只允许作为短期 in-process path，不作为唯一事实源。
- video 必须带 `fps` / `num_frames` 或显式 unknown。

### 2. RewardArtifactStore

新增 artifact store，把 rollout 输出稳定保存：

```text
vrl/rewards/model_inference/artifacts.py
```

建议输出：

```text
outputs/<run>/reward_artifacts/manifest.jsonl
outputs/<run>/reward_artifacts/videos/*.mp4
outputs/<run>/reward_artifacts/images/*.png
outputs/<run>/reward_artifacts/tensors/*.npy
```

manifest 每行至少包含：

```text
artifact_id
prompt
media_type
path
sample_id
seed
fps
num_frames
model_family
task
reward_scores
metadata
```

原则：

- reward model inference 前先 materialize artifact。
- debug 不依赖训练进程里的 live tensor。
- 失败时也要能从 artifact manifest 复现 reward call。

### 3. RewardModelRequest / RewardModelResult

新增统一 request/result schema：

```text
vrl/rewards/model_inference/schema.py
```

`RewardModelRequest`：

```text
request_id: str
reward_name: str
score_keys: tuple[str, ...]
artifacts: tuple[RewardArtifact, ...]
backend: str
timeout_s: float
metadata: dict[str, object]
```

`RewardModelResult`：

```text
request_id: str
artifact_id: str
reward_name: str
scores: dict[str, float]
selected_score: float
backend: str
reward_model_version: str | None
latency_ms: float | None
raw_response: dict[str, object] | None
error: str | None
```

原则：

- backend 可以返回多个 score key，adapter 决定最终 `selected_score`。
- score schema 要 fail-fast：缺 key、NaN、长度不匹配都不能静默变 0。
- raw response 是 audit payload，不参与训练主语义。

### 4. RewardModelBackend

新增 backend protocol：

```text
vrl/rewards/model_inference/backends/base.py
```

接口：

```python
class RewardModelBackend(Protocol):
    async def score_batch(
        self,
        request: RewardModelRequest,
    ) -> list[RewardModelResult]: ...
```

实现三类 backend：

```text
vrl/rewards/model_inference/backends/remote.py
vrl/rewards/model_inference/backends/local.py
vrl/rewards/model_inference/backends/stub.py
```

要求：

- `remote` 封装当前 `RemoteVideoRewardClient` 语义，但输入输出改成 request/result。
- `local` 只定义 runtime 边界，第一版可以 fail-fast，直到接入明确的 reward model wrapper。
- `stub` 只允许 tests / plumbing，并且 config 中必须显式 `allow_stub: true`。

### 5. Remote backend

remote backend 负责：

- 把 `RewardArtifact` 转成 service payload。
- 支持 path-based upload 或 tensor/npy fallback。
- 记录 enqueue / fetch / poll latency。
- 保存 raw request / raw response。
- 校验 batch size 和 score keys。

输出：

```text
outputs/<run>/reward_debug/reward_model_requests.jsonl
outputs/<run>/reward_debug/reward_model_results.jsonl
outputs/<run>/reward_debug/remote_raw.jsonl
```

第一版继续兼容 cosmos-rl style endpoint：

```text
enqueue_url
fetch_url
reward_name
score_key
video_infos
```

但不要把 cosmos-rl response shape 写进上层 `VideoReward`。

### 6. Local backend

local backend 是未来大块，必须先设计边界再接模型。

目标：

```text
RewardModelRuntime
  load reward model
  score artifacts
  release resources
```

建议文件：

```text
vrl/rewards/model_inference/backends/local.py
vrl/rewards/model_inference/runtime.py
```

设计原则：

- local reward model 不和 policy model 混在同一个 module。
- 允许独立 device，例如 `reward.device=cuda:1`。
- 后续可以迁到 Ray actor 或 subprocess worker。
- local wrapper 必须声明 `reward_model_version`。
- local backend 不存在时 fail-fast，不能自动退回 stub。

后续 candidate：

```text
dance_grpo local wrapper
cosmos_reason1 local wrapper
video-caption / VLM judge wrapper
```

### 7. VideoReward adapter

`VideoReward` 长期只做薄 adapter：

```text
Rollout -> RewardArtifactStore -> RewardModelRequest -> RewardModelBackend -> float scores
```

建议文件：

```text
vrl/rewards/video_reward.py
```

要求：

- 保留 `RewardFunction.score_batch(...)` API。
- 不在 `VideoReward` 里写 backend-specific protocol。
- `VideoReward` 从 config 选择 backend，并把 rollout metadata 转成 artifact metadata。
- `VideoReward.last_results` 或 debug writer 可用于 trainer 记录 component score breakdown。

### 8. Config

建议替换当前 `configs/base/reward/video_reward.yaml` 为更明确的结构：

```yaml
reward:
  components:
    video_reward: 1.0
  kwargs:
    video_reward:
      backend: remote
      allow_stub: false
      reward_name: cosmos_reason1
      score_key: overall_reward
      media_type: video
      artifact_dir: outputs/reward_artifacts
      debug_dir: outputs/reward_debug
      remote:
        enqueue_url: ${oc.env:REMOTE_REWARD_ENQUEUE_URL,""}
        fetch_url: ${oc.env:REMOTE_REWARD_FETCH_URL,""}
        token: ${oc.env:REMOTE_REWARD_TOKEN,""}
      local:
        model_path: ""
        device: cuda:1
        dtype: bf16
```

原则：

- `stub` 不再是 default。
- `allow_stub: true` 只能出现在 tests/smoke config。
- remote/local backend config 分层，避免互相污染。

## 实施阶段

### Phase 1：schema + artifact store

编辑：

```text
vrl/rewards/model_inference/types.py
vrl/rewards/model_inference/schema.py
vrl/rewards/model_inference/artifacts.py
tests/rewards/test_reward_artifacts.py
tests/rewards/test_reward_model_schema.py
```

完成标准：

- image/video tensor 能 materialize 成 artifact。
- manifest 包含 artifact id、prompt、sample id、fps、path。
- bad media shape fail-fast。

### Phase 2：backend protocol + stub backend

编辑：

```text
vrl/rewards/model_inference/backends/base.py
vrl/rewards/model_inference/backends/stub.py
tests/rewards/test_reward_model_backend.py
```

完成标准：

- backend 输入输出都是 request/result。
- stub backend 必须显式 `allow_stub: true`。
- score key 缺失、NaN、长度不匹配 fail-fast。

### Phase 3：remote backend migration

编辑：

```text
vrl/rewards/model_inference/backends/remote.py
vrl/rewards/remote_video.py
tests/rewards/test_remote_reward_model_backend.py
```

完成标准：

- 兼容当前 cosmos-rl remote reward service。
- raw enqueue / fetch response 进入 debug JSONL。
- `RewardModelResult` 包含 selected score、raw response、latency。
- remote backend 不再把 service response shape 泄漏到 `VideoReward`。

### Phase 4：VideoReward adapter

编辑：

```text
vrl/rewards/video_reward.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

完成标准：

- `VideoReward.score_batch(...)` 通过 artifact/request/backend/result 链路。
- `MultiReward` 不需要改调用方式。
- training recipe 仍然只看到 float reward。
- debug artifact 和 raw reward result 都能落盘。

### Phase 5：local backend boundary

编辑：

```text
vrl/rewards/model_inference/backends/local.py
vrl/rewards/model_inference/runtime.py
tests/rewards/test_local_reward_model_backend.py
```

完成标准：

- local backend 配置存在但没有 wrapper 时 fail-fast。
- local runtime 允许声明 device、dtype、model path、version。
- 不允许自动退回 remote 或 stub。

### Phase 6：Cosmos DiffusionNFT validation consumer

编辑：

```text
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
vrl/scripts/cosmos/train.py
tests/config/test_load_all_experiments.py
```

完成标准：

- Cosmos recipe 可以选择 `video_reward` backend，但不把 backend 逻辑写进 Cosmos trainer。
- 真实 reward run 至少完成一个 optimizer step。
- 记录：

```text
global_step > 0
grad_norm > 0
trainable_sha256_changed = true
reward_std > 0 or nonzero advantage batch
```

- 生成：

```text
outputs/<run>/optimization_check.json
outputs/<run>/reward_artifacts/manifest.jsonl
outputs/<run>/reward_debug/reward_model_requests.jsonl
outputs/<run>/reward_debug/reward_model_results.jsonl
```

## 验收命令

基础 infra：

```bash
pytest tests/rewards/test_reward_artifacts.py \
  tests/rewards/test_reward_model_schema.py \
  tests/rewards/test_reward_model_backend.py \
  tests/rewards/test_remote_reward_model_backend.py \
  tests/rewards/test_video_reward.py
```

配置：

```bash
pytest tests/config/test_load_all_experiments.py
```

Cosmos consumer：

```bash
python -m vrl.scripts.train --config experiment/cosmos_predict2_5_2b_diffusionnft
```

## 完成标准

- reward-model inference 有独立 schema、artifact store、backend protocol。
- `VideoReward` 是 thin adapter，不再混 backend protocol 细节。
- remote/local/stub backend 边界清楚。
- stub 不能作为真实验收 backend。
- reward artifact 和 reward debug result 可复现一次 reward call。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为 consumer 跑真实 video reward optimizer update。
- README 不能把 Cosmos-Predict2.5 DiffusionNFT 写成 validated route，除非真实 reward-model backend run 通过。

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/video_reward.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/remote_video.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/multi.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/reward/video_reward.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml`

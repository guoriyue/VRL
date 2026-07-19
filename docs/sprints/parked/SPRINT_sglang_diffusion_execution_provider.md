# SPRINT：SGLang-Diffusion official RL execution provider

日期：2026-07-13

状态：**parked**。触发事件：FlashDreams/Self-Forcing 已证明 native
`GenerationChunkExecutor + trajectory` 能承载第一个外部 provider，且 bounded Ray
operation-deadline gate 完成。按 program 顺序它是第二个外部实现。

## 0. 结论先行

先使用 SGLang-Diffusion 官方 RL API，不先 fork。它已有 rollout log-prob、denoising
environment、DiT trajectory 与 `POST /rollout/generate`；wm-infra 写 process-backed
full-chunk executor，把 response 转成 native trajectory。

```text
wm-infra GenerationRuntime
  -> GenerationWorkerCore inside each Ray worker
     -> SGLang chunk executor owns one child server process
        -> official /rollout/generate
        -> official weight API behind strict drain
```

child 不是 driver 旁路 service，也不是独立于 Ray worker 的第二个 fleet。它由构造
executor 的 `GenerationWorkerCore` 持有、监控并终止；native runtime 仍拥有 admission、
policy version、terminal failure handoff 与 resource lease。

首个 pilot 是官方已有 rollout mixin 的 Qwen-Image T2I。request schema 有 `num_frames`
不证明 RL video pipeline 已实现；Wan/video 必须等真实 producer 和 parity。

## 1. 已核对的 upstream 事实

本地审阅基线：`sgl-project/sglang@492883c8ca66aad38ee2a8912f31ce98708a2e27`。

- `1b7c33a5b751dac6187367d798a7b80bd12ccaaf` / PR #21204：diffusion
  SDE/CPS/ODE rollout log-prob。
- `47ac830c07eedfe0b1b7e36e568de0c6fdd73600` / PR #22604：standalone
  `/rollout/generate`、denoising environment、`[T+1]` latent trajectory。
- `RolloutTrajectoryData` 包含 rollout log-probs、denoising env 与
  `RolloutDitTrajectory(latents, timesteps)`。
- latents 是 `[B, T+1, ...]`：相邻 tensor 构成 T 个 transition。
- tensor 先序列化为 safetensors bytes，再 base64 放进 JSON；适合 correctness pilot，
  不是高吞吐 binary transport。
- 当前找到 rollout-specific mixin 的是 Qwen-Image 与 Z-Image。
- official `compute_weights_checksum()` 已对排序后的 parameter name + tensor bytes 做 SHA-256，
  `/get_weights_checksum` 可按 module 返回这个 named-state checksum；真正缺的是独立可验证的
  expected-module completeness、完整 name/shape/dtype schema、wm-infra policy transaction
  与 version slots。
- response 不返回 source commit/API schema identity；`/server_info` 的 package version
  也不足以单独证明运行进程与 adapter pin 一致。

## 2. Ownership

### SGLang owns

- model/pipeline loading、diffusion execution、sequence/context parallel；
- scheduler transition、local rollout log-prob、denoising environment；
- engine-local cache flush 与 model checksum。

### wm-infra owns

- immutable source/image pin、child launch/readiness/deadline/termination；
- request id、seed、admission/drain、policy version 与 supervisor handoff；
- response → native trajectory/replay mapping；
- native evaluator replay 与 old/fresh drift；
- all-worker transactional weight commit；
- reward、trainer、readiness 与 provider selection。

SGLang log-prob 是 conformance evidence，不是第二份 trainer schema。trainer 只消费 native
trajectory/evaluator。

## 3. Provider selection 与一处 provenance

当前 `ModelFamilyEntry` 只有一个默认 `executor_cls`，并通过 `family_build` 保存 family
构建语义；registry 里还没有 provider binding。不要复制第二个 Qwen family，也不要把
SGLang pin 复制到所有支持 family。

S0 新增 program 定义的 typed provider binding：

```text
provider provenance (constructed once)
  source/image digest
  adapter schema + wire schema hash
  process builder
  family bindings
    qwen_image -> executor/runtime builder + proven capabilities
```

launch 前显式解析 `family entry + provider choice`：默认没有 provider choice 时保持 native；
选择 `sglang_diffusion` 才派生 SGLang executor。不存在的 family binding、不可验证 pin、
未实现 capability 在启动前 fail closed。family registry 不复制 provenance，tests 从 binding
发现 case，不维护 `SGLANG_SUPPORTED_MODELS`。

## 4. 实施阶段

### S0 — Immutable pin + verifiable handshake

1. implementation start 固定 upstream commit 或 immutable image digest；不用 `latest`。
2. 为 request/response Pydantic schema 生成 canonical JSON schema hash，为 tensor codec、
   `[T+1]` latents/timesteps 写 CPU fixtures。
3. 启动握手必须来自**正在运行的 child**，不能回显 launch config。最小证据包含：
   - child PID/start identity 与实际 executable/package root；
   - immutable build/source or image digest；
   - running package version；
   - rollout request/response schema hash；
   - model/checkpoint identity、supported rollout fields/capabilities；
   - visible device identity。
4. activation 把 handshake 与 resolved launch contract 逐字段比较；source/schema/model/device
   任一不一致立即终止 child 并 raise。schema/digest 字段有这个行为 consumer，不是
   log-only provenance。

实现优先级：先使用 official `/server_info` + running OpenAPI/schema evidence + immutable
container inspection。如果 official server 无法证明 commit/schema，则提交最小 upstream
handshake；必要时用同进程 startup hook 暴露 wrapper-owned endpoint。由 driver/config
单独提供的 sidecar JSON 不算证据。无法验证 running process identity 时 provider 只能
保持 experimental，不能标记 Runnable。

不在当前有用户改动的 `/home/mingfeiguo/Desktop/sglang` checkout 改 remote、branch 或
patch。

### S1 — Worker-owned process adapter

首版使用 loopback/worker-private HTTP boundary：

```text
activate  -> executor launches pinned child; wait for typed handshake
generate  -> one bounded full-chunk request with deadline
offload   -> drain; dedicated GPU may remain resident
             shared GPU requires verified sleep/offload or terminate child
shutdown  -> terminate process group, join, close port, release worker resources
```

child 只继承 Ray worker 分配的 visible GPU。handshake device 与 placement 不一致 fail
closed；shared-GPU offload 不能退化为 resident no-op。adapter/serializer/process launcher
即使薄也保留，因为它们分别是 protocol、wire 与 lifecycle boundary。

CPU/fake-server tests：startup/request timeout、malformed handshake/response、child death、
shutdown 与 in-flight race、repeated shutdown、port/process-group cleanup、admission closed
后零新 request。

### S2 — Full-chunk trajectory mapping

```text
latents[:, :-1]       -> observations
latents[:, 1:]        -> actions
rollout_log_probs     -> old_log_prob
timesteps             -> timesteps
denoising_env         -> typed replay inputs/context
generated_output      -> reward artifact
```

验证 batch/sample/step 轴、T log-probs 对齐 T+1 latents、guidance/conditioning 每个保存字段
都有 trainer consumer。完整 HTTP response、headers、base64 string 不进入 trajectory；
inference time/peak memory 若只展示，在字段定义处标注 provenance-only。

### S3 — Strict version/weight transaction

首版只支持 strict/draining：

```text
close native admission
-> drain every child request
-> update every provider worker
-> compare exact module/schema evidence + one named-state checksum ACK
-> commit native policy version
-> reopen admission
```

不要为同一份权重发明 `state digest`、`provider checksum`、`trainer checksum` 三个名字。
transaction 只有三种正交事实：

1. `policy_version`：native transaction label；
2. `key_schema_hash`：expected module set + parameter names/shapes/dtypes，证明 completeness；
3. `state_checksum`：双方对**同一 exact module set**按 SGLang 的 sorted name+bytes SHA-256
   算法计算并逐 module 比较，证明值一致。

wm-infra 从实际 trainable payload/target-module resolution 派生 expected set，先拒绝
`not_found`、额外/缺失 module 或 schema mismatch，再比较 checksum。若 official API 无法
返回 observed parameter schema，先补最小 upstream/wrapper evidence；不能让 checksum
替代 completeness。

official disk update 可用于 pilot，但不是高频终局。timeout/rejection/mismatch terminates
the provider runtime；wrapper 只用 committed version stamp response；update 与 generate 不
并发。只有 request 能绑定具体 version 且 old slot 仍能执行时，才新增 non-draining。

### S4 — Qwen-Image GPU pilot

GPU 可用后按顺序：repeatability；one-request artifact + trajectory；native validation；
native replay drift；strict weight update 后再 collect；最后比较 native/SGLang quality、
throughput 与 peak memory。无需 bitwise identical output，但 transition、schedule、
conditioning、ratio≈1 必须闭合。性能报告单独扣除 base64/wire overhead。

本稿创建时 GPU 被占用，未运行或声称任何 GPU gate 通过。

### S5 — Transport decision

用 S4 测出的 trajectory bytes、serialization time、request wall time 决定：占比可忽略则
保持 official API；显著则优先 upstream binary local/shared-memory transport；只有 upstream
不接受且 production 被阻塞才创建最小 tracking fork。不复制 slime 的长期大 patch。

### S6 — Wan/video gate

只有 upstream 有真实 video rollout producer、trajectory/conditioning layout 可被 native
Wan replay 消费、parallel gather parity 与 strict transaction 全绿后才启用。否则 capability
保持 T2I-only。

## 5. Architecture hygiene

### 应改变

- 增加 worker-owned process executor，因为 HTTP/child lifecycle 是真实 framework boundary。
- response 在一个 adapter 边界转换为 native typed trajectory。
- provider provenance 构造一次，family/provider binding 与 launch validation 从它派生。

### 保持不变

- native runtime、collector、trajectory/evaluator 与 family registry 继续是 source of truth。
- SGLang typed Pydantic schema 直接消费；不在 wm-infra 手抄字段清单。
- native 继续默认；provider 未被选择时 import/config/runtime 不依赖 SGLang。

### ALL_CAPS / thin-file rules

- 不新增手写 model/capability 表；capability 来自 binding + running handshake + tests。
- 不复制 upstream `_VALID_ROLLOUT_SDE_TYPES`；使用 upstream typed validation。
- endpoint、wire/schema hash、环境变量名可做常量，因为它们是真实 protocol boundary。
- T2I/video taxonomy 不混进 process workflow code。

## 6. Definition of Done

- [ ] source/image、running process、wire schema 与 device identity 可在 activation 验证。
- [ ] explicit provider choice 在 launch 前解析，native default 与单一 provenance source 保持。
- [ ] fake-server tests 覆盖 deadline、death、malformed payload、race 与 cleanup。
- [ ] Qwen response 无损映射 native trajectory，T+1/T 对齐。
- [ ] native replay drift 在真实 checkpoint 通过。
- [ ] strict update 仅在全 worker exact schema + single state-checksum ACK 后提交 version。
- [ ] base64/HTTP overhead 已测，transport 决策有数据。
- [ ] non-draining/video capability 无法被误选。
- [ ] native diffusion/AR path 无 SGLang dependency 继续运行。

## 7. KILL / rollback

- 无法验证 running process source/schema：保持 experimental，不进 Runnable recipe。
- 缺 trainer replay conditioning：inference-only，先提 upstream change，不从 image 反推。
- strict transaction 无法证明全 worker 同版本：禁止训练。
- Qwen T2I parity 未过：不启动 Wan/video。
- transport 不可接受且无小型 upstream 修复：默认关闭，不改 native schema迁就。
- upstream 已满足时不 fork；“像 slime 一样”不是 fork 理由。

## 参考

- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/entrypoints/post_training/rollout_api.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/entrypoints/post_training/io_struct.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/entrypoints/post_training/utils.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/entrypoints/post_training/weights_api.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/loader/weight_utils.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/post_training/rl_dataclasses.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/post_training/scheduler_rl_mixin.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- `docs/sprints/SPRINT_native_generation_engine_program.md`
- `docs/sprints/SPRINT_ray_rollout_operation_deadlines.md`
- `docs/sprints/parked/SPRINT_multi_engine_rollout_conformance.md`
- https://github.com/sgl-project/sglang/pull/21204
- https://github.com/sgl-project/sglang/pull/22604

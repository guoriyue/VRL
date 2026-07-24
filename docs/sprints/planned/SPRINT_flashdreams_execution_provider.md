# SPRINT：FlashDreams controlled fork + execution provider primitives

日期：2026-07-13

Status: **planned**. This is the native-engine program's next implementation
sprint. F0-F2 may proceed in the isolated fork. The completed
[worker process-health sprint](../done/SPRINT_rollout_worker_liveness.md) covers
actor-process reachability only; its health concurrency group does not prove
default-group business-call progress. The configured blocking-call deadline for
F3 runtime integration remains an unfinished, independent gate.

## 0. 结论先行

FlashDreams 值得接，但它只进入 wm-infra native engine 的模型执行层。它提供 causal
streaming diffusion、temporal AR cache 与 world-model stack；wm-infra 继续拥有 Ray
runtime、request scheduling、policy version、trajectory/log-prob/replay、reward 与 trainer。

```text
wm-infra GenerationRuntime / GenerationWorkerCore
  -> family GenerationChunkExecutor
     -> FlashDreams step adapter
        -> FlashDreams model + cache primitives
```

当前 public API 把 denoise loop 放在 `Scheduler.sample` 内，外部 caller 无法逐 transition
记录并 replay；因此创建 controlled fork，抽出通用 primitive，同时让原
`generate/finalize` 原样组合它们。fork 禁止依赖 wm-infra、Ray 或 RL trajectory types。

本 sprint 只交付 **generic provider core**。Self-Forcing 是第一个真实 family consumer，
它在后续 sprint 关闭真 checkpoint/replay/GPU production conformance。这样依赖是单向的：

```text
provider generic primitives
  -> Self-Forcing family + renoise/cache replay
     -> provider production conformance
```

## 1. 已核对的上游事实

审阅基线：`NVIDIA/flashdreams@7780904d1c6acac3ca8b184a56a343ae47c536b8`。

- `DiffusionModel.generate()` patchify input、调用 `cache.start`、生成 initial noise，再把
  局部 `predict_flow` closure 交给 `scheduler.sample`。
- public return 是 `(clean_latent, FinalState)`；`finalize(FinalState)` 推进 AR cache。
- `FlowMatchScheduler.sample` 自己持有完整 renoise loop，并从第二个 forward 起重采样。
- transformer 已有 `predict_flow(noisy_latent, timestep, cache, input)`；尚缺 caller 可组合
  的 start/finish session API。
- Wan load 后会做 modulation shape normalization，然后默认 compile；Self-Forcing recipe
  还可启用 CUDA graph。
- 现有 Self/Causal-Forcing checkpoint transform 只解决 cold checkpoint wrapper keys，
  不等于 trainer hot update 或 LoRA 已实现。

## 2. 仓库边界

### FlashDreams fork 拥有

- composable denoise session API 与 caller-supplied initial noise/generator；
- cache/session 生命周期和 double-finish/finalize guard；
- raw weight owner、execution callable 与 in-place parameter installer；
- source checkpoint normalization hook、post-load normalization hook 与 raw destination
  schema validation；
- upstream public compatibility tests、CPU contract tests、最小 GPU parity tests。

### wm-infra 拥有

- `GenerationRuntime`、Ray actor、placement、admission/drain、sleep/wake/shutdown；
- request id、seed derivation、policy version 与 bounded failure handoff；
- SDE/renoise math、old log-prob、trajectory schema、replay 与 autograd；
- trainer canonical trainable-key mapping、reward、algorithm、family/provider binding；
- provider source pin、capability/readiness validation 与 conformance。

### Environment contract

FlashDreams 不是 SGLang 式远程 autograd service。fork 作为 lazy/optional dependency，以同一
immutable commit/build 安装到 rollout worker 和 trainer replay 两个环境；trainer 只在自己
进程内构图/backward。native-only family 的 import/config resolve 不得触发 FlashDreams
import。若两个环境的 package/source/schema identity 不一致，launch/preflight fail closed。

## 3. 实施阶段

### F0 — Controlled fork hygiene

1. implementation start 时重新核对 upstream commit，并从 immutable commit 创建 GitHub
   fork；不用 mutable branch 充当 source pin。
2. 使用新 sibling clone；`origin` 指向个人 fork，`upstream` 指向 NVIDIA。禁止修改当前
   `/home/mingfeiguo/Desktop/flashdreams` checkout 的 remote。
3. fork `main` 只追踪 upstream；primitive 按可 cherry-pick 的小 feature branch/commit
   提交。
4. commit 遵守 DCO/sign-off，保留 Apache-2.0、NOTICE、SPDX 与第三方 attribution。

验收：fresh clone 可从 pin 安装并运行 upstream CPU tests；未 import fork 时 wm-infra
行为不变。

### F1 — Composable denoise session

从现有 `generate` 抽出以下跨 family execution boundary：

```python
state = model.start_denoising(
    autoregressive_index=autoregressive_index,
    cache=cache,
    input=input,
    initial_noise=initial_noise,
    generator=generator,
)
flow = model.predict_denoise_flow(state, noisy_latent, timestep)
clean_latent, final_state = model.finish_denoising(state, clean_latent)
```

public `generate()` 继续调用这些 primitive 和原 scheduler；`finalize(FinalState)` 不改名。
`DenoiseState` 只保留 predict/finish 实际消费的字段；只为展示保留的字段在定义处标注
`display/provenance-only`。

行为不变量：

- fixed seed/input 下重构前后 public `generate` bitwise identical；
- 每次 session `cache.start` 与 `finish_denoising` 各一次；
- 同一 `FinalState` 至多 finalize 一次；进入下一 temporal AR chunk 前必须 finalize，
  最后一个 chunk 若不再读取可以不 finalize；
- caller-supplied initial noise 不会触发 provider 第二次抽样；
- model instance 首版由 native worker admission 保证单 active session，不新增不可恢复的
  module-global session。

### F2 — Raw state ownership + installation

Wan 当前会让 compiled wrapper 覆盖 raw `self.network`。fork 改成：

```python
self.network = raw_network
self._compiled_network = (
    compile_module(self.network) if compile_network else self.network
)
self._network_call = (
    CUDAGraphWrapper(self._compiled_network, ...)
    if use_cuda_graph else None
)
self._network_call_uncond = (
    CUDAGraphWrapper(self._compiled_network, ...)
    if use_cuda_graph else None
)
```

同时修改 `_select_network()`：CUDA graph 关闭时必须返回 `_compiled_network`，不能返回 raw
`network` 绕过 compile；开启时，filling 使用 cond/uncond wrapper 的 `drain`，steady state
使用对应 wrapper。完整调用链固定为：

```text
raw weight owner
  -> compiled/eager callable
     -> optional cond/uncond CUDA-graph wrappers
```

状态处理明确分成四个边界与 ownership：

1. fork：source checkpoint normalization；
2. fork：post-load module normalization；
3. wm-infra/Self-Forcing adapter：trainer canonical keys → raw destination names；
4. fork：只接受 raw destination keys 的 in-place installer + schema/digest validation。

它们共享一个 canonical raw destination key space 和 validation，不共享虚假的输入 mapping。
family-specific checkpoint wrapper 与 trainer trainable subset 可以不同；目标参数的名字只在
raw module/schema 一处产生。fork installer 不认识 `transformer.*` 或任何 wm-infra key。

hot update 对 raw parameter 做 in-place `copy_`，保持 storage address。更新后必须清空旧
权重产生的 AR cache/session。不要无条件重编译或 reset CUDA graph：只有 eager/compiled/
captured hot-update parity 证明旧输出仍存在，或 pointer/layout 改变时才 reset/re-capture。

首版 capability 是 strict/draining。完成所有 worker 的 key-schema、shape、dtype、digest
ACK 前不提交 native version；不支持 request-bound old slots 时禁止 non-draining。

CPU 门：对 raw destination payload 的 missing/unexpected/duplicate key、shape/dtype mismatch、
partial install 全部 fail closed；失败不发布新 schema/digest/version。GPU 门：eager、
compiled、captured 三条路径在 fixed input 上都必须看到新权重。

### F3 — wm-infra step adapter (after the configured blocking-call deadline gate)

新增一个薄的 `FlashDreamsStepAdapter`，先用 fork 内 test model/wm-infra fake model 验证：

- 把 FlashDreams provenance 与实际 family binding 注册到唯一 typed provider source；
- launch 前显式 provider choice 派生 executor/build/schema，外部-only family 缺省时拒绝；
- native request seed/initial noise → provider state；
- start/predict/finish/finalize lifecycle；
- native terminal cleanup 关闭 worker session/cache；
- trainer canonical trainable keys → raw destination mapping → F2 installer；
- malformed state、deadline、partial update 都 fail closed。

adapter 不拥有 renoise 数学、不导出 wm-infra trajectory，也不预建一个假想的统一
step-provider framework。具体 trajectory mapping、temporal executor 与 grouped replay
属于 Self-Forcing sprint。

F3 接入 `GenerationWorkerCore` 前，provider startup、generation 与 update calls 必须使用
native configured deadline；timeout 或 partial update 必须拒绝 partial output/version
publication，关闭 admission，并通过 terminal cleanup 把当前 attempt 交给 supervisor。

### F4 — Generic compatibility gates

按顺序执行：

1. upstream CPU tests；
2. public generate fixed-seed compatibility；
3. step API fake-model transition/session tests；
4. F2 CPU install transaction tests；
5. F3 fake worker lifecycle/failure tests；
6. GPU 可用后运行 eager/compiled/captured hot-update parity。

真实 Self-Forcing generation、renoise likelihood、cache replay 和 mp4 不属于本 sprint 的
完成条件；它们是下游 family 与 conformance gate。本稿创建时 GPU 被占用，未运行或声称
任何 GPU gate 通过。

## 4. Architecture hygiene

### 应改变

- 抽出 start/predict/finish，因为它们是外部 execution protocol 的真实边界。
- raw module 保持唯一 weight owner，显式串起 raw → compiled/eager → optional graph wrappers；
  cold/hot 输入分别 normalization 后汇入同一 destination schema。
- state/schema/digest 由实际 raw parameters 派生，不手写重复 key set。

### 保持不变

- public `generate/finalize` 保持薄 facade：它们是稳定 API 与 cache lifecycle boundary。
- transformer `predict_flow` 保持薄：它是跨 model family 的统一 hook。
- wm-infra adapter 保持独立：它隔离仓库类型、版本、生命周期与故障语义。
- `_FP32_BUFFERS` 被 scheduler `_apply` 直接消费，保证 schedule 不被 bf16 破坏；这是
  合理 module-state taxonomy，不删除。

### 禁止新增

- 不新增 `SUPPORTED_FLASHDREAMS_MODELS` 或 capability 大表；能力从实现和 typed binding
  派生。
- 不把 prompt/model taxonomy 塞进 adapter 的 ALL_CAPS 数据。
- 不创建没有 protocol/public/lazy-import/cache ownership 的转发 helper。
- 不让 fork import wm-infra 或导出 wm-infra-specific trajectory API。

## 5. Definition of Done（generic provider core）

- [ ] fork 可从 immutable upstream pin 重建，remote/branch/DCO/NOTICE 正确。
- [ ] public `generate/finalize` fixed-seed compatibility 通过。
- [ ] caller 可控制 initial noise，并逐次调用 transformer flow hook。
- [ ] source/post-load/raw install 留在 fork，trainer-name mapping 留在 wm adapter；raw
  destination schema 只有一处。
- [ ] partial install fail closed，strict/draining capability 无法被误报为 non-draining。
- [ ] raw owner 与 eager/compiled/captured callable 的 update contract 有测试。
- [ ] fake wm-infra adapter 不绕过 native runtime、deadline、terminal cleanup 或 version transaction。
- [ ] provider provenance 只构造一次，external-only family 无显式 binding 时 fail closed。
- [ ] adapter failure/shutdown 不留下 active session/cache。
- [ ] native provider 未被删除或降级为 fallback。

## 6. Production handoff 与 rollback

- Self-Forcing sprint 完成真 checkpoint collect/replay 后，FlashDreams binding 才可达到
  Integrated；只有完整 recipe/environment contract 也闭合时，multi-provider conformance
  才能把它提升为 Runnable。
- F1 无法保持 public compatibility：缩小/停止 fork patch，不在 wm-infra 复制 scheduler。
- F2 无法证明执行路径读取新权重：只允许 cold restart，不进入 continuous rollout。
- downstream family 无法重建 trajectory/cache：保持 inference-only；不污染 trainer schema。
- 性能不优于 native 不构成 correctness 失败；causal model coverage 本身可以是采用理由。

## 参考

- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/diffusion/model/base.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/diffusion/scheduler/fm.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/diffusion/transformer/base.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/cuda_graph.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/recipes/wan/transformer/wan21.py`
- `/home/mingfeiguo/Desktop/flashdreams/integrations/self_forcing/self_forcing/config.py`
- `/home/mingfeiguo/Desktop/flashdreams/CONTRIBUTING.md`
- `/home/mingfeiguo/Desktop/flashdreams/LICENSE`
- `docs/sprints/SPRINT_native_generation_engine_program.md`
- `docs/sprints/done/SPRINT_rollout_worker_liveness.md`
- `docs/sprints/parked/SPRINT_self_forcing_causal_family.md`
- https://github.com/NVIDIA/flashdreams

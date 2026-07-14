# SPRINT: Media artifact async write —— 把 reward 落盘（mp4/.pt 编码 + D2H）移出热路径

状态：**PARKED（2026-06-29）。** PROFILED + 重大更正：原「fire-and-forget」前提被**证伪**（写盘是 reward 打分的硬依赖——kling 按 path 读回 mp4,见下）。实测 encode 占 rollout **240p 2% / 480p 4.85%**。**Un-park 触发：committed 到 480p（droid）AND 已在改 reward 输入契约 → 那时做「Ray object-store 内存传输」（砍掉 libx264+落盘+读盘），不是写队列。** 否则不做。`.pt` 捷径对 kling 无效。 性质：**EXACT（无损）throughput 杠杆。** 把落盘型 reward 每个 rollout 的「GPU→CPU 拷贝 + libx264 逐帧编码」从同步内联（head-of-line 阻塞单线程事件循环）改成「`asyncio.to_thread` offload」,或更根本地用 Ray object-store 内存传输绕过编码+落盘。

> ## ⚠️ PROFILE + 重大更正（2026-06-29，RTX 5090，真 `write_mp4`）
> 探针：`scratchpad/io_stall_profile.py`（一次性）。
> - **实测 encode 成本**（真 `vrl.utils.media.write_mp4`, libx264）：240p_33f（416×240×33）**210ms/video**；480p_33f（832×480×33）**504ms/video**。每个 score call materialize ×8（`n_samples_per_prompt=8`）：240p **1.76s**、480p **4.32s**。占 89s rollout wall：**240p 2.0% / 480p 4.85%**。`.pt`（torch.save）比 mp4 便宜 ~10x（156ms vs 1758ms/call）。
> - **🔴 前提证伪——写盘不是 fire-and-forget,是 reward 打分的硬依赖。** artifact 只带 `path`（不带 in-memory tensor,`artifacts.py:79` `path=str(path.resolve())`）,Ray reward worker **从磁盘按 path 读回**再打分：`vrl/rewards/models/kling_video_reward.py:277-279` `artifact_path = str(Path(artifact.as_path())...)` → `self._reward([artifact_path], ...)`。所以 `materialize → score_batch(request, artifacts=...)`（`base.py:275,290`）里 **score 必须等 mp4 写完可读**。原 sprint「reward 数值与 mp4 无关、可 fire-and-forget、退出时 drain」是**错的**——不能 fire-and-forget。
> - **`.pt` 捷径对 kling 不可用**：kling reward model 读的是**可解码的视频文件**,不是 raw 张量 `.pt`；切 `artifact_format=tensor` 会让 kling 读不了。`.pt` 只对「能直接吃张量」的 reward 有效,kling 不在内。
> - **真正还能拿的杠杆（更正后）**：① **`asyncio.to_thread` offload encode**——score(N) 仍要 await 写完,但编码期间事件循环不被 head-of-line 冻住,continuous 模式下 generate(N+1) 可在编码时推进 → 把 2-5% 藏到下一组生成后面（modest,只在 continuous 模式真兑现）；② **根本解:Ray object-store 内存传输**——`ray.put(tensor)` 传 `ObjectRef` 给 reward actor,reward model 直接读张量,**砍掉 libx264 编码 + 落盘 + 读盘整条 round-trip**（mirror `generation/ray/weight_sync.py:57` 已有的 ray.put 模式）。代价:reward model 要加「吃张量而非 path」的入口（kling `_reward` 现在只接受 path）。
> - **净结论**：encode 是真成本（240p 2% / 480p 4.85% rollout，落在 droid 480p 配置上更值），但**①不是 fire-and-forget,②`.pt` 捷径对 kling 无效**。优先级:480p > 240p；内存传输（砍掉编码）> to_thread offload（只藏不消）。下面 §0/§4/§6 的「fire-and-forget / 退出 drain」段落以本 box 为准更正。

> 证据：`vrl/rewards/base.py:273-276`（`artifacts = artifact_builder(rollouts)` 内联同步、无 await/to_thread）、`vrl/rewards/base.py:242`（落盘型 reward 把 `artifact_builder` 接到 `VideoRewardArtifactStore.materialize`）、`vrl/rewards/base.py:134-138`（in-memory reward 走 `build_inmemory_artifacts`，**训练中不写盘**）、`vrl/rewards/base.py:286,304`（`artifact_materialization_ms` 已被计时——profile 门用它）、`vrl/rewards/artifacts.py:50-77`（`materialize` 循环 → `output.detach().cpu()` D2H → `write_mp4` / `torch.save`）、`vrl/rewards/artifacts.py:96-99`（每 artifact 一条 JSONL manifest append）、`vrl/utils/media.py:210-220`（`imageio` + `codec="libx264"` 逐帧 CPU 编码）、`vrl/rollouts/collector/rewards.py:97-101` + `vrl/rollouts/collector/core.py:182-188`（每个 score 批一次 → 热路径,不是 eval）、`vrl/rewards/functions/kling_video_reward.py:25-30`（`default_execution="pool"` → `_init_disk_artifact_reward`）。
> Current primitive correction (2026-07-13): the former physical-stage `pipeline_runner.py` example was deleted with its test-only seam. Current blocking-call offload examples live in `vrl/generation/ray/launcher.py`, `vrl/generation/ray/runtime.py`, `vrl/rollouts/orchestration/continuous/owner.py`, and the reward-service owner. Bounded task ownership remains in continuous orchestration, and pinned nonblocking D2H remains in `vrl/generation/execution/worker.py`; these are reusable patterns, not a physical-stage runtime contract.
> 相关：[[feedback_mfu_bound_north_star]]（probe 先行,先定 bound class 再背书）、[[project_real_run_profiling.md]]（量 GPU-idle 用 kernel-union,别用 nsys projection）、[[SPRINT_video_rollout_stage_overlap]]（同一类 EXACT 杠杆;那条 sprint 实测真正大 bubble 是 reward stage 14%,与本 sprint 互补）、[[project_reward_execution_backend]]（`execution: inline|pool`,pool=落盘型 reward 的来源）。

## 0. 一句话 + 为什么这是真 stall 而不是「写盘当然慢」

落盘型 reward（Kling / videocon / videoscore2 等 `execution=pool` 的视频 reward）**每打一批 rollout 就同步落一批盘**。落盘这步在**单线程 asyncio 事件循环上同步执行、没有任何 await**，所以编码 N 个 mp4 期间它 **head-of-line 阻塞整个事件循环**——卡住所有本可以去驱动下一组 denoise 的并发 collect 任务。这不是「写盘慢、忍一下」，而是「一个纯 CPU 的编码占着事件循环，把 GPU 也一起饿着」。

**为什么这个杠杆收益最干净**：reward 数值来自 Ray 打分，**与编码后的 mp4 文件无关**——文件只是给人看的 artifact。所以编码可以 fire-and-forget：把已经在 CPU 上的张量丢给后台 writer，主循环立刻继续。下游没有任何东西等这个文件 → 无 join、无正确性风险。这和 checkpoint（退出/weight-sync 前必须 drain）不同，是异步落盘里最容易拿、最该先拿的一个。

## 1. 现状（坐实，全是同步内联）

落盘调用链，从热路径一路到 libx264，**全程没有 await / to_thread / Ray-remote**：

```python
# vrl/rollouts/collector/core.py:182-188  —— 热路径入口,每个 score 批一次
rewards = await self.reward_scorer.score_many([...])
# vrl/rollouts/collector/rewards.py:97-101 —— score_batch per 批
raw = batch_fn(rollouts)               # → RewardFunction.score_batch
# vrl/rewards/base.py:273-276           —— 内联同步,无 await
materialize_started = time.perf_counter()
artifacts = artifact_builder(rollouts)   # == VideoRewardArtifactStore.materialize
materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
```

`materialize` 对 batch 里每个 rollout 循环，先 D2H（裸 `.cpu()`，无 pinned/non_blocking），再逐帧 libx264 编码或 `torch.save`：

```python
# vrl/rewards/artifacts.py:63-77
tensor = output.detach().cpu()            # ← 同步 D2H,卡 GPU
if self.artifact_format == "mp4":
    write_mp4(tensor, path, fps=fps)       # ← 逐帧 CPU 编码
else:
    torch.save(tensor, path)               # ← 同步写盘
# vrl/utils/media.py:210-220
with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
    for frame in frames:                   # 逐帧 append,纯 CPU
        writer.append_data(frame)
```

**只在落盘型 reward 时发生**：`execution=pool` 的视频/图像 reward（`kling_video_reward`、`videocon_physics`、`videoscore2`、`unified_reward_video`、`cosmos3_reasoner` 等）把 `artifact_builder` 接到 `materialize`（base.py:242）；in-memory reward（nsfw/aesthetic/pickscore/ocr/target_dino…）走 `build_inmemory_artifacts`（base.py:134-138），媒体留在内存，**训练中一个文件都不写**。所以本 sprint 只对「配了落盘型视频 reward」的 run 有意义。

> 注：online 路径只产 `mp4` 或 `.pt`（artifacts.py:72-77），**不产 png**；`write_png` 只在 `vrl/scripts/eval/*`、`vrl/scripts/data/*` 这些 `__main__` 离线入口里，与训练循环无关，不在本 sprint 范围。

## 2. 把成本拆两段（决定异步化能省什么）

| 成本段 | 真正卡什么硬件 | 异步化能不能藏 |
|---|---|---|
| (a) `output.detach().cpu()` D2H | 卡 GPU（copy engine 同步 + 隐式 sync） | pinned `non_blocking` 拷贝可缩短,但这步要在主路径上快做（见 §4 入队前） |
| (b) libx264 逐帧编码 / `torch.save` 写盘 | 卡 CPU + 磁盘,占着事件循环 | **后台 writer 队列可完全藏掉**（下游不依赖产物） |

(b) 是本 sprint 的主菜：纯 CPU 工作、fire-and-forget、藏得最干净。(a) 顺手用已有的 pinned 原语削一刀（与 [[SPRINT_checkpoint_async_write]] 的 gather 改进同源）。

## 3. 北极星纪律：KILL-RISK profile 门（必须先过，再写一行实现）

按 [[feedback_mfu_bound_north_star]]：**不准先建队列再说**。先证明编码 stall 是 rollout wall 的可观比例，否则就是给一个不痛的地方上麻药。

**门 1 —— 量 `artifact_materialization_ms` 占比（代码已有计时,base.py:286/304）。**
- 跑一个真落盘型 reward 的 cosmos/wan run（如 `kling_video_reward`），从 RolloutStats 里读 `artifact_materialization_ms`，对比同一组的 rollout wall（或 `collector.reward_score` stage wall，见 [[SPRINT_video_rollout_stage_overlap]] §3 的 nsys+NVTX 方法）。
- **过门判据**：`materialization_ms / rollout_wall ≳ 5%`（且绝对值不是个位数 ms）。低于此 → 不值得建队列，关掉本 sprint。
- 同时记录 `artifact_format` 实际是 `mp4`（libx264）还是 `tensor`（torch.save）——两者成本差一个数量级。

**门 2 —— 量它对事件循环的 head-of-line 阻塞（continuous 模式才有意义）。**
- 在 continuous producer 模式下，用 [[project_real_run_profiling.md]] 的 kernel-union 法量：编码窗口内 GPU 是否真的空转（即编码确实把本可驱动下一组 denoise 的任务饿着）。
- 若 GPU 在编码窗口内并不空（被别的并发任务填满）→ 阻塞是「纸面的」，收益打折，降级本 sprint 优先级。

**先考虑更便宜的替代（可能直接关掉本 sprint）：**
- 若门 1 显示是 **libx264 编码**（不是 D2H、不是 torch.save）主导，且不需要人看 mp4 → 把 `artifact_format` 切到 `tensor`（.pt，无 ffmpeg）可能比建队列更省事。
- 若该 reward 其实能 in-memory 打分（不需落盘）→ 走 `build_inmemory_artifacts`，根本不写盘。
- **只有当「确实需要 mp4 落盘 + 编码是实测可观成本 + 下游确实被饿」三者同时成立，才建队列。**

## 4. 设计（复用已有原语，别另造）

仓库零 `ThreadPoolExecutor`/`threading.Thread`/`queue.Queue`，但有现成范式：

**① 后台 writer = trainer 事件循环上的 `asyncio.Task`（mirror `ContinuousRolloutProducer`），不开独立线程。** 实际阻塞写用 `asyncio.to_thread` 丢给 stdlib 默认 executor：

```python
# 落点：VideoRewardArtifactStore 内部新增一个 enqueue + 后台 drain task
# mirror vrl/rollouts/orchestration/continuous/producer.py:77-90,109-143
await asyncio.to_thread(write_mp4, host_tensor, path, fps=fps)   # 编码在 worker 线程
# vrl/generation/ray/weight_sync.py:59 是这个 idiom 的现成样板
```

**② 入队的是「已经在 CPU 上的张量」,绝不入队 CUDA 张量。** 主路径只留一个快速 pinned `non_blocking` D2H，慢的编码/写盘丢后台：

```python
# 把 artifacts.py:63 的裸 output.detach().cpu() 换成 worker._to_cpu 的 pinned 版本
# vrl/generation/execution/worker.py:486-521（pin_memory=True; host.copy_(t, non_blocking=True); 末尾一次 cuda.synchronize()）
# 更进阶（专用 stream + event,不全局 sync,可与下一段 compute 重叠）：
# vrl/generation/diffusion/pipeline.py:156-169 _move_tree_to_cpu_async
```

**③ 有界队列 + 背压**：mirror `vrl/rollouts/orchestration/continuous/queue.py:19-37`（deque + byte 预算 + 满了淘汰/阻塞）。artifact 体积大（视频张量），必须有 byte 上限，避免后台编码跟不上时 host RAM 爆。背压策略二选一：满了就**同步退化**为内联编码（保证不丢），或**丢最旧**（artifact 仅供观察,可丢——但要计数 + 告警，见 §6）。

## 5. 阶段

- **P0 — KILL-RISK profile 门（§3）**：跑落盘型 reward run，量 `artifact_materialization_ms` 占比 + head-of-line 阻塞。**不过门就停。** 同时确认 `artifact_format` 与「是否真需要 mp4」。
- **P1 — pinned D2H（无条件小赢，独立可落）**：把 `artifacts.py:63` 的裸 `.cpu()` 换成 `worker._to_cpu` 的 pinned `non_blocking` 版本。这一步即使不建队列也成立，先落。
- **P2 — 后台 writer 队列**：在 `VideoRewardArtifactStore` 里加 enqueue（入队 host 张量 + 目标 path）+ 后台 `asyncio.Task` 用 `asyncio.to_thread(write_mp4/torch.save)` drain；`materialize` 改成「pinned D2H → enqueue → 立刻返回 artifact 句柄（path 已知，文件后台落）」。manifest JSONL append 也进后台。
- **P3 — 有界队列 + 背压 + 计时回归**：加 byte cap + 背压策略；用 P0 同一套 profile 确认 `materialization_ms` 在主路径上掉到 ~0、rollout wall 下降，且 GPU busy% 在编码窗口回升。

## 6. Drain 与失败处理

- **唯一硬 drain 点 = 进程退出 / 训练结束前**：reward artifact 无 weight-sync 依赖（不读模型参数），所以不像 checkpoint 那样需要 weight-sync 前 drain。退出前用 producer 的 `drain_inflight()` 语义（`producer.py:134-143`）排空，避免丢最后一批可观察 artifact。
- **失败必须可见（但可降级）**：artifact 是观察用途，编码失败可降级为「告警 + 丢失计数器」，不必让训练 crash——但**计数必须暴露在 RolloutStats**（mirror 现有 `materialization_ms` 的计量），否则会静默丢 artifact 还以为存了。背压淘汰同理：丢了几个、为什么丢，要进 metric。
- **与 checkpoint sprint 的区别**：那条（[[SPRINT_checkpoint_async_write]]）丢写必须 raise（丢 checkpoint = 丢训练状态）；本条丢写可容忍 + 告警。两者 drain 语义不同，不要混用一个队列。

## 7. Non-goals / 不确定

- **不碰 in-memory reward 路径**：`build_inmemory_artifacts` 本就不写盘，无事可做。
- **不碰 eval/data 离线脚本**：`vrl/scripts/eval/*`、`vrl/scripts/data/*` 的 `write_mp4`/`write_png` 是 `__main__` 入口，吞吐无关，留着。
- **不把 png 纳入**：online 路径不产 png。
- **不确定（profile 才能定）**：典型落盘 batch 的 N（rollout 数）× 视频帧数 T 决定绝对 stall ms——门 1 实测回答；`artifact_format` 在生产 run 里默认 `tensor` 还是被 YAML 覆盖成 `mp4`（base.py:155 `default_artifact_format`，per-reward 可覆盖）——P0 一并确认；rollout collector 与 optimizer step 是否同进程/同事件循环（决定阻塞是「卡梯度步」还是「只卡 rollout 吞吐」）——continuous 模式下是同一事件循环，strict 模式下 collect 在 step 内，两种都受影响，但量级不同。

# SPRINT: rollout 内存释放时保留冻结的 VAE / text-encoder（offload-and-restore，不要 discard+reload）

状态：**done（2026-07-09 复核）**。缺陷 B 的 driver frozen-component offload、
缺陷 A 的 colocated-trainer sleep/wake、CuMemAllocator 路径和 GPU 验收均已完成。
旧文所称“reward actor 需要 rollout bundle，因此 reward handoff 仍 deferred”已经失效：
reward 现在只支持 in-process `execution="inline"`，Ray reward pool 已删除；当前 resolver
也明确按 inline reward 派生 handoff。因此本 sprint 没有剩余交付，以下章节保留为实现过程
与测量记录，不再作为待办清单。

> **Runtime follow-up（2026-07-30）：** 下文“实现状态”与 §0–§1 保留 6 月当时的
> 缺陷分析和 API 名称，只用于解释 sleep/wake 决策的来源。`_RuntimeLease`、
> `with_release_after_collect`、`runtime._with_sleep_offload` 和
> `RayGenerationLauncher.launch` 均已删除。当前路径是 topology 派生
> `rollout.mode="on_demand"`，`RayGenerationLauncher.create_runtime()` 创建唯一
> `RayGenerationRuntime`，runtime 的 `activate()/offload()` 驱动
> `RayGenerationSession.wake_workers()/sleep_workers()`；不得按历史段落恢复 lease facade。

## 实现状态（2026-06-20）

**✅ 缺陷 B（in-process driver-offload，sprint 指定"先做、风险最低"）已落地：**
- 复核确认缺陷仍在：`offload_driver_model_for_rollout` 仍是 `self.model.to("cpu")`（`lifecycle.py:108`），而 SD3.5/cosmos/wan 把 pipeline 经 `object.__setattr__` 旁挂、只注册 transformer（`sd3_5/model.py:83-84`），故 `nn.Module.to` 漏搬冻结 VAE/text-encoder。（注：nextstep 做 `self.vae = pipeline.vae` 已注册，无此缺陷。）
- `DiffusionModelBase` 新增 `move_frozen_components(device)` + `_frozen_pipeline_modules()`：从 `pipeline.components` **派生**冻结子模块集合（nn.Module 且 ≠ trainable transformer），**单一真源**、不手维护名单；families 无 pipeline（anima 单文件 / replay model）则返回空、no-op。device-only 搬运，dtype 保留（VAE fp32、encoder frozen_dtype 不变 → 生成逐 bit 不变）。
- `RolloutLifecycle.offload/restore_driver_model_*` 在 `self.model.to(...)` 旁补 `self._move_frozen_components(...)`（getattr 守卫，AR family 无 hook 时跳过）。
- 测试：`tests/models/diffusion/test_frozen_offload.py`（派生集合排除 transformer/非 module；只搬冻结；无 pipeline no-op）+ `tests/rollouts/orchestration/test_driver_frozen_offload.py`（offload/restore 调用 + AR 无 hook 不崩）。`tests/models/diffusion` + `tests/rollouts/orchestration` 共 172 passed。
- 验收口径：本机无 GPU，§3 的 nvidia-smi/逐 bit/计时探针留作 GPU 跑测；CPU 侧用「记录 `.to` 目标」的 fake 证明搬运范围正确（冻结组件被搬、transformer 不被本 hook 重复搬）。

**✅ 缺陷 A（历史 lease 模式 kill→sleep）——colocated-trainer 这一类已落地（2026-06-29）：** 复核 §1.D 后发现"未定的 actor 生命周期架构"其实窄得多（见下方更新），于是把 colocated-trainer timeshare 这一类的 kill 改成了 sleep/wake。这里的 `_RuntimeLease` / `with_release_after_collect` 是当时实现名，当前等价行为由唯一 runtime + session 承担：
- `GenerationWorkerCore.sleep()/wake()`（`vrl/generation/execution/worker.py`）：level-1 offload——保住 executor，把 model（transformer 经 `nn.Module.to` + 冻结组件经**缺陷 B 的 `move_frozen_components`**）搬 CPU、`release_cuda_memory`；wake 从 host RAM 搬回**捕获到的原 GPU**，不重读磁盘。executor 被硬释放过则 `load_policy` 兜底重建。`RayGenerationWorker` 暴露 `sleep/wake` 透传（`vrl/generation/ray/worker.py`）。
- lease FSM（`vrl/generation/ray/runtime.py`）：`_RuntimeLease` 加 `sleep_eligible`/`asleep`；`release()` 对 sleep-eligible 租约调 `sleep_workers()`、保活 `state.runtime`，否则照旧 `shutdown()`（teardown）；`_ensure_runtime()` 见到"睡着"就 `wake_workers()` 并**重放 `last_state`**（与冷启动后 `update_weights(last_state)` 同一刷新契约）；`update_weights` 在睡眠期只记 `last_state` 不下推到 CPU 上的 worker。
- sleep-eligibility 派生（内联在 `with_release_after_collect` 里，不单拆函数）：仅当 `handoff.release_rollout_before_train and not release_rollout_before_reward` —— 纯 trainer 让位才 sleep；reward 要 bundle 一律 teardown；无 resolved topology 的手搓 config 默认 teardown。

**✅ cumem-backed offload（2026-06-29，把 naive 的"打平"推到"真赢"）：** 朴素 `to(cpu)+empty_cache+to(gpu)` 实测往返 3.7s、且 wake 后碎片化 +4.8GB（几十周期会把 colocated trainer 挤 OOM）。改用 vLLM 的 `CuMemAllocator`（verl-omni 同款机制，0.21 已装）：
- sleep-eligible worker 的 model 在 `CuMemAllocator.use_memory_pool(tag="weights")` 里分配（`GenerationWorkerCore._build_executor_maybe_pooled`，由 lease 给契约打 `extra["sleep_offload"]` 触发，`runtime._with_sleep_offload`）；`sleep()/wake()` 走 `alloc.sleep(offload_tags=("weights",))` / `alloc.wake_up(tags=["weights"])`，cumem 不可用（CPU 机/无 vLLM）或非 sleep-offload worker 自动回落朴素路径。
- cumem 用 CUDA 虚拟内存释放/重映射物理页、保留虚拟地址 → wake 不 re-malloc、不碎片化。
- 测试：`test_worker_sleep.py` 加 cumem 分支（sleep/wake 走 allocator 不动 module；`sleep_offload` 时进 pool、cumem 不可用回落、无 flag 不进 pool）+ `test_runtime_lease_sleep.py` 加契约打标（sleep-eligible 打 `sleep_offload`、teardown 租约不打）。全套 266 passed。

**✅ GPU 验收已跑（2026-06-29，GPU 0 独占，SD3.5-medium bf16，`vrl/scripts/perf/rollout_sleep_probe.py --backend both`）：**

> **探针已退役（2026-07-28）。** `rollout_sleep_probe.py` 在产出下表后删除：`97b3a89f`
> 之后 declared-CuMem family 不再回退到 naive CPU round trip，探针的 naive/cumem A/B
> 失去生产参照物；脚本本身也已随 `ModelBuild` 演进失效（手搓 `SimpleNamespace` 缺
> `pretrained_kwargs`，在加载模型前就抛 `AttributeError`）。**下表是这次测量的最终
> provenance**，不再可重跑。表中未归档的只有 sleep/wake 的分项与逐周期 min/max 分布。

| 指标 | naive | **cumem** | 冷重载 |
|---|---|---|---|
| sleep 后残留（驱动级 `mem_get_info`） | 1297 MB（仅 context） | 1327 MB（仅 context） | — |
| trainer 拿回 | 11132 MB | 11104 MB | — |
| wake 后碎片（vs load） | **+4830 MB** | **+28 MB** | — |
| sleep+wake 往返 | 3722 ms | **845 ms** | — |
| 冷重载本身 | — | — | 5425 ms |

结论：cumem sleep/wake **845ms**，比 naive 快 4.4×、比冷重载快 6.4×；对比真实 kill 路径（冷重载 5.4s + context 初始化 1.06s + Ray 重建调度 ~1–3s ≈ 6.5–8.5s）**快约 8–10×**，且几乎零碎片。残留两 backend 都降到 ~context（~1.3GB），"context 残留挤掉训练"确认是伪问题。注：probe 的残留早期误用 `memory_reserved()`（看不见 cumem 虚拟内存），已改驱动级 `mem_get_info`。坑：cumem pool 上下文里不能 `expandable_segments` / `empty_cache`（PyTorch pluggable-allocator bug），故只在 sleep-eligible diffusion worker 上启用。

**✅ 旧 reward-handoff 假设已关闭（2026-07-09 复核）：** reward 改为
in-process inline 执行后，不再存在“Ray reward actor 要进入 rollout bundle”的调度前提。
若未来重新引入远程 reward transport，必须按当时的 runtime 和 placement contract 另立
sprint，不能复活本文的旧 bundle 假设。

关联：[[SPRINT_compile_rollout_lifecycle]]（同一条 worker 生命周期上的常驻-vs-每周期重建权衡，编译产物的摊销与本 sprint 的冻结权重摊销是同一根轴）、[[SPRINT_framework_lessons_vrl]]（P1-2：sleep/wake vs actor teardown —— 本 sprint 是该课的具体落点）。

---

## 0. Core Decision（先看这一段）

**结论：vrl 当前在 Ray on-demand（release-after-collect）模式下，每个 collect 周期都会 discard 整个 executor 并冷重载冻结组件，这正是 verl-omni 用 `sleep_level=1` 刻意避免的反模式 —— 应改为 offload-and-restore（sleep/wake）。** 已逐跳核实：rollout 的 `release_rollout_runtime_memory` 最终把 worker 的 `self.executor` 置空并释放 CUDA（`vrl/generation/execution/worker.py:75`），下一次 `generate()` 重新 `launch()` 时走 `from_build` → `StableDiffusion3Pipeline.from_pretrained(...)`（`vrl/models/diffusion/sd3_5/model.py:131`），把 VAE + 3 个 text-encoder 从磁盘**整盘冷重载**一遍 —— 这些组件从不参与 weight-sync（只同步可训练的 transformer/LoRA），所以重载它们是纯浪费。verl-omni 对完全相同的问题给出了明确范式：diffusion pipeline 的 text-encoder/VAE "are not part of the trainable actor and therefore are not included in full-model weight syncs"，因此用 level-1 sleep 把它们 offload、wake-up 时 restore，而不是被 level-2 discard（`verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py:204-216`）。本 sprint 把这套 offload-and-restore 纪律引入 vrl 的两条释放路径（lease 模式 + 单卡 colocated 的 driver-offload 模式），让冻结权重在 GPU↔CPU 之间搬运而非销毁重建。

---

## 1. 已核实（verified） vs 未验证（未验证）

### A. 历史 lease / release-after-collect 模式：**discard + 冷重载（当时已核实，现已修复）**

逐跳链路（每一跳都本轮读过）：

1. 调度层调用 `release_rollout_runtime_memory`，它转调 collector 的 `release_runtime_memory`：
   > `release = getattr(self.collector, "release_runtime_memory", None); if callable(release): await release()`
   （`vrl/rollouts/orchestration/lifecycle.py:122-124`）
2. collector 把它转给 runtime 的 `release()`：
   > `release = getattr(self._runtime, "release", None); if release is not None: await release()`
   （`vrl/rollouts/collector/core.py:97-100`）
3. lease 模式下 `release()` 直接 `shutdown()` 内层 runtime（`vrl/generation/ray/runtime.py:131-144`，`state.runtime = None; await runtime.shutdown()`）；`shutdown()` 先对每个 worker 发 `actor.release_policy.remote()` 再 `kill_actors`：
   > `release_refs.append(actor.release_policy.remote())` … `kill_actors(ray, [worker.actor ...])`
   （`vrl/generation/ray/runtime.py:159,163-166`）
4. worker 的 `release_policy()` 把整个 executor 丢弃并释放 CUDA：
   > `self.executor = None; release_cuda_memory(gc_collect=True, ipc_collect=True)`
   （`vrl/generation/execution/worker.py:75-76`）—— executor 持有 family model，model 持有 VAE+text-encoder，全部随之销毁。
5. （历史实现，相关 API 已删除）下个周期 `generate()` 经旧 `_ensure_runtime()` 冷建
   worker，再走 `load_policy()` → `_build_executor()` → `build_runtime_bundle(build)`；
   当前实现由唯一 `RayGenerationRuntime.activate()` 延迟创建或唤醒
   `RayGenerationSession`，不得恢复旧 `RayGenerationLauncher.launch(...)` 路径。
6. SD3.5 的构建是整盘从磁盘加载并冻结：
   > `pipeline = StableDiffusion3Pipeline.from_pretrained(build.model_name_or_path, **load_kwargs)` … `pipeline.vae.requires_grad_(False)` … 三个 `enc.requires_grad_(False)`
   （`vrl/models/diffusion/sd3_5/model.py:131-144`）。

**判定：lease 模式确实是 discard + cold reload。** 冻结组件（VAE、text_encoder/2/3）每周期被销毁后从磁盘重载，而它们从不进入 weight-sync（worker 只 `install/load_trainable_state` 可训练权重，`vrl/generation/execution/worker.py:90-102`），所以这部分重载是纯开销，可被 offload-and-restore 完全省掉。

> 历史命名说明：当时 MEMORY/提示里的 `ReleasableRayGenerationRuntime` 对应
> `RayGenerationRuntime.with_release_after_collect(...)` lease 工厂。该工厂已经删除；当前由
> topology 派生的 `resources.lifecycle.rollout.mode == "on_demand"` 选择 deferred
> `RayGenerationRuntime`，session 在 handoff 时 sleep/wake，而不是恢复历史 kill+relaunch
> 路径。

### B. 单卡 colocated 的 driver-offload 模式：冻结组件**根本没被 offload（已核实，独立第二个缺陷）**

非 Ray 的 in-process colocated 路径走的是 `offload_driver_model_for_rollout`：
> `self.model.to("cpu"); empty_cuda_cache()`（`vrl/rollouts/orchestration/lifecycle.py:108-109`），restore 是 `self.model.to(self.device)`（`:114`）。

但 `SD3_5Model` 只把 transformer 注册成 `nn.Module` 子模块，pipeline（含 VAE + text-encoders）是用 `object.__setattr__` 旁挂的、**未注册**：
> `super().__init__()` 后 `object.__setattr__(self, "_pipeline", pipeline)`，随后只 `self.transformer = pipeline.transformer`
（`vrl/models/diffusion/sd3_5/model.py:84-90`）。

`DiffusionModelBase` 没有重写 `to()`（本轮 grep 确认无 `def to`，`vrl/models/diffusion/base.py:30` 起仅 `load/forward/encode_prompt/...`），因此继承 `nn.Module.to`，它**只搬注册过的子模块** —— 即只搬 transformer。**结论：driver-offload 路径里 VAE/text-encoder 一直留在 GPU 上**，既没被释放也没被搬到 CPU。这与 A 是相反方向的同一类 bug：A 把冻结组件销毁过头，B 把冻结组件该让位时没让位。两者都说明"冻结组件的生命周期没有被显式建模"。

> 注：driver-offload 的取舍（whether/where）由 GPU topology 决定、`"cpu"` 是唯一 off-GPU 目标，这点设计上是对的（`vrl/rollouts/orchestration/lifecycle.py:98-110` 注释）。缺陷只在"搬运范围漏了未注册的冻结组件"。

### C. verl-omni 的范式（已核实，作为 north star）

> "vLLM-Omni diffusion pipelines include components such as the text encoder and VAE that are loaded by the rollout server, but are not part of the trainable actor and therefore are not included in full-model weight syncs. Use level-1 sleep so those weights are offloaded and can be restored on wake-up instead of discarded by level-2 sleep."
> `await self.engine.collective_rpc("sleep", kwargs={"level": 1}); await self.engine.reset_encoder_cache()`
（`verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py:204-216`；wake-up 用 `tags=["weights"]` 通过 `collective_rpc("wake_up", ...)` 恢复，`:184-202`）

要点：**level-1 = offload-and-restore；level-2 = discard。** verl-omni 明确选 level-1 来保留冻结组件，并在注释里把 level-2（discard）标为"等 trainer 侧纳入整模型后"才用的 TODO。vrl 的 lease 模式现在等价于"永远用 level-2"。

### D. 未验证（未验证）

- **未验证：** cosmos / wan / janus_pro / nextstep 等其他 family 的 `from_build` 是否同样整盘冷重载冻结组件。本轮只核实了 SD3.5（`vrl/models/diffusion/sd3_5/model.py`）。设计上其余 family 走相同的 `_build_executor → build_runtime_bundle → from_build` 通道，预期同构，但**未逐个核实**。
- **未验证：** offload 冻结组件到 CPU 再搬回的耗时是否真的小于 `from_pretrained` 冷重载（直觉上 CPU↔GPU 拷贝远快于磁盘反序列化 + 构图 + 冻结，但**未实测**）。需在落地前用一次 probe 量化（见 §3 验收）。
- **已核实（2026-06-29，原"架构选型未定"项收窄）：** 复核两处代码后，"actor 生命周期是否每周期保活"不是一个不可知的架构分叉，而是一个可测量的显存问题：
  1. **lease 模式下 bundle 根本不交还。** `_ensure_runtime` 重 launch 时传的是同一个 `state.placement`（`runtime.py:181-186`），launcher 里 `owned_placement_group = None`、PG 由 `GlobalRayPlacementOwner` 持有、`shutdown` 从不 remove PG（`launcher.py:88` 注释）。所以 `kill_actors` 杀掉的只是 **actor 进程**，bundle 一直为 rollout 保留、下周期同 bundle 重建——kill 的唯一作用是回收那张物理卡的显存（含 CUDA context），**不是**把 Ray 调度位让给别的角色。
  2. **on_demand 只有两类触发**（`resources.py:328-335`）：`release_before_train = colocated and not persistent_colocated`（trainer 同卡相位切换）或 `release_before_reward = reward_shared_with_rollout`（reward 共享卡）。结合 (1)：trainer 这一类只需让出物理显存 → in-process sleep 充分；reward 这一类可能要让出 bundle → 保留 teardown。
  - 于是 A 收窄为**一个经验问题**：colocated 场景里 sleeping actor 残留的 CUDA context（~几百 MB）放在共享卡上，trainer 的训练步还塞得下吗？`rollout_sleep_probe.py`（一次性验收探针，已于 2026-07-28 退役删除；数字见上文表格，该表是最终 provenance）在 free-GPU 机上给数字。这正是 verl-omni `sleep_level=1`（offload 保活）vs `level=2`（discard）的同一道选择——vrl 原本的 lease=kill+reload 等价于"永远 level-2"。

---

## 2. 方案（落点与边界）

核心：把"释放 rollout 内存"从"销毁 executor / 杀 actor"细化为"按可训练性分层处理"——**可训练权重**走 weight-sync（已有），**冻结组件**走 offload-and-restore（本 sprint 新增），二者生命周期解耦。

- **缺陷 B（in-process driver-offload，先做、风险最低）：** 让 driver-offload 真正搬运冻结组件。两种合规实现，取其一：(1) 在 `SD3_5Model`（及同构 family）上提供显式的 `offload_frozen(device)/restore_frozen(device)`，由 `RolloutLifecycle.offload_driver_model_for_rollout/restore_driver_model_after_rollout` 调用，搬 `pipeline.vae` + 三个 text-encoder；(2) 若选择让 `model.to()` 覆盖全 pipeline，需在 base 显式重写 `to()` 并保证 dtype 契约（VAE fp32、frozen text-encoder 的 `frozen_dtype`，见 `vrl/models/diffusion/sd3_5/model.py:119-144`）不被破坏。**倾向 (1)**：语义明确（"冻结组件让位"而非"整模型搬家"），不动 `nn.Module.to` 的注册语义，且与 verl-omni 的"非 actor 组件单独 sleep"一一对应。

- **缺陷 A（lease 模式，主目标，依赖架构决策）：** 把 `release_after_collect` 的 `kill_actors` 路径替换为 "sleep actor"：actor 进程常驻，`release_policy` 不再 `executor=None`，而是把可训练 transformer 卸下（仍由 weight-sync 重装）+ 把冻结组件 offload 到 CPU；`_ensure_runtime` 的重 `launch` 改为 "wake"（把冻结组件搬回 GPU，不重读磁盘）。这需要 §1.D 未验证项落定：actor 是否保活。**这一步与 [[SPRINT_compile_rollout_lifecycle]] 的"编译产物常驻 vs 每周期重建"是同一根生命周期轴** —— 若 actor 保活，编译产物与冻结权重可一起摊销到 ~0。

- **冻结组件的 single source of truth：** 不要在生命周期层硬编码"哪些是冻结组件"。冻结性已在 `from_build` 用 `requires_grad_(False)` 标注（`vrl/models/diffusion/sd3_5/model.py:135-142`）。offload/restore 的对象集合应从 model 暴露的"非可训练子模块/pipeline 组件"派生，避免在两处各维护一份名单而 rot。

---

## Non-Goals

- 不改 weight-sync 语义、不改 `install/load_trainable_state` 的版本化 slot 逻辑（`vrl/generation/execution/worker.py:90-107`）——可训练权重的同步通道保持不变。
- 不改调度架构（strict_on_policy / continuous / StalenessPolicy），不引入新的 overlap 模式。本 sprint 只动"释放/恢复内存"的内部实现。
- 不改 GPU topology 派生的 whether/where-offload 决策（`vrl/rollouts/orchestration/lifecycle.py:95-110`、`vrl/generation/ray/launcher.py:202-213`）——只补"搬运范围"和"搬运 vs 销毁"。
- 不在本 sprint 验证 cosmos/wan/janus/nextstep 的同构性（列入 §1.D 未验证，落地时逐个核实，不预先改）。
- 不引入新的 public YAML 开关（与 [[SPRINT_compile_rollout_lifecycle]] §0 一致：lease/colocated 由 topology 派生，不是用户旋钮）。

## 验收（finishing criteria）

1. 缺陷 B：单卡 colocated 跑一周期，offload 后 `nvidia-smi`/CUDA mem 显示 VAE+text-encoder 确实离开 GPU，restore 后回到 GPU，生成结果逐 bit 与未 offload 一致（冻结组件无随机性）。
2. 缺陷 A：lease 模式一周期，确认 `from_pretrained` 不再在第二个周期被调用（加临时计数/日志或 host-memory 探针 `log_host_memory`，`vrl/generation/execution/worker.py:67,70`），且端到端 reward 曲线不变。
3. 实测 offload-restore 周期耗时 < 原 discard-reload 周期耗时（兑现 §1.D 未验证项），数字记入本 sprint 收尾段。

## References

阅读文档 / 范式来源：
- verl-omni `_sleep_hybrid` / `wake_up`（level-1 offload-and-restore 非 actor 组件）：`/home/mingfeiguo/Desktop/verl-omni/verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py:184-216`

vrl 代码路径（本轮逐跳核实）：
- `vrl/rollouts/orchestration/lifecycle.py:98-125`（driver-offload `model.to("cpu")` + `release_rollout_runtime_memory`）
- `vrl/rollouts/collector/core.py:97-100`（`release_runtime_memory` → runtime `release`）
- `vrl/generation/ray/runtime.py:67-95,131-193`（lease 工厂、`release/shutdown/_ensure_runtime` kill+relaunch）
- `vrl/generation/ray/launcher.py:202-213`（on-demand 触发条件）
- `vrl/generation/execution/worker.py:56-76,88-107,317-347`（`load_policy/release_policy/update_weights/_build_executor`）
- `vrl/models/diffusion/sd3_5/model.py:84-90,119-144`（pipeline 旁挂未注册、`from_build` 整盘冷重载 + 冻结）
- `vrl/models/diffusion/base.py:30-`（`DiffusionModelBase` 未重写 `to()`）

相关 sprint：
- [[SPRINT_compile_rollout_lifecycle]]（`docs/sprints/done/SPRINT_compile_rollout_lifecycle.md`）
- [[SPRINT_framework_lessons_vrl]]（`docs/sprints/reading/SPRINT_framework_lessons_vrl.md`）

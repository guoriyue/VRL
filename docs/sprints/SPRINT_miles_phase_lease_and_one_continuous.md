# SPRINT: slime/MILES 边界对齐——单卡 phase lease + 唯一真实 continuous

状态：in progress（2026-07-11）。

> **阅读边界（2026-07-18）**：顶部“实施记录”和 §0 的 topology contract 是当前事实；
> §1 是固定 upstream commit 的源码审计；§2–§8 保留 2026-07-11 实施前计划与验收设计，
> 其中的“当前”、旧路径和待删除清单描述的是当时基线，不是现在的仓库状态。当前已完成项与
> 剩余 gate 只以上方实施记录为准。被删除的 single-GPU async debug preset 仅作为历史 producer
> 证据保留在原计划中，不能再当作现有 config。

## 实施记录（2026-07-11）

本轮已经落地代码合同，但 umbrella sprint 尚未完成；真实一卡多轮和真实两卡 overlap 两组硬件
acceptance 仍是完成门槛。

已完成：

- 删除 role `memory_fraction`、`require_separate_gpus`、bounded-resident lifecycle、worker cap、
  probe fraction 与 test-only pipeline fraction；旧 key 只保留 targeted migration rejection；
- resolved shared trainer/rollout topology 唯一派生 `on_demand`，continuous shared/reward handoff 在
  model/Ray launch 前失败，direct schedule construction 仍有 unconditional runtime backstop；
- single-process trainer parking 覆盖 live model/ref、Adam/AdamW8bit state、EMA、GradScaler 与 live
  gradients；DDP/FSDP shared topology 在 model/Ray launch 前 capability-gate；
- rollout worker 对支持声明、concrete CPU/CuMem parking path、完整 worker report 与 physical residual
  做行为验证；offload/residual 失败时 strict 不恢复 trainer，先由 terminal cleanup 回收 rollout；
- continuous producer/queue/consumer/runtime async 操作由 schedule-owned dedicated thread/event loop
  独占；main thread 只 export immutable CPU snapshot 并通过 command/future boundary 交给 owner；
- continuous commit 固定为 pause -> drain-or-slot -> update -> installed-version ACK -> publish/purge ->
  resume；partial/mismatched ACK 关闭 admission、quarantine runtime；shutdown cleanup 失败可重试，成功
  后才 stop/join owner；
- typed continuous staleness default/minimum 都是 `1`；zero-window production 配置改用
  `strict_on_policy`；旧 single-GPU continuous debug preset 已删除。
- reward preflight 现在按有效执行设备验证完整 parking proof：单卡 DINO/RAFT 配方显式走 CPU，
  Kling+VideoCon 复合配方只让 Kling 成为共享 GPU/CuMem owner、VideoCon 显式走 CPU；空
  `import_path` 的 GenEval 必失败 experiment 已从 active surface 删除。全量 online experiment
  静态测试统一执行 load -> build -> resource resolve -> schedule topology -> reward parking preflight。
- reward lifecycle 现在只由 resource topology 派生；YAML 的 `sleep_offload` / residual-limit override
  在 model launch 前给 targeted rejection。一个进程最多一个 configured-GPU reward CuMem owner，
  CPU siblings 与 weight=0 observation-only siblings 仍执行；SANA 的 PickScore observer 明确降级 CPU。
- reward component observation 不再写共享 `last_components`：每次 score 的 aligned values 随
  `RolloutBatch.extras` 到 `TrainingBatch`，trainer 只汇总本次实际 consumed rows 并写入
  `TrainStepMetrics`。continuous 已预取/评分的未来 batch 不会污染当前 metric row。
- terminal ownership 已收敛为 schedule/collector 唯一释放整条 rollout+reward pipeline；continuous
  cleanup 是 owner-loop single-flight，失败可重试且不覆盖首个 operation/ACK root。placement group
  在 raw handle 返回时立即被 owner claim，ready/probe/assign 或 probe-actor kill 失败都可保留 exact
  handle 供 shutdown retry；Ray session 只在成功后 commit closed。
- CuMem model-building scope 变成 one-shot，score/generate 不再重入；terminal close 显式释放 vLLM
  retained MemPool。shared reward shutdown 做 physical residual gate，dedicated CUDA reward 也无条件
  GC/empty-cache，避免长驻 driver 在下一次 recipe 中保留 allocator pages。

本机一次性硬件证据（RTX 5090）：

- `AdamW8bit` 建立 optimizer state 后完成 CPU park/restore，并成功执行第二次 optimizer step；
- live model/ref/optimizer/EMA/GradScaler/gradients 的 CUDA round-trip 后继续 step；
- trainer main thread 连续执行真实 CUDA backward 时，dedicated owner 的 tick、submitted、completed
  counters 均继续前进，证明 owner 不依赖 trainer asyncio loop 的让步点；
- CuMem one-shot sleep/wake/terminal-close subprocess 通过；256 MiB pooled model 从
  `13,404,209,152` bytes 回到 `13,135,773,696` baseline，allocator registry 的 tag 被删除；
- dedicated 256 MiB reward shutdown 同样从 `13,404,209,152` 回到 `13,135,773,696`，
  `torch.cuda.memory_reserved()==0`；tiny shared reward 的首次 CUDA runtime drift 为 2 MiB；真实
  CLIP-L score 在模型池完整卸载 1.64 GiB 后留下 42 MiB（fresh process 为 126 MiB），与 generation
  共用实测上界 256 MiB 的 CUDA runtime residual allowance；
- tiny generation worker CPU fallback sleep 返回到 pre-load physical baseline；真实 Ray actor 的
  ObjectRef 直接 await/ACK 通过，不再创建不可 join 的 `to_thread(ray.get)` waiter。

本轮仓库验证：`pytest -q` 为 `1907 passed, 11 skipped`；phase-lease/continuous/reward/placement
聚焦回归、全量 active online config preflight、Ruff、`compileall` 与 `git diff --check` 全部通过。
skip 是需要外部真实 checkpoint 的既有 E2E gate，不作为本轮硬件 acceptance 的替代。

仍未完成，因此状态保持 `in progress`：

- 尚未用真实 production model 跑完一卡三轮 rollout/train + checkpoint + clean shutdown；
- 当前机器只有一张 GPU，尚无真实两卡 generate/backward interval overlap 与 placement-release 证据；
- worker host RSS 目前只有 load bookend logging，没有可信 host-memory budget/raise consumer，不能宣称
  host-OOM gate 完成；
- DDP/FSDP collective-safe shared parking 未实现；原 2x1 colocated configs 保留为明确
  capability-gated 的未来 acceptance shape，不再声称当前可运行。

## 0. 决策

本文把“1 fixed continuous”定义为：**仓库只保留一种 production
`continuous` 语义**，而不是承诺一张 GPU 上无预算地同时跑 rollout 和
backward。

VRL 最终只支持下面三种运行形态；shared GPU 只有一个 lifecycle，production
`continuous` 也只有一个：

| schedule | trainer / rollout topology | GPU execution | lifecycle | intent |
| --- | --- | --- | --- | --- |
| `strict_on_policy` | shared GPU | phase-serial | on-demand；trainer / rollout 交替 park / resume | production 单卡/共卡唯一模式 |
| `strict_on_policy` | trainer/rollout disjoint | serial by policy | 默认两边 resident；若 reward 与 rollout 共卡，rollout 仅在 reward handoff 时 park | strict 语义验证 / reward phase sharing |
| `continuous` | trainer/rollout disjoint，且 reward 不要求 rollout mid-iteration park | real wall-clock overlap | 两边 resident，producer 长驻 | 唯一 production continuous |

该表定义 schedule/topology 语义，不承诺每个 trainer strategy/backend 都已实现 parking。
Single-process、DDP、FSDP、bitsandbytes、各 rollout family 必须分别通过 capability；未实现者在
preflight fail-fast，不能继承一个看似通用的 `.to("cpu")` 路径假装支持。

下面这组组合必须在 model、placement group、Ray actor launch 前失败：

```text
continuous + shared GPU = configuration error

continuous + rollout/reward phase handoff = configuration error

distributed.resources.rollout.memory_fraction = removed configuration error
```

物理边界不能靠调度命名绕过：

```text
single GPU
    => exclusive phase lease
    => rollout and backward do not execute concurrently

real continuous overlap
    => independent trainer and rollout GPU capacity
```

这与 slime/MILES 在 schedule × topology 轴上的硬边界一致：两者都把 colocate 放在同步
`train.py`，并在 `train_async.py` 明确拒绝 colocate。两者还保留显式 no-offload resident-sync
recipes，但 upstream parity 不是 VRL 保留一个产品面的充分理由：VRL 唯一设置 role
`memory_fraction` 的 active preset 正是本 sprint 要删除的 same-GPU continuous probe；其他真实
shared-GPU recipes、E2E 和 scripts 都不使用它。因此 VRL 有意采用更窄的产品面，不把 upstream
resident-sync 子模式搬进来。

本文的 `on_demand` 只描述 GPU state residency，不描述 Ray actor/process 是否存活；actor 可以
长驻，只是 GPU physical pages 在 handoff 时被 park。

本 sprint 的两个交付目标是：

1. 单卡 shared-GPU 路径通过**完整双侧 parking**，稳定跑多步 strict on-policy；
2. 分卡路径把现有 continuous producer 从 trainer 事件循环中解耦，得到唯一、
   可测量、不会悄悄退化成 strict 的 continuous 实现。

这两个目标不是一个原子实现批次。本文件是共同决策/拓扑合同，实施拆成两个可独立验收的
track：

- **Track A — shared strict phase lease**：P0 的一卡 gates + P1 + P2 + P3 + real one-GPU
  acceptance；
- **Track B — disaggregated continuous owner**：P0 的 owner/overlap gates + P4 + P5 + real
  two-GPU acceptance。

Track B 依赖 P1 的 pre-launch topology verdict，但不依赖 Track A 为每个 trainer/backend 完成
parking。两个 track 应分别记录状态和 DoD；不能因为一卡 lease 完成就宣称 continuous 完成，
也不能因两卡 owner 完成就宣称 shared lifecycle 完成。若执行阶段需要独立 sprint 文件，直接
按这两个 track 拆，不复制本文件的共同决策。

共同前置依赖已经由 `done/SPRINT_remove_inline_fixed_eval.md` 完成：生产 surface 已删除并迁移到
独立 evaluator。本 sprint 继续以“training loop 中不存在 inline eval”为输入合同，不同时维护
新旧两套 lifecycle。

## 1. slime/MILES 源码结论

审计基线（2026-07-11，固定到 upstream commit permalink）：

- MILES `5e392cf2cef761d4508e81b2a8f4c4b191d7a73c`；
- slime `680824dd5e01a2e83750bf87fc366ec6fa98766c`。

### 1.1 默认 colocate 是严格 phase handoff

[MILES `train.py`](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/train.py#L89-L121)
和 [slime `train.py`](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/train.py#L49-L91)
的默认 colocate 循环都是：

```text
generate
-> offload rollout CUDA graph / KV / weights
-> wake trainer and train
-> offload trainer
-> onload rollout weights
-> update rollout weights
-> onload rollout KV / CUDA graph
```

`--colocate` 在用户未显式覆盖时会派生 `offload_train=True` 与
`offload_rollout=True`（[MILES resolver](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/utils/arguments.py#L2623-L2651)、
[slime resolver](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/slime/utils/arguments.py#L1882-L1904)）。
Trainer 侧不是只搬一个 `nn.Module`：

- Megatron 使用 `torch_memory_saver.pause()/resume()`，并销毁/恢复 process groups
  （[source](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/backends/megatron_utils/actor.py#L257-L284)）；
- FSDP 明确移动 model 和 optimizer state
  （[source](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/backends/experimental/fsdp_utils/actor.py#L282-L306)）。

这对应 VRL 的 `strict_on_policy + on_demand`，不对应 continuous。

### 1.2 显式 no-offload 是 resident，但仍然 sync

两家 upstream 都有真实的 resident-sync recipe：

- [MILES Gemma-4](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/scripts/run_gemma_4_31b.py#L120-L145)
  使用 `--no-offload-train --no-offload-rollout` 与
  `--sglang-mem-fraction-static 0.5`；
- [slime AMD Qwen3-4B](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/scripts/run-qwen3-4B-amd.sh#L115-L157)
  使用相同 no-offload 形状与 `--sglang-mem-fraction-static 0.7`。

这证明“shared GPU 必然 on-demand”不是上游事实；它不证明 VRL 必须实现同一子模式。VRL 的
repo-owned producer 审计当时只有一个真实 `memory_fraction` preset：
`online_grpo_ocr_single_gpu_async_debug.yaml`，其创建目的就是 single-GPU continuous probe；该
preset 已按本 sprint 删除。
本 sprint 禁止该 topology 后，bounded-resident feature 不再有 recipe、script、E2E 或 CI
consumer。把旧 preset 改名成 strict resident harness 会是在制造消费者来保留功能，因此不做。

结论是有意缩小产品面：VRL shared GPU 永远 on-demand phase lease；如未来出现具体 recipe、
测得 park/wake 是瓶颈并补齐真实硬件 acceptance，再从 engine-specific source of truth 重新引入
resident budget，而不是保留 dormant public key。

### 1.3 Colocate 即使 offload 也有 engine-local inference budget

MILES 的单卡 true-on-policy 示例仍设置
`--sglang-mem-fraction-static`：Qwen3-0.6B 为 `0.4`，Qwen3-4B 为 `0.2`
（[官方 recipe](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/examples/true_on_policy/run_simple.py#L72-L94)）。
[MILES 文档](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/docs/user-guide/training-script-walkthrough.md#L321-L350)
与 [slime 文档](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/docs/en/get_started/usage.md#L138-L150)
也明确说，即使顺序 offload，Megatron 与 SGLang 的初始化/残留内存仍要求彼此
留空间。

因此不要把 MILES 描述成“自动用完当前 free memory，不需要任何 budget”。它的选择是：

```text
fixed SGLang active/KV budget + phase switching
```

这里还有三个不能混淆的数字：

- Ray `num_gpus=0.4/0.2` 是 trainer/rollout actor 的 placement 资源声明，不是 VRAM cap
  （[trainer actor](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/ray/placement_group.py#L133-L147)、
  [rollout actor](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/ray/rollout/server_group.py#L80-L123)）；
- `sglang_mem_fraction_static` 是 SGLang active model/KV budget；
- `train_memory_margin_bytes` 是 trainer memory-saver margin。

这不等于 VRL 当前的 `distributed.resources.rollout.memory_fraction`：前者是有真实 engine
consumer 的 active/KV capacity，后者只为即将删除的 shared-resident lifecycle 服务。删除 VRL
role cap 不等于删除 backend budget；SGLang/KV/cache 等预算继续留在各自 engine 配置。它们不能
承担 schedule 选择，也不应被通用 role-level dormant knob 代替。

### 1.4 真 async 明确要求 disaggregation

MILES 与 slime 的 async 入口都有相同硬约束
（[MILES](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/train_async.py#L18-L94)、
[slime](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/train_async.py#L9-L70)）：

```python
assert not args.colocate, "Colocation is not supported for async training."
```

普通 async 在训练当前 batch 时提前生成下一 batch；
[fully-async 示例](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/examples/fully_async/fully_async_rollout.py#L52-L170)
再加一个独立 thread + asyncio loop，让 producer 持续拉 prompts。这个
rollout function 运行在 MILES 的 `RolloutManager` Ray actor 内，thread 并不依附 trainer
driver 的事件循环。

权重更新不是让新请求边生成边看到半套权重。更准确的 MILES 顺序是：

```text
choose/stamp target weight_version
-> pause new generation
-> transfer weights carrying that version
-> wait for every update/barrier
-> resume generation (the effective publish point for new admission)
```

FSDP updater 的实现直接体现了 version increment、`pause_generation`/cache flush、分 bucket
update、barrier、`continue_generation` 的顺序
（[source](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/backends/experimental/fsdp_utils/update_weight_utils.py#L48-L90)）。

VRL 应搬的是这个**模式边界和 commit protocol**，不是照抄 example 里的 module-global
worker、`Queue(maxsize=1000)`、一秒轮询和 minimal error handling。

## 2. VRL 实施前差距（2026-07-11 历史快照）

### 2.1 Bounded resident 将失去唯一 producer，应连根删除

当前 `vrl/ray/resources.py:157-163,296-318` 的推导是：

```text
gpu_pool=trainer, no memory_fraction  -> on_demand
gpu_pool=trainer + memory_fraction    -> resident
```

但“字段最终被 worker 读取”不能单独证明这是应保留的产品能力。Repo-owned producer 审计显示，
实施前唯一实际设置 role `memory_fraction` 的 active preset 是后来已删除的
`vrl/config/presets/experiment/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml`；它只为
same-GPU continuous probe 存在。本 sprint 禁止该 topology 后，仓库没有 bounded-resident recipe、
script、E2E 或 CI consumer。把旧 preset 改名成 strict resident harness 会是在制造消费者来保留
无人使用的能力。

这条 plumbing 还有两个完整性问题：

- `RayGenerationRuntime._rollout_memory_fraction()` 只读 `_on_demand` state；设置 cap 时 runtime
  必然 resident、`_on_demand=None`，所以 chunk probe 的 fraction reader 是 live caller / dead
  semantics；
- `cap_cuda_memory_fraction()` 声称安装 hard cap，却吞掉所有异常；现有测试只覆盖数值范围和
  `None`，没有证明真实 allocator cap 安装成功，也没有硬件 acceptance。

真正的 topology 缺口是 continuous 还能用 `require_separate_gpus:false` 绕过 gate
（`vrl/rollouts/orchestration/continuous/schedule.py:277-301`）。该手动 flag 与 resolved
topology 重复，而且 guard 直到 `next_iteration()` 才执行；当前 online wiring 已经在这之前
构建 trainer model、placement group 和 rollout runtime。最终矩阵应只有：

```text
strict + shared      -> on_demand
strict + disjoint    -> resident
continuous + shared  -> pre-launch error
continuous + disjoint -> resident
memory_fraction      -> removed-key error
```

因此本 sprint 同时删除 `require_separate_gpus` 和 role-level cap 的配置、resolved field、launch
contract、worker cap 与 probe plumbing。Topology verdict 必须在
`resolve_distributed_resources()` 之后、model/Ray 初始化之前由纯 preflight 消费 resolved plan；
schedule 内的 runtime capability guard只作为协议 backstop，不能再承担首个用户错误。

### 2.2 Strict 控制流已正确，但 trainer parking 不完整

`StrictOnPolicyRolloutSchedule` 已经按正确顺序执行：

```text
ensure weights
-> offload driver model
-> activate rollout
-> collect
-> offload rollout
-> restore driver model
-> train
-> stage/push next weights
```

证据：`vrl/rollouts/orchestration/strict_on_policy.py:45-82`。

真正缺口在 `RolloutRuntimeCoordinator.offload_driver_model_for_rollout()`：当前只做

```python
self.model.to("cpu")
move_frozen_components("cpu")
empty_cuda_cache()
```

见 `vrl/rollouts/orchestration/rollout_runtime.py:95-125`。它没有统一 park：

- optimizer state；
- EMA tensors；
- 独立 `ref_model`；
- strategy / FSDP / DDP 持有的 CUDA state；
- backend workspace、compile/CUDA graph residual。

Optimizer 和 EMA 又是 lazy 的（`vrl/trainers/online/trainer.py:572-588`），所以“第一步
能跑”不是有效验证；第二个 rollout phase 才第一次撞上已经建立的 Adam/EMA state。

### 2.3 Rollout parking 需要从 best-effort 变成可验证合同

On-demand worker 已经有正确骨架：CuMem 可用时 sleep/unmap、wake/remap；不可用时
fallback 到 model/frozen CPU move（`vrl/generation/execution/worker.py:77-166`）。但
`sleep_offload=True` 当前不保证 backend 的所有 CUDA allocation 都被 parking 覆盖，也没有
parking 后 residual 验证。

稳定的 shared-GPU 模式不能把“不完整 offload”静默当成功。每个 backend 必须满足以下一个
合同：

```text
complete pause/resume implementation
OR explicit unsupported error before training starts
```

### 2.4 Inline fixed eval 不再是 training lifecycle 的消费者

`docs/sprints/done/SPRINT_remove_inline_fixed_eval.md` 已把根因定为：fixed eval 错误复用了
live training collector/runtime，并手工复制了半套 handoff。该 companion sprint 已从
`vrl/scripts/common/online.py` 删除 inline eval、`trainer.eval` 与
`vrl/scripts/common/fixed_eval.py`，并改由独立 evaluator 加载完整 checkpoint；迁移和测试已经独立
验收，而不是只凭某个 worktree 中的部分删除宣称完成。

本 sprint 不应把这条待删除旁路重新包装成通用 `rollout_phase` context manager。删除 inline
eval 后，live trainer phase lease 的唯一生产调用方就是 strict schedule；再抽一个只有它调用的
context manager 会把同一顺序拆成两个私有概念，违反 single-caller concept-split 审计。正确边界是：

```text
strict schedule owns the phase order top-to-bottom
trainer/strategy owns parking mechanics
generation runtime owns activate/offload mechanics
standalone evaluator owns a separate runtime process
```

Standalone evaluator 可以调用通用 `activate/generate/offload/shutdown` 协议，但不能读取 live
trainer、复用 training placement group，或要求 continuous owner 增加 eval command。任何未来确有
生产调用方的 live rollout-only flow，必须先单独证明其 ownership，再决定是否抽共享 phase boundary；
本 sprint 不为假想调用方预建薄抽象。

### 2.5 当前 continuous 还不是唯一、可证明的 production 语义

当前 continuous 有三个问题：

1. `require_separate_gpus:false` 允许 resident same-GPU debug，并由唯一 role-cap preset 为它
   提供显存切分；
2. `max_stale_policy_versions=0` 时虽然数据语义等价 strict，仍启动整套 background
   producer/queue；
3. producer 跟 trainer orchestration 共用 asyncio loop，长同步 backward 期间只能依赖
   timestep 间的 `await asyncio.sleep(0)` 获得调度机会。

第三点不会停止已经发到远端 Ray worker 的 kernel，但会阻止 driver 及时 refill/harvest，
所以“有 background task”不等于持续 producer。真正的 continuous 验收必须看 wall-clock
区间重叠和训练期间完成量，不能只看 `schedule_mode`。

## 3. 目标合同

### 3.1 Shared-GPU 默认 on-demand exclusive lease

```text
trainer active, rollout parked
    -> park complete trainer-owned GPU state
    -> activate/wake rollout and install desired policy
    -> generate (and shared reward phase if configured)
    -> drain admitted generation
    -> park complete rollout-owned GPU state
    -> restore trainer-owned GPU state
    -> backward / optimizer step
    -> stage desired policy for next wake
```

关键不变量：

- trainer restore 之前 rollout 必须已经完整 park；
- rollout wake 之前 trainer 必须已经完整 park；
- `update_weights` 在 rollout parked 时只推进 `desired_policy`；
- next wake 先安装 desired version，再允许 generate；
- actors/processes 可以长驻；除已声明、经上限验证的 CUDA context/library residual 外，
  role-owned GPU physical pages 不能跨 phase 偷留；
- normal handoff 不 kill actors，不重建 placement group。

### 3.2 Disaggregated continuous

```text
trainer GPU:  train n ------- train n+1 -------
rollout GPU:  generate n+1 -- generate n+2 -----
                         ^ policy commit gate
```

关键不变量：

- continuous launch 前证明 trainer/rollout GPU pools disjoint；
- continuous launch 前证明 `RayLifecyclePlan.handoff.release_rollout_before_reward=false`；
- runtime 始终 resident；continuous 不调用 trainer park、rollout activate/offload；
- producer 的控制 loop 不依赖 trainer asyncio loop 的让步点；
- 每个 group 带 submit-time policy version；
- commit 是 `pause admission -> drain or version-slot handoff -> transactional update -> resume`；
- 所有 worker ACK 后才发布 committed version；部分更新失败 quarantine runtime；
- `max_stale_policy_versions >= 1`，否则拒绝把 strict-equivalent run 命名为 continuous；
- 不容忍 off-policy staleness 的算法继续 fail-fast，不偷偷降级。

## 4. 实施计划

### P0 — 先锁语义和 KILL-RISK

先锁 baseline 与真正的新失败路径：resolver 只保留 shared on-demand 与 disjoint resident 两类
派生；pre-launch rejection、removed-key rejection、完整 state parking 与 independent owner 应先写
失败测试。

1. `continuous + shared GPU` 在 actor launch 前失败；
2. `strict + gpu_pool=trainer` 唯一解析成 on-demand；
3. legacy `distributed.resources.rollout.memory_fraction` 给出带迁移说明的 removed-key hard error；
4. 第二个 on-demand rollout phase 开始前，optimizer/EMA 已创建且必须被 park；
5. streaming accumulation 在已经存在 live gradients 时跨一次 on-demand rollout phase，
   梯度逐值不变；
6. continuous blocking fake 证明 trainer main loop 被占用时 producer owner 仍前进；
7. 在真实 backward 窗口跑 thread-vs-Ray-actor KILL-RISK spike，先选择 continuous owner
   进程模型，再进入 P4。

增加一次性 GPU probe，覆盖：

```text
train one optimizer step
-> park trainer
-> wake/generate/sleep rollout
-> restore trainer
-> train second optimizer step
```

Probe 必须记录每个 phase 的：

```text
allocated_bytes
reserved_bytes
device_free_bytes
host_rss_bytes
park_s / resume_s
```

Parking 后不能要求 CUDA context 绝对归零；driver context 与库基线可能不可迁移。每个支持的
backend/strategy 必须由 probe 建立可解释的 residual baseline/上限，长期合同在超限时 raise。
Residual snapshot 的每个字段必须被该验证或控制流消费，不能只进入日志。

它要单独验证 `torch.optim.AdamW`、bitsandbytes `AdamW8bit`、EMA、`ref_model` 与
GradScaler。这里分开两类资产：

- 一次性 feasibility spike 只回答 bitsandbytes/CuMem/strategy 能否 round-trip，结论写回
  sprint 后删除脚本、数据和输出；
- 多轮 park/wake、live-gradient round-trip 与一卡/两卡验收是长期 acceptance harness，放在
  `tests/e2e/` 与对应 canonical config 路径并保留；不要放进属于 standalone checkpoint
  evaluator 的 `vrl/scripts/eval/`。

不支持 CPU round-trip 的 backend/strategy 直接形成 capability gate；不能为了让 on-demand
probe 变绿而给默认路径加经验 fraction，也不存在 resident fallback。该 topology 必须在
preflight 失败，或改用 disjoint GPUs。

### P1 — 收缩 topology/config，先消灭假 continuous

#### 删除

- `RolloutOrchestrationConfig.require_separate_gpus`；
- `ContinuousRolloutSchedule.require_separate_gpus` 与 builder 透传；
- public `distributed.resources.rollout.memory_fraction`，以及
  `DistributedResourceConfig` / `ResolvedDistributedResources` 的
  `rollout_gpu_memory_fraction`；
- `persistent_colocated` 的 cap-based lifecycle 派生与相关 validation；
- launcher 的 `gpu_memory_fraction`、worker 的 `cap_cuda_memory_fraction(...)` 调用、
  `cap_cuda_memory_fraction` 本体及只验证该链路的测试；
- runtime `_rollout_memory_fraction()`、chunk-probe fraction 参数/日志与相关 feature tests；
- `online_grpo_ocr_single_gpu_async_debug.yaml`（已删除）；不把它重命名成 resident harness。只有确实验证
  长期 phase lease 时，才新建 shared on-demand acceptance config；
- 只验证 `require_separate_gpus` escape hatch 的 config/schedule tests；
- E2E common override 里的 `trainer.rollout_orchestration.require_separate_gpus=false`。

#### 保留与收紧

```text
strict + trainer_devices intersects rollout_devices
    -> rollout.mode = on_demand; release_rollout_before_train = true

strict + no trainer/rollout intersection + no rollout/reward intersection
    -> rollout.mode = resident

strict + rollout_devices intersects reward_devices
    -> release_rollout_before_reward = true
    -> rollout.mode = on_demand

continuous + trainer/rollout intersection
    -> configuration error before model/Ray launch

continuous + no trainer/rollout intersection + no rollout/reward handoff
    -> rollout.mode = resident
```

`rollout.mode` 不能只从 trainer/rollout 轴推导。现有 source of truth 是：

```text
rollout_on_demand = release_rollout_before_train OR release_rollout_before_reward
```

Preflight 直接消费 `RayLifecyclePlan.handoff`，不要复制这组布尔公式。删除 role cap 后，
`samples_per_chunk:auto` 只探测当前 rollout phase 可用的整段容量；shared on-demand 与 dedicated
resident 都在各自 active phase 拥有这份容量。Backend-specific KV/cache budget 继续由 engine
配置消费，不再通过通用 role fraction 间接传递。

新增 `validate_rollout_schedule_topology(orchestration, resolved_resources)` 这类纯 preflight
边界：在 `resolve_distributed_resources()` 后、trainer bundle/placement/Ray launch 前调用，只读
resolved plan，不在 schedule/runtime 再用 raw device sets 重推导。`ContinuousRolloutSchedule`
仍保留 driver/reward offload capability backstop，并把现有 runtime colocation guard 改成无条件；
删掉的是手动 override，不是 safety check。

保留 `GenerationRuntime.is_colocated()`、`RayGenerationRuntime.is_colocated()` 与
`RolloutRuntimeCoordinator.runtime_is_colocated()`：移除 escape hatch 后，continuous schedule 无条件
消费这条协议，作为绕过 online preflight、直接构造 schedule 时的 resident-colocation backstop。
它不选择 lifecycle，也不重算 device sets；runtime 只携带 resolver/launcher 交下来的 verdict。
这组 thin methods 提供真实 protocol/adapter boundary，不是 dead wrapper。

Keep `ResolvedDistributedResources.colocated` and runtime
`requires_driver_model_offload`: they drive launcher/runtime placement and
trainer parking. The former `RayGenerationConfig.allow_driver_gpu_overlap`
mirror was deleted; `RayGenerationConfig.resources` is required and every
consumer now reads the resolved plan directly. Keep
`requires_driver_model_offload` on the `GenerationRuntime` protocol so the
coordinator consumes the behavior capability without a `getattr` fallback or
re-deriving raw topology.

对旧配置给 hard error 和两条明确迁移：

```text
one GPU / shared phase lease:
  use strict_on_policy + gpu_pool: trainer
  remove memory_fraction and require_separate_gpus

real continuous:
  expose a disjoint rollout GPU pool; remove require_separate_gpus
```

旧的 continuous shared 配置不是等价字段迁移：必须显式选择 strict shared phase lease 或增加
disjoint rollout GPU。Legacy role `memory_fraction` 必须给 targeted hard error，不能静默忽略。
A future engine-local KV/cache budget is distinct from a role lifecycle
selector. The former `PipelineStageRuntimePolicy.memory_fraction` had only a
test reader and no runtime consumer, so it was deleted with the test-only
physical-stage seam. Reintroduce such a budget only from a real engine
consumer's source of truth, never as a placeholder field.

### P2 — 完整 trainer memory lease

把 trainer-owned state 的 parking 放到 trainer/strategy 边界，不让 rollout coordinator 继续
猜 `_optimizer`、`_ema`、`ref_model` 的内部布局。

在 `Strategy` 增加真实硬件边界：

```python
def park_training_state(self, state: TrainingMemoryState) -> None: ...
def restore_training_state(self, state: TrainingMemoryState) -> None: ...
```

`TrainingMemoryState` 只携带真实被管理对象：model、optional ref model、optional optimizer、
optional EMA、optional GradScaler、device。它不是新的 lifecycle FSM，也不持有 rollout
runtime、queue 或 policy version。该对象必须由 trainer-owned live getter 在**每次进入 phase**
时构造，不能在 trainer 初始化时缓存：optimizer/EMA/GradScaler state 都可能 lazy 建立。Getter
还要按对象 identity 去重（例如 `ref_model is model`），避免同一 tensor tree 被搬两次。每个
字段都必须被 strategy parking 行为消费，不能成为只被日志/测试读取的 dead field。

具体 wiring 由 `OnlineTrainer` 向 schedule/coordinator 传入
`training_state_getter=self._training_memory_state`（或等价 park/restore callback）；coordinator
不读取 trainer 私有字段名。Getter 每次调用读取当前 `_optimizer` / `_ema` / `_grad_scaler` /
`ref_model`，而不是把初始化时的 `None` 快照交给 schedule。

实现要求：

- `SingleProcessStrategy`：model/ref model、optimizer tensor tree、EMA、GradScaler 全量 CPU
  round-trip；
- streaming accumulation 会在 optimizer step 前多次执行
  `collect -> backward -> collect`，因此 phase handoff 可能遇到 live gradients；parking 必须把
  gradients 与参数一起无损搬运并逐值恢复，不能要求它们提前清零；
- `DDPStrategy` / `FSDPStrategy`：实现 collective-safe parking，或在 shared topology 下
  capability fail-fast；不能继承 single-process 实现假装支持；
- bitsandbytes optimizer 必须由 P0 probe 证明 round-trip；失败则 shared-GPU + 8-bit optimizer
  显式不支持；
- restore 任一步失败，禁止进入 backward；保留原始异常并执行 terminal cleanup；
- parking/resume 幂等只用 adapter 自身一个状态，不引入 operation tickets/waiter map。

`RolloutRuntimeCoordinator` 接收 live training-state getter，并只提供 trainer parking 与 runtime
activation 的协议操作；`StrictOnPolicyRolloutSchedule.next_iteration()` 继续在一个函数内从上到下
拥有完整顺序：initial weight sync、trainer park、rollout activate、collect、rollout offload、
trainer restore、backward、next weight stage。Disjoint strict 仍走同一 schedule 形状，但
topology-derived parking 操作为 no-op；shared strict 永远执行完整 parking，`continuous` 永不调用
这些操作。

不要新增只有 strict schedule 一个 caller 的 `rollout_phase` context manager，也不要为已删除的
inline fixed eval 新建 callback。现有 `ensure_initial_weights()` 仍必须在第一次 training generate
前完成；这是 strict schedule 自己的权重合同，不需要再造一个 eval-specific initial commit。

### P3 — 把 rollout sleep/wake 升级成完整 backend contract

保留 `RayGenerationRuntime.activate()/offload()` 与 desired-policy coalescing；它们当前的职责
边界是正确的。补强 worker contract：

- validate the concrete loaded executor/model during worker startup; do not add
  another family capability flag for parking support;
- sleep 后返回结构化 residual snapshot，而不是仅凭 RPC 成功；
- residual snapshot 与 backend-declared baseline/上限比较，超限直接判定 parking 失败；
- CuMem 路径覆盖 build-time 与 generation-time 长寿命 allocation；
- CPU fallback 必须逐 family 验证 model + frozen components + backend caches；未验证 family
  fail-fast，不静默降级；
- wake 后先恢复 memory，再安装缺失的 desired policy version；
- sleep/wake failure 关闭 admission 并 quarantine，不能把未知显存状态交给 trainer；
- host-memory telemetry 纳入 phase metrics，防止 GPU OOM 被换成 pinned/host OOM。

不要把 `samples_per_chunk:auto` 删除或改成跨角色 arbiter。它只负责**当前 rollout phase 内**
选择工作量，不能证明另一个进程稍后 backward 的峰值。删除通用 role cap 后，它按当前 phase
容量探测；SGLang/KV/cache 等 backend-specific budget 仍由对应 engine 配置约束。

### P4 — 独立 continuous owner

现有 `RolloutScheduler`、producer、ready queue、consumer、staleness gates 全部保留；缺的是让
它们不依赖 trainer event loop。

首选实现是一个由 `ContinuousRolloutSchedule` 实例拥有的 dedicated thread + asyncio loop：

```text
trainer/main loop              continuous owner loop
-----------------             ---------------------
next_iteration request   ---> producer/queue/consumer
CPU policy snapshot      ---> pause/commit/resume
iteration + metrics      <---
shutdown request         ---> stop admission, join, teardown
```

约束：

- 不是 module-global singleton；一个 schedule 拥有一个 owner；
- owner loop 独占 continuous collector/runtime 的 async 操作；main loop 不跨 loop 直接调
  `activate/update/offload/shutdown`；
- trainer rank 在 main thread 完成 trainable-state export/collectives，再把 immutable CPU
  snapshot 交给 owner commit；
- initial weight sync 也遵守同一拆分：main thread 先 export immutable CPU snapshot，owner 再
  commit；`ContinuousRolloutProducer.start()` 不得在 owner thread 调用可能包含 DDP/FSDP
  collective 的 `sync_state_getter()`；
- thread communication 使用 future/command boundary，不共享 asyncio Task；
- checkpoint restore 触发的 `reset()` 也必须变成 owner command；不能从 main thread 直接取消
  owner-loop task；
- shutdown 停 admission、join in-flight policy、关闭 queue、join thread；
- owner 是 continuous collector/runtime 的 terminal owner：它完成 runtime/collector teardown 后
  main cleanup 只 join/读取结果，不能再跨 loop 二次调用 `collector.shutdown()`；
- cancellation of one waiter 不取消 owner terminal cleanup；
- owner error保存第一个 root cause并向 `next_iteration`/commit caller fail-fast。

这一步借 MILES “独立 worker loop”的形状，但不复制它的 global worker、untyped queue 或轮询
错误处理。Thread vs CPU-only Ray actor 必须在本 track 的 P0 用**真实 backward** KILL-RISK
spike 先定：若 dedicated thread 因 GIL/driver contention 不能在 backward 窗口持续
admit/harvest，直接实现 Ray actor，不先落一版已知过不了 overlap gate 的 thread owner。

### P5 — 唯一 continuous weight commit

Continuous owner 的 commit 顺序固定为：

```text
pause producer admission
-> if no versioned slots: drain in-flight
-> update every rollout worker
-> require all ACK(policy_version)
-> publish committed version
-> purge items outside staleness window
-> resume admission only after successful commit
```

有 versioned trainable-state slots 时保留 VRL 已有的 non-draining 优势：旧 request 继续绑定旧
slot，新 request 使用新 slot。没有 slots 时才采用 MILES 式 draining/retract 等价语义。

事务规则：

- policy version 单调；
- worker/weight-sync protocol 必须返回实际 installed `policy_version`；`ray.get(refs)` 返回且
  payload 为 `None` 只证明 RPC 没抛错，不算版本 ACK；
- partial multi-worker update 不发布 committed version；
- partial failure 保持 admission closed、quarantine 整个 runtime并进入 terminal cleanup；
- queue item 始终保留 submit-time version；
- future version、mixed-version group、超 staleness window 一律 fail/discard，不能偷换 latest；
- `continuous.max_stale_policy_versions` 必须 `>=1`；算法能力 gate 继续生效。

把 `ContinuousRolloutConfig.max_stale_policy_versions` 的 typed default 与 cross-node preset
从 `0` 改成 `1`，或移除默认并要求所有 continuous recipe 显式设置；不能只在 DoD 写
`>=1` 而让默认继续构造 strict-equivalent background queue。

迁移必须闭环修改 `ContinuousRolloutConfig.__post_init__`、continuous factory 的 zero-window
warning 分支、algorithm capability error（不再建议 continuous 设 `0`，而是建议切回
`strict_on_policy`）、base continuous preset、cross-node debug preset 及 zero-window config /
schedule / trust-region tests。`StalenessPolicy(0)` 若只为 scheduler mechanism 单测保留，必须在
注释中明确它不是 production-config 可达状态；否则一并收紧。

本 sprint 不改变 object-store/NCCL transport 选择；那属于
`SPRINT_weight_sync_transport_seam.md`。这里固定的是 commit ordering 和 ownership。

### P6 — 验证与完成门槛

#### CPU / deterministic tests

- topology matrix 覆盖三个主要 trainer/rollout shapes，以及 strict disjoint + reward-shared 的
  on-demand 子形态；`continuous + shared` 必须 rejected；
- topology matrix 单独覆盖 reward 轴：rollout/reward 共卡会要求 mid-iteration handoff，
  continuous 必须拒绝，不能只检查 trainer/rollout intersection；
- removed `require_separate_gpus` 与 role `memory_fraction` 都给出 targeted hard error；
- 绕过 config preflight、直接构造 resident-colocated runtime/schedule 时，unconditional runtime
  backstop 仍拒绝 continuous；不存在测试用 escape hatch；
- trainer model/optimizer/EMA/ref/GradScaler/live gradients 的 park/restore round-trip 内容逐值一致；
- on-demand strict 调用序列：

```text
trainer.park
< rollout.activate
< generate(vN)
< rollout.offload
< trainer.restore
< backward
< desired_policy(vN+1)
```

- `samples_per_chunk:auto` 按当前 rollout phase 的容量探测，不再接收通用 role cap，也不存在
  `_rollout_memory_fraction()` 不可达 getter；
- next activation 在任何 generate 前安装 `vN+1`；
- strict 第一次 training generate 前仍先完成 `ensure_initial_weights()`；inline fixed eval 的旧
  bypass tests 随 companion sprint 删除，不在这里重建；
- streaming accumulation 在两次 collect 之间保留的 live gradients 经 park/restore 后逐值一致；
- shared reward 的顺序固定为 trainer park -> rollout generate -> rollout park -> reward
  score/park -> trainer restore；
- on-demand collect/reward/update/offload 异常时 trainer 最终恢复或 run terminal-fail，绝不让
  两侧 GPU state 同时 active；
- continuous owner 在 trainer main event loop 被 blocking fake 占用时仍 admit/harvest；
- commit pause/drain-or-slot/update/resume 顺序；
- worker update protocol 返回实际 installed `policy_version`；partial/mismatched ACK 不发布
  committed version、不恢复 admission；
- stale/future/mixed versions 的 discard/fail；
- shutdown 单飞、owner thread join、无 queue/task/actor 残留。

#### Real one-GPU acceptance

使用 shared topology（role `memory_fraction` 已删除），至少完整跑：

```text
initial rollout
-> train step 1 (optimizer/EMA state created)
-> rollout 2
-> train step 2
-> rollout 3
-> checkpoint
-> clean shutdown
```

验收：

- 每个 phase event 顺序正确；
- policy versions 单调且 rollout 使用预期 version；
- trainer 与 rollout 各自 phase peak 可以超过旧 0.55/0.45 切分，但从不同时 active；
- 无 CUDA OOM、无 stale active worker、无 actor/placement leak；
- host RSS/pinned-memory 峰值在日志中可见；
- 第二、三轮结果证明 optimizer state 建立后仍稳定，不能只交第一步 smoke。

#### Real two-GPU continuous acceptance

使用 disjoint trainer/rollout GPUs、`max_stale_policy_versions=1`：

- 记录 generate interval 与 backward interval，至少一个训练 step 有正 wall-clock
  intersection；
- interval 必须写 start/end timestamp，或在 backward 前后采样 producer counter；只有各 phase
  duration 不能证明区间相交；
- backward 期间 producer completed/submitted counters 前进；
- ready queue 在稳定段非永久清零；
- version lag 始终在配置窗口内；
- commit pause 只覆盖 weight commit，不覆盖整个 rollout request（versioned-slot backend）；
- continuous 全程不调用 trainer park 或 rollout sleep/wake；
- shutdown 后 owner、worker actors、placement resources 全部释放。

没有这组真实两卡证据，不能把 P4/P5 标记为“真 continuous 已完成”。

## 5. 配置迁移

### 单 GPU / shared phase lease

```yaml
defaults:
  - /base/rollout/orchestration/strict

distributed:
  resources:
    rollout:
      gpu_pool: trainer
```

不再出现：

```yaml
trainer:
  rollout_orchestration:
    require_separate_gpus: false
distributed:
  resources:
    rollout:
      memory_fraction: 0.55
```

两项都属于 removed configuration。旧 shared 配置删除它们并使用 strict phase lease；旧
continuous shared 配置必须改成 strict shared，或提供 disjoint rollout GPU。不会把
`memory_fraction` 静默解释成 engine/KV budget。

### 真 continuous

```yaml
defaults:
  - /base/rollout/orchestration/continuous

distributed:
  resources:
    rollout:
      gpu_pool: dedicated

trainer:
  rollout_orchestration:
    continuous:
      max_stale_policy_versions: 1
```

Local topology 在 model/Ray launch 前由 resolved device sets 证明 trainer/rollout disjoint；
cross-node 的 CUDA ordinal 不在同一物理命名空间，不能做假 set intersection，必须在 actor
launch 前用 node-level preflight 证明 head 不暴露 rollout GPU，并在 worker 启动后用既有
node/GPU identity 校验 placement。“先启动再由 schedule 猜”不算完成。

## 6. 应改 / 应留 / 非目标

### 应改变

- 删除 `require_separate_gpus` 与 same-GPU continuous escape hatch；
- 删除 public role `memory_fraction` 及其 resolved/launch/worker/probe 整条 cap plumbing；
- pre-launch topology validator 只读 resolved plan，continuous 拒绝所有 trainer/rollout overlap
  和需要 rollout mid-iteration parking 的 reward topology；
- shared strict 收敛为唯一 `on_demand` lifecycle；
- 完整 park trainer-owned 与 rollout-owned CUDA state；
- strict schedule 在一个控制流内拥有完整 phase order，trainer/strategy 与 generation runtime
  分别实现 parking mechanics；
- continuous producer 独立于 trainer event loop；
- continuous 只允许有意义的 staleness window 和 disjoint topology；
- 用真实一卡/两卡证据分别验收 phase lease 与 overlap。

### 应保持不变

- `RayLifecyclePlan`：topology -> role lease / phase handoff 的 typed source of truth；
- `RayGenerationRuntime.activate/offload/update_weights/shutdown`：runtime protocol boundary；
- `GenerationRuntime.is_colocated()` 与 coordinator adapter：continuous 的 unconditional runtime
  safety backstop；它们不参与 schedule selection；
- desired vs active policy snapshot：sleeping worker 的事务性 coalescing；
- terminal `lifecycle_fsm.py`：shutdown admission、首个 failure、terminal phase；
- bounded in-flight/ready/byte budgets、staleness scheduler、versioned slots；
- `RayGenerationLaunchInputs`：lazy launch 所需的 typed input boundary；
- family runtime/executor adapters：跨 family 一致性与 grep boundary；
- partial cleanup retry、shutdown single-flight、release/offload 幂等。

这些薄函数/文件有真实协议、lazy launch、framework adapter 或跨 family 一致性价值，不能为了
少几行 flatten。`runtime_is_colocated()` 的唯一 production caller 是 safety guard，但这是保留
single-caller helper 的 protocol-boundary 例外。`cap_cuda_memory_fraction()` 不属于该例外：它的
唯一 production caller 随唯一 producer 一起消失，且它吞异常的“hard cap”没有硬件合同；应与
launcher field 和 `_rollout_memory_fraction()` 一起删除。新增的 training memory seam 只有在真正
承载 model/optimizer/EMA/ref/GradScaler 与 strategy-specific parking 后才成立；不能建一个只转发
`model.to()` 的装饰 manager。

### ALL_CAPS 审计

保留 `_MB`、`_CPU`、`_MISSING`、观测时间阈值等：它们分别是单位、设备 singleton、sentinel、
观测边界。本文范围没有大型 ALL_CAPS 业务词表或重复 typed structure 的手写集合。

The former `PipelineStageRuntimePolicy.memory_fraction` was not worth keeping
merely because a future engine might use a similarly named budget. It had only
a field definition, range validation, and a test reader, so the field, its
assertions, and the physical-stage contract were deleted together. Do not
recreate that dormant field. A future engine-local budget must originate from
a real runtime consumer's source of truth.

### 非目标

- 不承诺单 GPU 上 rollout kernel 与 backward 真并发；
- 不保留通用 role-level resident cap；未来 engine budget 只有出现真实 producer、runtime consumer
  与硬件 acceptance 后，才从对应 backend source of truth 引入；
- 不用 auto chunk size 或 `mem_get_info()` snapshot 冒充跨进程 reservation；
- 不在本 sprint 做 NCCL/P2P weight transport；
- 不把 host queue byte cap 与 engine-local GPU budget 混为一谈；
- 不把 MILES 的 global thread example 原样搬进生产；
- 不删除 strict schedule 再在 continuous 内藏一个 phase-serial 大分支；明确的两种 schedule
  比一个根据 topology 偷换语义的类更容易维护；
- 不重新引入 `OperationTicket`、`QUIESCING`、通用 waiter/barrier FSM；
- 不在 schedule/runtime 外重新计算 raw device-set topology；preflight 与 runtime 都消费同一个
  resolved plan；
- 不把已删除的 inline fixed eval 重新接回 live trainer，也不为 standalone evaluator 增加
  continuous-owner command；
- 不改 family trajectory schema、algorithm loss 或 reward 数学；
- 不重写已完成 sprint 的历史，只给被取代文档加 superseded note。

## 7. 文件级实施清单

### Topology/config

- `vrl/ray/resources.py`
- `vrl/config/schema.py`
- `vrl/trainers/core/types.py`
- `vrl/rollouts/orchestration/schedule.py`
- `vrl/rollouts/orchestration/continuous/schedule.py`
- `vrl/generation/protocols.py`（保留 runtime topology backstop，并补齐 driver-offload capability）
- deleted physical-stage topology package（its test-only `memory_fraction` was removed; it is not an implementation target）
- `vrl/generation/ray/config.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/execution/worker.py`
- `vrl/utils/cuda_memory.py`

### Full phase lease

- `vrl/trainers/strategy.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/orchestration/rollout_runtime.py`
- `vrl/rollouts/orchestration/strict_on_policy.py`
- `vrl/rollouts/collector/core.py`
- `vrl/scripts/common/online.py`

### Continuous owner / commit

- `vrl/rollouts/orchestration/continuous/owner.py`（new, only if it owns real loop/state）
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/rollouts/orchestration/continuous/schedule.py`
- `vrl/rollouts/orchestration/continuous/scheduler.py`
- `vrl/trainers/weight_sync.py`
- `vrl/generation/ray/weight_sync.py`

### Config/tests/docs migration

- delete（已完成）
  `vrl/config/presets/experiment/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml`；它是
  same-GPU continuous 一次性 probe，不重命名成 resident harness；
- update `vrl/config/presets/base/distributed/ray_rollout_colocated_single_gpu.yaml`;
- update `vrl/config/presets/base/rollout/orchestration/{strict,continuous}.yaml`;
- update `vrl/config/presets/experiment/sd3_5/online_grpo_ocr_crossnode_debug.yaml`
  与 `ContinuousRolloutConfig.max_stale_policy_versions` 的 single source of truth；
- update resource/config/runtime/orchestration/online lifecycle tests found by:

```bash
rg -n "require_separate_gpus" vrl tests
rg -n "runtime_is_colocated|def is_colocated" vrl tests
rg -n "rollout_gpu_memory_fraction|gpu_memory_fraction|memory_fraction" vrl tests
```

第一个 grep 在迁移后必须零命中。第二个 grep 应只剩 protocol、runtime implementation、coordinator
adapter、continuous unconditional guard 与对应 tests；不能再出现 override branch。第三个 grep
用于完成删除审计：`rollout_gpu_memory_fraction` 与 launch-contract `gpu_memory_fraction` 在
production/tests 中必须零命中；generic `memory_fraction` 的任何剩余命中必须逐个证明是独立的
engine/config source of truth，不能来自 `distributed.resources.rollout`，也不能把测试-only reader
算行为消费者。

- add superseded notes to:
  - `docs/sprints/done/SPRINT_single_gpu_continuous_rollout_debug.md`;
  - relevant single-GPU claims in
    `docs/sprints/parked/SPRINT_async_rollout_train_overlap.md`;
- preserve completed docs as history; do not rewrite old result sections to pretend the old design never existed.

## 8. Definition of done

本 sprint 只有同时满足以下条件才算完成：

1. production/config/test behavior 中不存在 `require_separate_gpus`、`rollout_gpu_memory_fraction`
   或 launch-contract `gpu_memory_fraction` 字段/分支；这些名字只允许出现在 targeted removed-key
   rejection 与 migration tests 中；legacy role `memory_fraction` 给 targeted hard error；continuous
   runtime colocation backstop 无条件生效，direct-construction tests 也不能绕过；
2. strict shared 唯一解析为 on-demand；strict/continuous disjoint 解析为 resident；continuous
   shared 在 model/Ray launch 前失败；
3. rollout/reward handoff 仍参与 `rollout.mode`，continuous 对 mid-iteration reward parking
   fail-fast；
4. local continuous topology 在 pre-launch 静态 disjoint；cross-node 通过 node-level preflight
   与 worker identity 校验，而不是混用 CUDA ordinal；
5. strict schedule 在一个控制流内完成 initial weight sync、trainer park、rollout activate/collect/
   offload、trainer restore 与 train；不新增 single-caller `rollout_phase`，也不重新接入 inline eval；
6. optimizer/EMA/ref/GradScaler 已创建后的真实单卡 on-demand 多轮 run 无 role cap、无 OOM、
   无 resource leak；
7. 真实两卡 run 给出 rollout/backward wall-clock overlap 证据；
8. continuous weight commit 验证每个 worker 返回的 installed version；全 ACK 后才发布/恢复
   admission，partial failure 保持 closed 并 quarantine；
9. owner 独占 continuous runtime async lifecycle，reset/shutdown/collector cleanup 无跨-loop
   二次调用，owner thread/actor、queue、task、placement 全部释放；
10. test-only dead `PipelineStageRuntimePolicy.memory_fraction` 已删除；
11. existing strict, continuous scheduler, Ray cleanup, config-load suites 全绿；
12. ruff/format/type-visible imports 全绿，dirty worktree 中与本 sprint 无关的用户改动保持不变。

Track A 以 1–6、10–12 为自己的 merge gate；Track B 以 1、3–5、7–9、11–12 为自己的
merge gate。整个 umbrella 只有两组 gate 都有对应硬件证据时才标记 done。

## 9. 参考

### MILES / slime upstream

- [MILES colocate defaults / explicit override](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/utils/arguments.py#L2623-L2651)
- [MILES synchronous phase sequence](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/train.py#L89-L121)
- [MILES async reject / overlap loop](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/train_async.py#L18-L94)
- [MILES disaggregated placement](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/ray/placement_group.py#L90-L121)
- [MILES trainer / rollout fractional Ray claims](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/ray/placement_group.py#L133-L147)
- [MILES rollout actor fractional Ray claim](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/ray/rollout/server_group.py#L80-L123)
- [MILES resident-sync recipe](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/scripts/run_gemma_4_31b.py#L120-L145)
- [MILES FSDP transactional weight-update order](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/miles/backends/experimental/fsdp_utils/update_weight_utils.py#L48-L90)
- [MILES topology / headroom documentation](https://github.com/radixark/miles/blob/5e392cf2cef761d4508e81b2a8f4c4b191d7a73c/docs/user-guide/training-script-walkthrough.md#L298-L350)
- [slime colocate defaults / explicit override](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/slime/utils/arguments.py#L1882-L1904)
- [slime synchronous phase sequence](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/train.py#L49-L91)
- [slime async reject / overlap loop](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/train_async.py#L9-L70)
- [slime resident-sync recipe](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/scripts/run-qwen3-4B-amd.sh#L115-L157)
- [slime colocate headroom documentation](https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/docs/en/get_started/usage.md#L138-L150)

### VRL

- `vrl/ray/resources.py:149-163,283-365,1158-1271`
- `vrl/config/presets/experiment/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:27-43`（历史路径，已删除）
- `vrl/config/presets/experiment/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml:48-54`
- `vrl/config/presets/experiment/sd3_5/online_grpo_ocr_fsdp_2x1_fullparam.yaml:53-57`
- `vrl/config/presets/experiment/cosmos_predict2_5/online_nft_kling_video_reward_fsdp_2x1.yaml:71-76`
- `vrl/rollouts/orchestration/rollout_runtime.py:25-149`
- `vrl/rollouts/orchestration/strict_on_policy.py:22-88`
- `vrl/rollouts/orchestration/continuous/schedule.py:39-155,277-301`
- `vrl/rollouts/orchestration/continuous/producer.py:85-190`
- `vrl/generation/protocols.py:51-79`
- `vrl/generation/ray/launcher.py:255-272`
- `vrl/generation/ray/runtime.py:98-137,238-306,317-379,381-457,633-690`
- `vrl/generation/execution/worker.py:61-180,307-496`
- `vrl/utils/cuda_memory.py:121-144`
- `docs/sprints/info/SPRINT_ray_generation_engine_map.md`（current canonical generation path and deleted-seam correction）
- `vrl/trainers/online/trainer.py:500-588,663-696,1021,1438`
- `vrl/trainers/core/types.py:157-207`
- `vrl/trainers/strategy.py:25-87,89-150,189-337,351-487`
- `vrl/trainers/weight_sync.py:17-75`
- `vrl/generation/ray/weight_sync.py:12-66`
- `vrl/scripts/common/online.py:720-878,962-1025`
- `vrl/rollouts/collector/core.py:95-108,120-170,229-242`
- `tests/ray/test_resources.py:1121-1153`
- `tests/rollouts/orchestration/continuous/test_schedule.py:325-359,470-542`
- `tests/generation/pipeline/test_pipeline_contracts.py:25-66`
- `tests/config/test_load_all_experiments.py:332-355`
- `docs/sprints/done/SPRINT_remove_inline_fixed_eval.md`

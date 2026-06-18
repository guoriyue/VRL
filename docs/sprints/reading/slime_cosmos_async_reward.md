# Async rollout/train/reward — slime vs cosmos-rl（reading）

类型：cross-system 源码对照（2026-06-17）。file:line 相对各 repo root（`~/Desktop/slime`、
`~/Desktop/cosmos-rl`），我们的相对 `vrl/`。index：[slime_cosmos_study_index.md](slime_cosmos_study_index.md)。

## 1. 循环结构

### slime —— 1 步双缓冲，显式 drain barrier
- **同步 driver**(`train.py:71-99`):严格串行,每步 blocking `ray.get`。`generate → offload rollout →
  train → offload train → onload weights → update_weights → onload KV`。**零重叠。**
- **异步 driver**(`train_async.py:35-43`):`ray.get` 训练第 N 步**之前**先 `generate.remote(N+1)`,
  下一轮 rollout 与当前训练重叠。
- **唯一 barrier**(`train_async.py:69-73`):每次权重推送前 drain 在飞的 generate future
  (`rollout_data_curr_ref = ray.get(x)`,nulls it,然后 `update_weights()`)——注释:"prevent update
  weight in the middle of generation"。
- **硬约束**:`train_async.py:11` `assert not args.colocate`——async **只在 disaggregated 跑**。
- 同步频率:`--update-weights-interval`(默认 1,`utils/arguments.py:427-431`)。

### cosmos-rl —— 控制反转，多在飞 + 版本门
- rollout worker 跑**自己的主循环**(`rollout_control.py:1755-1823`),不停从 controller 拉 prompt;
  policy replica 并行训练。重叠是 disaggregated 默认。
- 不用硬 drain,用 **freshness gate**(`rollout_control.py:1807-1815`:`weight_version > current +
  allowed_outdated_steps` 就跳过生成)+ controller 背压(`controller.py:285-342`:软 clamp 到
  `outdated_rollout_fetch_batch_size`、硬 `return [], True` 当超 `max_inflight_steps`)。
- version 盖戳:`controller.py:273-275` `weight_version = current_step + pending // rollouts_per_global_batch`。
- async vLLM 更进一步:在 `generate()` **内部**每 N engine step 消费 weight-sync 命令
  (`COSMOS_ROLLOUT_STEP_INTERVAL` 默认 100),`WeightSyncThread` 在**独立 CUDA stream** 上写 buffer
  model,生成边界才 copy 到 live 权重——真正的"同步/生成"重叠(slime 没有)。

## 2. Reward 在哪算（核心问题）

**两家都在 rollout 侧、绝不在训练 GPU；但机制完全不同。**

### slime —— 内联在 rollout 的 asyncio 协程
- `generate_and_rm`(`sglang_rollout.py:239-302`):semaphore 内 `await generate()`,**semaphore 外**
  `await async_rm()`(`:299-301`)——已完成样本的打分和在飞样本的生成重叠。
- 规则类 reward(math/dapo/f1/gpqa,`rm_hub/__init__.py:55-93`)= **协程里的同步 Python,无进程无 GPU**;
  只有 `remote_rm`(`:34-52`)是 async aiohttp 打外部服务。
- group 模式(`--group-rm`)在组内全部完成后 `batched_async_rm`(`:310-342`)。
- GRPO 组归一在 RolloutManager CPU 上(`rollout.py:655-680`),DP split 前。
- **没有独立 reward actor / GPU。**

### cosmos-rl —— 独立 RewardDispatcher + worker 池（注意有两条路）
- **视频/diffusion 路**(我们关心的):reward 在 **rollout worker 进程**内,由 `RewardDispatcher`
  (`reward/dispatcher.py`,worker `__init__` 起,`rollout_control.py:191-220`)算。生成后立刻
  `enqueue_rewards_cal`(`rollout_control.py:1913`),**之后**某轮主循环 `dequeue_rewards_cal()`
  (`:940`)非阻塞取——下一批生成时上一批 reward 在后台算。**关键**:`dispatcher.py` 文本用
  `ProcessPoolExecutor`,**`non_text`(视频)用 `ThreadPoolExecutor`**(张量/视频不走 pickle);
  且只有 `should_report` rank(`tp_coord[0]==0` 且 PP-last,`:250-252`)算 reward + 发 HTTP。
- **LLM/GRPO 文本路**(对照):reward 在 **controller 进程**的 `dispatcher/algo/` 里同步算
  (`run_web_panel.py:434-507`,POST /rollout 到达后,HTTP handler 线程内),advantage 也在 controller 算。
- remote reward(`use_remote_reward=True`):completion 打到独立 reward_service,controller/worker 阻塞等返回。

> **结论**:slime 把 reward 折进生成协程(asyncio 重叠,主要 CPU 规则)。cosmos 把 reward 做成**一级流水
> 阶段 + 自己的池**,队列与生成解耦,**专为 GPU 解码的视频 reward 在训练 GPU 之外设计**。我们要做视频
> reward,cosmos 这条对路。

## 3. 对照

| | slime | cosmos-rl |
|---|---|---|
| 重叠 | 1 步双缓冲(`train_async.py:35-43`) | 多在飞,`allowed_outdated_steps` 深度 |
| barrier | 硬 drain in-flight(`:69-73`) | freshness gate + 背压(无硬 drain) |
| 权重同步隐藏 | 无(barrier 串行) | WeightSyncThread 独立 stream + buffer swap |
| reward 位置 | rollout 协程内联(CPU 规则) | rollout-worker RewardDispatcher 池(视频)/ controller(文本) |
| reward 与生成 | asyncio 重叠 | 队列解耦,跨批后台算 |
| 独立 reward 资源 | 无 | 有(进程池/线程池/远程服务) |

## 4. 对我们
- continuous barrier 已对(`vrl/rollouts/orchestration/continuous/schedule.py:115-119`)。
- **reward 抄 cosmos 的解耦 dispatcher + `non_text→ThreadPoolExecutor`**;我们 reward-execution
  `inline|pool` 后端已能对上(见 `project_reward_execution_backend`)。

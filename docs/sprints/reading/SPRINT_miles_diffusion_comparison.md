# 阅读：miles / miles_diffusion 与 VRL 的对照（2026-09-14）

来源：`/home/ubuntu/miles`（radixark/miles @2ef603a）、`/home/ubuntu/miles_diffusion`
（@ebd55fc，30 个提交，17k 行，62 个测试文件 / 192 个测试）。VRL 同日：110k 行，
384 个测试文件 / 3320 个测试，27 个模型家族，76 个 experiment recipe，24 个 reward。

## 进程模型（本次对照的起因：driver 侧 GIL 争用）

| | miles_diffusion | VRL |
|---|---|---|
| driver | 纯控制平面，只 `await` actor | torchrun rank 既训练又跑 producer |
| 训练 | `TrainRayActor`（FSDP2 + USP） | rank 进程内 FSDP2/DDP/single |
| rollout | sglang-diffusion 服务进程（引擎侧 TP/SP、CUDA graph） | 自研 denoise loop，Ray actor，eager（SD3.5 512px 发射绑定，SM 44%） |
| 反序列化 | msgpack 原始字节 + `RolloutImageResponseParserActor` 池 | driver 内 pickle |
| reward | Ray actor 池（PickScore GPU 池、OCR CPU 池） | 托管服务子进程 + park/wake 租约（2026-09-14 落地） |
| 流水 | 每 microgroup 一个 asyncio task：generate → deserialize → reward | strict 串行 / continuous prefetch |
| 大 reward 与 trainer 共卡 | 无（独立 reward GPU，或常驻小模型 colocate 0.05 GPU） | CuMem park/wake 时分（4 卡跑 7B HPSv3） |

## 他们领先的

1. 进程模型本身：计算从不进 driver，我们刚花一天证明并修补的问题在那里不存在。
2. rollout 引擎：sglang-diffusion（引擎侧并行与优化），并用 monkey patch + deterministic
   模式压 train/rollout 不一致；我们的 parity 门在 compile / batch>1 下都失败。
3. 训练侧 USP（Ulysses × Ring，来自 diffusers `_cp_plan`）；我们只有 rollout 侧手写 Ulysses（SD3）。
4. LoRA 权重经 CUDA IPC 直接同步到 colocated 引擎。
5. 已验证的多节点配方（2×8 + 1 reward GPU）、dashboard 时间线。

## 我们领先的

1. 广度：27 家族（含 AR token 模型 janus/llamagen/emu3/glm_image 和 cosmos 2/2.5/3、hunyuan、
   mochi、magi、H3），76 recipe，24 reward（Kling、VideoCon physics、UnifiedReward、HPSv3、机器人 IDM）。
2. 算法：GRPO / flash-GRPO / continuous GRPO、DiffusionNFT、V-GRPO、DPO、TIS/RS 精度修正、
   replay parity 与 drift 门；他们是 Flow-GRPO、NFT、SFT。
3. GPU 时分：4 卡跑 Wan + 7B reward 的 phase-cycled 布局，他们要第 5 张卡。
4. 类型化配置、resume 的 RNG 逐位一致、run evidence、测试量级（17×）。

## 结论

不是"整体学"而是"进程模型学、其余保留"。按收益顺序：

1. artifact 材料化与反序列化出 driver（parser actor 池的形状），
   见 `planned/SPRINT_reward_service_isolation.md` 之后。
2. sglang-diffusion 作为 rollout provider：`parked/SPRINT_sglang_diffusion_execution_provider.md`
   的触发条件应重估——miles_diffusion 已证明它能跑 Wan2.2/LTX/Cosmos3。
3. 训练侧 USP（视频大模型）。
4. driver 拆成控制 + 训练 actor（`TrainRayActor` 形状），最后做。

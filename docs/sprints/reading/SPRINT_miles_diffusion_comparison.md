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

## 附录 A（2026-09-14）：训练侧 USP 与 parity 工具链的逐行对照

### A.1 训练侧 USP（对应本仓库 WS-C）

| 维度 | miles_diffusion | VRL（WS-C 落地形状） |
|---|---|---|
| 配置 | `--sequence-parallel-size` + `--ulysses-degree`（0=auto，ring 派生）| `fsdp.mesh: [dp_shard, cp]` + `fsdp.context_parallel.{ulysses_degree, ring_degree}` |
| mesh | `world._unflatten` 成 `(dp, sp)`，再 `sp → (ring, ulysses)`；**SP rank 在 FSDP 分片轴内** | FSDP 仍在整个 world 上 1D 分片；另建 3D `("dp_shard","ring","ulysses")` 给 diffusers |
| 注意力 | 复用 diffusers `_cp_plan` 元数据，但自写 hook + 自写 `usp_attention`（all-to-all + torch 私有 ring） | 直接调 diffusers `enable_parallelism`（同一份 plan，官方 Ulysses/ring 实现） |
| 样本分发 | rollout manager 只按 dp 切，SP 同组按 `dp` mesh local rank 取**同一个 Ray object** | leader 采样、组内 gloo 广播（VRL 的 rollout 是 rank 本地 Ray runtime） |
| loss / log-prob | 边界 gather 后全序列，loss 代码不感知 SP | 同 |
| 梯度 | gather 的 backward 先在 SP 组 all-reduce，配合 FSDP 的 1/(dp·sp) 平均 | diffusers gather backward 只取本地分片；`FSDPStrategy.backward` 把 loss 乘 cp，配合 FSDP 的 1/(dp·cp) 平均。数学等价，少一次集合通信 |
| 裁剪 | 普通 `clip_grad_norm_` + `full_tensor()` | 普通 DTensor 感知 `clip_grad_norm_` |
| 指标 | 只在 dp 组归约；`grad_norm` 标记 REPLICATED | world 归约；CP 同组值相同，max/all_true 不变，sum 类只用作比值 |
| rollout SP 与训练 SP | 完全解耦（引擎 `--sglang-sp-degree`，`tp*sp == gpus_per_engine`） | 解耦；follower 的 rollout 引擎目前空转，后续可与 `gpus_per_engine == cp` 配对 |
| 一致性守卫 | `validate_same_microbatch_counts_across_train_ranks` 在 dp×sp gloo 组 all-gather 微批数 | `_TrainingMicrobatch.plan_balanced` 已用 `max_int` 拉平 |

### A.2 rollout/train parity（对应 WS-F 与 batch-16 / compile 漂移问题）

它们的原则：**不放宽 ε，把两个前向做到一致**。`--diffusion-clip-range` 默认 1e-4（与我们相同），没有 TIS/mask 修正。

1. `--diffusion-recompute-old-log-prob`：训练侧用更新前权重重算 old log-prob；第 0 个优化窗口直接
   `old = new.detach()`（ratio 恒 1，零额外前向）。这正是 rollout 核 vs 训练核漂移的逃生口。
2. rollout microgroup 与训练 micro-batch 形状对齐（`train-dp-split-mode contiguous`），承认 batch 形状
   决定数值（`patch_qwen_image.py` 的 `split_seqs` 连续化就是 batch>1 才出现的漂移源）。
3. 引擎侧 `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true`（batch 不变 matmul）；rollout
   路径**不用** torch.compile 与 CUDA graph。
4. 融合核的中途 bf16 舍入：Wan 每个 norm/residual 站点 ~3e-3 相对误差，30 块累积正好落在我们测到的
   0.014–0.03；修法是两侧在**同样的位置**舍入（fp32 LN/residual/RoPE + 一次 `.type_as`）。
5. 边界 dtype 显式声明（`input_dtype_policy`：latents/cond/timestep），timestep/sigma 网格保持 fp32。
6. 诊断模式：`--diffusion-debug-mode --debug-skip-optimizer-step` + `MILES_VERIFY_WEIGHT_SYNC=1`，引擎
   回传每步 `model_output`，训练侧记 `model_output_{mean,max}_abs_diff / rel_max`、
   `log_prob_mean_abs_diff`、`ratio_abs_minus_1`、`approx_kl`、`clipfrac`；把权重同步、前向路径、
   优化器三种漂移分开看。验收带：H3 mean|Δlogp| 1.4e-5..7.1e-5；Wan2.2 全参 + norm patch +
   `torch_sdpa` 做到 diff 恰好 0、ratio 恰好 1（200 rollout 持续）。CI 用确定性模式做逐位对比。

对 VRL 的落地顺序（先便宜后贵）：
- 立刻：SD3.5 batch-16 配方开 `precision_correction.recompute_old_logprob`（ppo_epochs=1 时第一窗口
  零成本），预期 pre-update clip 从 54–60% 归零；这同时解锁 compile。
- 其次：replay 批形状 = rollout 批形状（内存允许时 `training_microbatch_size` 对齐 16），已知等形状漂移为 0。
- 再次：Wan 家族对齐 norm/residual/RoPE 舍入位置，作为 B spike 引擎 provider 的 patch 组。
- 长期：frozen-weight 诊断模式 + `model_output_*` 指标 + 确定性模式下的逐位 CI 标准。

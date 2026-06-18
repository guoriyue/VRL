# slime vs cosmos-rl — async / GPU-topology / mini-batch study（index）

类型：reading / cross-system 源码对照研究（2026-06-17,7-agent workflow,file:line 全核过)。
源码：`~/Desktop/slime`(Megatron+SGLang)、`~/Desktop/cosmos-rl`(NVIDIA,TRT-LLM/vLLM)。
file:line 相对各自 repo root;我们的相对 `vrl/`。

这次研究回答三个问题,各拆一份 doc:

| Doc | 问题 |
|---|---|
| [slime_cosmos_async_reward.md](slime_cosmos_async_reward.md) | 怎么做 async rollout/train,**reward 在哪算** |
| [slime_cosmos_gpu_topology.md](slime_cosmos_gpu_topology.md) | 单卡 vs 多卡怎么**支持 + 区分** |
| [slime_cosmos_minibatch.md](slime_cosmos_minibatch.md) | mini-batch + **跨 rollout/train 的 mini-batch** |

配套:逐系统深读在 [slime.md](slime.md) / [cosmos-rl.md](cosmos-rl.md);async overlap 的设计裁决(DiffusionNFT-locked)在
`../parked/SPRINT_async_rollout_train_overlap.md`。

---

## 三句话总览

- **async**:slime = 1 步双缓冲 + reward 折进生成协程(asyncio,主要 CPU 规则);cosmos = controller/replica 多在飞 + reward 独立 dispatcher 池(为 GPU 视频 reward 设计)。
- **单卡/多卡**:两家都**单 flag、不自动探测、单卡退化串行**。slime `--colocate`,cosmos `mode`。
- **mini-batch**:slime 一次 rollout → `num_steps_per_rollout` 个**独立**优化步,无内层重放;cosmos 三层嵌套 `mu_iterations → optimize-chunk → mini-batch`,NFT 还多 loop `num_train_timesteps`。

## 对我们的 4 条 takeaway（汇总，细节见各 doc）

1. **continuous barrier 已对**:`vrl/rollouts/orchestration/continuous/schedule.py:115-119` 的 `pause→drain→sync→resume` = slime `train_async.py:69-73` 同 spirit;`StalenessPolicy`(`staleness.py:38-44`)= cosmos `allowed_outdated_steps`(`controller.py:273-275`)。
2. **单卡权重同步抄 cosmos 指针交换**:`set_underlying_model`(cosmos `colocated/rollout_control.py:68`),零拷贝零 NCCL,比我们 `CPU state_dict→ray.put→load`(`vrl/generation/ray/weight_sync.py:50-59`)便宜得多;且保持串行(两家都证明单卡不重叠)。
3. **Reward 抄 cosmos 解耦 dispatcher + `non_text→ThreadPoolExecutor`**(`reward/dispatcher.py`),适配 GPU 解码视频 reward;我们 reward-execution `inline|pool` 已能对上。
4. **mini-batch 已对上 NFT 范式**:我们 streaming `gradient_accumulation_steps = microbatches × timesteps` = cosmos NFT 的 per-optimize ×(mini_batch × timestep)。多卡时再 mirror `mu_iterations` + slime DP-MAX micro-batch 对齐。

## 一个诚实的反证据（值得回头看）

cosmos 的 `nft_trainer.py:374` **有 `mu_iterations`**(对冻结引用、randperm 打乱的内层重放)。我们前面以"DiffusionNFT 不该做 inner-replay"为由撤掉了 ppo_epochs——但 NVIDIA 自己的 NFT trainer **确实在用内层重放**。inner-replay ≠ async-overlap(cosmos 用前者、不用后者)。结论:**inner-replay 对 DiffusionNFT 也许没我当时说的那么不可取**;若以后再试,`nft_trainer.py:374-666` 是可对照的真实实现。详见 minibatch doc §4。

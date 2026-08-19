# SPRINT: 全仓 Triton 地图与性能账本 — 按负载类判定，不按直觉

**日期**: 2026-08-18  **状态**: PLANNED
**触发**: 用户问 "能给仓库加什么 Triton？vLLM 用了很多 Triton"，追问 thorough
estimation，并指出判定应覆盖全仓而非只看 anima。
**证据来源**: Run E 实测（`train_E_0818.log`，pid 1238430，config
`experiment/anima_preview3/online_grpo_animereward_nsfw_safety`，2026-08-18 15:04 起）+
`outputs/anima_animereward_nsfw_grpo/reward_debug/animereward_quality_results.jsonl`（2112 条）+
`outputs/anima_animereward_nsfw_grpo/resolved_config.yaml` +
`SPRINT_ppo_update_cycle_speedup.md` 的 cosmos V2W 实测（3.58s/前向）+ 仓内源码逐文件核对。
姊妹 sprint：`SPRINT_nvfp4_rollout.md`（in progress，P2 待验收）、
`SPRINT_ppo_update_cycle_speedup.md`（in progress，CFG-off 杠杆的出处）。

---

## 0. 一句话

Triton 的价值按**负载类**分，不是全仓一个答案：

- **扩散类（图像+视频）**：不值得新写任何 kernel——SDE/logprob 逐元素数学占端到端
  ≤0.03%（图像实测算术 §2.5，视频外推 §5.2）。收益全在两个**已写好但没开**的路径：
  compile pass（inductor 生成 Triton）和 NVFP4 量化 rollout（手写 Triton 已在仓）。
- **AR token 类（llamagen / janus_pro / emu3 / nextstep_1 / glm_image）**：唯一值得
  新写的 kernel——Liger 式 **fused linear + log-softmax + gather**——**已落地**
  （`vrl/nn/kernels/fused_linear_logprob.py`，janus replay 已接入）：janus 形状实测
  峰值显存 564 MiB → 34 MiB（**16.6×**），细节见 §5.3。
- **当前最大的墙钟杠杆根本不是 kernel**：Run E 每迭代 ~25 分钟里 82% 是训练+reward
  间隙，reward 服务每图排队 24.4s 才做 0.38s 推理（§1.2）。

顺序：**P0 打开已有 PhaseTimer 拆间隙 → P1 启用 compile pass → P2 启用 NVFP4 →
P3 修 reward 服务 → P4（独立轨）AR token FLCE**。P1/P2 的 seam
（`vrl/nn/optimization/passes.py`）是 family 无关的，落地即全仓生效。

## 1. 实测账本（图像扩散基准：Run E，2026-08-18）

负载：Anima preview3（Cosmos Predict2 DiT 2B + Qwen3-0.6B TE + Qwen-Image VAE），
1×RTX 5090 32.6GB（sm_120），512×512，20 步去噪，CFG 4.5（每步 2 次前向），
LoRA，16 prompts × 16 samples = 256 张/迭代，`timestep_fraction: 1.0`，bf16 eager
（resolved config 中 **torch_compile 与 quantization 均未启用**）。

### 1.1 周期结构（`train_E_0818.log` 时间戳）

一个迭代 = 4 个 mini-cycle，每个 mini-cycle：

| 环节 | 实测 |
|---|---|
| 4 批生成（16 样本/批） | 4 × 16.4–16.6s ≈ **66s**（`generation wall ... wall_s=16.376/16.455/16.520/16.578`，多个周期稳定复现） |
| cumem park（rollout worker 睡眠） | 生成结束后 ~31s 出现 `sleep freed 5.66 GiB` |
| 静默间隙（无任何日志行） | **~305s**（例：17:40:47 park → 17:45:52 下一批生成） |

每迭代合计：rollout 16 批 × 16.5s = **264s（18%）**；间隙 4 × ~305s = **~1220s（82%）**；
全迭代 **~24.7 min**。显存 peak 6596MB / budget 25030MB（denoise 6596，decode 6556，
baseline 5834）——瓶颈不在容量。

### 1.2 Reward 服务延迟（`animereward_quality_results.jsonl`，末 256 条）

```json
"timing_ms": {"http_roundtrip_ms": 25625, "inference_ms": 381,
              "service_inference_wall_ms": 24350,
              "service_artifact_validation_ms": 1271, "queue_wait_ms": 0.008}
```

- 每图 HTTP 往返 **~25.6s**，真实模型推理 **381ms**：**~24.4s 是服务端排队/串行**。
- 分量合计 mean ~52s/图（p50 51.8s，max 52.7s；分量重叠，不能当墙钟加总）。
- 256 图/迭代。即使客户端高并发，服务端近串行意味着 reward 墙钟以百秒计。

### 1.3 间隙内部拆分（推算，待 P0 实测证实/推翻）

- 训练侧（evaluate 重放 + backward）：重放样本-前向数与 rollout 相同
  （256 × 20 步 × 2 CFG 前向），forward-only ≈ ~264s 减 VAE 份额；backward 对被训
  pass 再加约 2× 前向 → 估 **evaluate+backward ≈ 400–500s/迭代**。
- 残差 **~700–800s/迭代** 无 GPU 训练可解释 → 最可能是 reward HTTP 等待 + weight sync。
- `trainer.profile` 默认 false（`vrl/trainers/online/config.py:234`），metrics.csv
  没有 advantage/evaluate/backward/optim_step 分段（`PhaseTimer`，
  `vrl/trainers/online/trainer.py:140`）。P0 补的就是这一个测量。

## 2. 扩散类候选的收益估算

基线：迭代 1484s（264 rollout + 1220 间隙）。"端到端%" 均对此基线。

### 2.1 P0 — `trainer.profile: true`（测量，先做）

收益 0，解锁一切。改一行 YAML；`_write_phase_events`（trainer.py:1921）落
per-phase 事件。**验收**：分段数据回填 §1.3 两个估算区间；若 evaluate+backward
实测落在 400–500s 外，§4 顺序重排。

### 2.2 P1 — 启用已有 torch.compile pass（全 family 生效）

- 现状：pass 在 `vrl/nn/optimization/passes.py:164`（`torch_compile_transformer`），
  anima_preview3 全部 preset 未开。seam 是 family 无关的。
- 估算：DiT forward 经验提速 10–25% → rollout 264s → ~210–240s（端到端 **2–4%**）；
  训练侧 forward 若也 compile（`vrl/trainers/fsdp.py:162` 的 unwrap 逻辑支持
  `compile(PeftModel(transformer))` 形态），再省 **~50–100s**。
- 风险：FSDP+LoRA graph break、CFG 双 batch 形状重编译、warmup 摊销。
- 工具：`vrl/scripts/perf/compile_benchmark.py`。
- **验收**：compile on/off 的 `generation wall` 差 ≥10%，replay parity 不劣化；
  否则记负结果关闭。

### 2.3 P2 — 启用已有 NVFP4 量化 rollout（手写 Triton 已在仓）

- 现状：kernel 已写好（`vrl/nn/quantization/fp4_kernels.py`），5090 原生 FP4；
  runtime 支持见 `SPRINT_nvfp4_rollout.md`（P1 落地，P2 真实验收未完成）。
- 估算：外部同契约报 ~2.5–3× rollout（H100）；本机 MLP-only 保守取 GEMM 1.5–2× →
  rollout 省 **~100s/迭代（端到端 ~7%）**。
- 风险：rollout/replay 漂移。修正与探针现成：TIS
  （`vrl/algorithms/logprob_mismatch.py` 的 `apply_truncated_importance_weight`）、
  `vrl/scripts/perf/quantized_rollout_drift_probe.py`。
- **验收**：drift probe 的 `logprob_abs_diff` / ratio 漂移在 TIS 可救范围
  （bf16 基线 `logprob_abs_diff_mean ≈ 2.5–2.9e-3`，metrics.csv epoch 3–5）；
  短 run 验 reward 曲线斜率不劣化。这同时是 NVFP4 sprint 欠的 P2 验收。

### 2.4 P3 — Reward 服务吞吐（非 Triton，可能是最大单项）

- 证据：381ms 推理排成 24.4s 墙钟（§1.2）→ 服务内部近串行。
- 估算：若 P0 证实 reward 份额 ≥ 500s/迭代，修服务端批处理/并发的上限收益
  **数百秒/迭代**——超过 P1+P2 之和。
- **验收**：修后 `service_inference_wall_ms` 降到与 `inference_ms` 同数量级 × 批深度。

### 2.5 手写 fused SDE logprob — 判定：图像扩散上**不值得**（~0.03%）

算术记录在案以便复查（对口头讨论第一轮结论的修正）：

- latent 每样本 16ch × 64 × 64 = **65,536 元素**；batch 16 ≈ 1.05M 元素。
- `sde_step_with_logprob`（`vrl/math/denoise/flow_matching.py`）每调用 ~25 个小
  kernel × ~10µs（launch 开销级）≈ **250µs/步**。
- 调用量：rollout 20 步 × 16 批 + replay 20 步 × 16 微批 ≈ 640 次 × 250µs ≈
  **0.16s/迭代**（宽松 0.2–0.5s）→ 端到端 **~0.03%**。
- 第一轮推荐它是因为"rollout+replay 两条热路径都命中"，没乘张量尺寸；乘上后翻转。
  视频负载的同类外推见 §5.2（结论同样成立）。

## 3. 复查清单（回填实测）

| 项 | 预期区间 | 实测（回填） |
|---|---|---|
| evaluate+backward /迭代 | 400–500s | |
| reward 等待份额 /迭代 | ~700–800s | |
| P1 compile rollout 提速 | 10–25% | |
| P2 NVFP4 rollout 提速 | 1.5–2× GEMM，~100s/迭代 | |
| P2 drift（logprob_abs_diff_mean） | ≤ TIS 可救；bf16 基线 2.5–2.9e-3 | |
| P3 修后 service_inference_wall_ms | ~inference_ms × 批深度 | |
| P4 FLCE 峰值显存节省（janus B16×L576） | ~1.2GB（§5.3） | B8×L576×V16384×D2048 bf16 实测 564→34 MiB（16.6×）；B16 约 2×该值 |

## 4. 执行顺序与门

Run E 正在跑（GPU 84%），P1/P2 的 GPU 探针须等其结束或用间隙窗口。

1. **P0（现在）**：下一 run 配置加 `trainer.profile: true`。零成本。
2. **P1（Run E 后）**：`compile_benchmark.py` 短探针 → 达标进 preset（全 family）。
3. **P2（与 P1 并行可跑探针）**：drift probe 定漂移 → TIS 范围内则短 run 验曲线。
4. **门**：P0 分段若显示 reward 份额 ≥ 500s/迭代 → P3 升为最高优先级。
5. **P4（独立轨）**：已完成（2026-08-18），验收记录见 §5.3。

## 5. 全仓 Triton 地图（按负载类）

仓内 21 个 family（`vrl/models/families/`）分三类，Triton 判定按类给：

### 5.1 图像扩散（anima/sd3_5/sana/flux/pixart_sigma/lumina2/qwen_image/hunyuan_image/glm_image 扩散侧）

即 §1–§2 的实测基准。判定：**不新写 kernel**；P1 compile + P2 NVFP4 即全部收益。

### 5.2 视频扩散（wan_2_1/cosmos V2W/mochi/hunyuan_video/cogvideox/causvid/magi_1/echo）

SDE 逐元素数学的账在视频上重算一遍，结论不变：

- wan 480p×93f latent ≈ 16ch × 24t × 60 × 104 ≈ **2.4M 元素/样本**（anima 的 37×），
  25 个 kernel 从 launch-bound 变 bandwidth-bound：~1GB 读写 / ~1.8TB/s ≈
  **~0.5–1ms/步**。
- 但分母同步变大：同负载实测 **3.58s/前向**（`SPRINT_ppo_update_cycle_speedup.md`
  §1，cosmos V2W 93f）→ SDE 占比 **~0.02%**。分子分母都随 latent 尺寸涨，比值不变，
  "图像上不值得"的判定对视频稳健。
- 视频类的真实杠杆在姊妹 sprint：CFG-off（rollout+PPO 同时砍半）、多卡结构解、
  VAE decode 占比实测（`SPRINT_fp4_off_policy_reward_vae.md` 挂起等这个数）。
  VAE GroupNorm+SiLU 融合在 93f 尺度可能翻案，但先等 decode 占比实测，不预支。

### 5.3 AR token（llamagen/janus_pro/emu3/nextstep_1/glm_image AR 侧）— 唯一的新 kernel 候选

vLLM 的 Triton 资产在这类**真的能用**（paged attention 已借入：
`vrl/nn/kernels/attention/vllm_paged.py`）。新候选是 Liger 式
**FLCE（fused lm_head linear + log-softmax + gather）**：

- 现状：replay 物化完整 logits——`token_logprob.py:103`
  `logits = result.require_value("logits")  # [B, L, V_img]`。
  下游 `gather_categorical_log_probs`（`vrl/math/token/logprob.py:43`）**已经分块**，
  log_softmax 中间张量已不是问题（其 docstring 记录了 colocation OOM 动机）；
  剩下的是 **logits 本体 + 其 backward 梯度**仍完整物化。
- 量化（janus_pro：`JANUS_IMAGE_VOCAB_SIZE = 16_384`，`model.py:74`；llamagen
  `vocab_size = 16384`，`vendor/gpt.py:93`）：B16 × L576 × V16384 × fp32 ≈
  **604MB logits + 604MB grad ≈ 1.2GB 峰值**；L1024 时 ~2.1GB。
- 性质：**显存杠杆**（解锁 trainer+rollout 同卡 colocate 下更大的 replay 微批），
  不是速度杠杆。FLCE 按 L 分块算部分 logits、立即规约、backward 重算，logits 永不
  完整存在。带手写 backward，是 §口头讨论里"Triton 练手项目"的正确标的
  （取代已被 §2.5 否决的 fused SDE logprob）。
- 先例：Liger Kernel 的 fused_linear_cross_entropy（对标对象，vLLM kernel 无
  backward 不适用）。
- **落地（2026-08-18）**：`vrl/nn/kernels/fused_linear_logprob.py`（Triton 前向
  online-logsumexp+gather、反向 in-place grad_z，GEMM 留 cuBLAS、反向重算换显存，
  CPU/fp64 走等价 torch 回退）；payload 契约新增 fused 形态
  （`ReplaySegmentResult.logprobs` 的 `head_hidden`/`head_weight`/`head_bias`），
  janus replay（含 R1 visual 段）已切换，`gen_head` 结构不匹配时自动回退 eager。
- **契约化（同日）**：能力提升为 family 级契约——`ARModelBase.vocab_head_split()`
  默认 None（eager 回退），`vrl/models/steps/token/vocab_head.py` 定义
  `VocabHeadSplit`（prefix + 最终投影的 weight/bias）与 payload 构造器。两个采用者：
  janus（MLP 头拆分）、glm_image（lm_head 行切片即 codebook 限制，view 零拷贝）。
  两个记录在案的不采用者：emu3（logits 上有 per-position 结构 mask，kernel 需先支持
  mask 输入）、llamagen（投影在 vendor trunk forward 内部）。LoRA 陷阱已封死：
  探测用精确类型判断（PEFT lora.Linear 可能继承 nn.Linear，`.weight` 会静默丢
  adapter 增量），包装过的头一律回退 eager。
- **验收结果**：①fp64 `gradcheck` 通过；CUDA 上 fwd/bwd 与 eager autograd 对照
  fp32 ≤2e-5 / ≤2e-4。②janus 形状（B8×L576×V16384×D2048 bf16 含 backward）峰值
  显存 564 MiB → 34 MiB（**16.6×**）。③kernel 对同一 logits 与 torch 参考一致到
  ~4e-6；bf16 下 chunked GEMM 与一次性 linear 的 logits 本身差几个 ulp（cuBLAS
  tiling，与改微批大小同类，逐位一致在 bf16 下不可达），测试改为对 fp64 真值
  断言 fused 误差 ≤ eager 误差——已通过。测试：
  `tests/nn/kernels/test_fused_linear_logprob.py`（11）+ janus/replay 套件（61）。

### 5.4 否决项汇总（记录以免重提）

| 候选 | 判定 | 理由 |
|---|---|---|
| 手写 attention kernel | 否 | SDPA/flash + 已借 vLLM paged 覆盖全部三类 |
| GRPO loss kernel | 否 | per-sample 标量，尺寸不构成问题 |
| fused SDE logprob | 否（性能） | §2.5 图像 + §5.2 视频算术；练手价值由 §5.3 FLCE 取代 |
| VAE GroupNorm+SiLU 融合 | 缓议 | 图像 <1%；视频等 decode 占比实测再判 |
| vLLM fused RoPE/RMSNorm/silu_and_mul 移植到扩散侧 | 否 | 推理 kernel 无 backward；训练侧先由 P1 compile 拿免费收益 |
| CFG off（guidance=0.0） | 转交 | 属 `SPRINT_ppo_update_cycle_speedup.md` 质量门；anima 的 CFG 4.5 是实测调优值 |

## 非目标

- 不为扩散类性能新写 Triton kernel（翻案需先推翻 §2.5 / §5.2 的算术）。
- 不动 CFG（姊妹 sprint 的质量门）。
- 不做多卡结构改造（`SPRINT_ppo_update_cycle_speedup.md` 已论证）。
- P4 FLCE 不进 P0–P3 的排期，不与当前 anima 实验争 GPU。

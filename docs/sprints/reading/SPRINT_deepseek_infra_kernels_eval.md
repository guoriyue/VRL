# SPRINT: DeepSeek 开源基础设施 / kernel 评估 —— PARKED（硬件/领域双 gating）

状态：**parked（2026-06-27）**。这是一次**评估归档**:把 `github.com/deepseek-ai` 下的 infra/kernel repo 逐个对本仓(单卡 Blackwell sm_120 / RTX 5090 / 32GB、dense diffusion+AR、Triton)做了适配性判定,结论是**当前不可用**,记下来防止以后重复调研。

## 0. 结论：infra/kernel 这批,现在都用不上

两条 gating 把整批 infra/kernel repo 挡在外面:
1. **硬件**：DeepSeek 的 fp8/attention kernel 只编 **sm_90(Hopper)/ sm_100(B200 数据中心 Blackwell)**,**没有 sm_120**(消费级 Blackwell,即 5090)。
2. **领域**：一大半是 **MoE-only** 工具,本仓是 dense 模型,用不上。

## 1. 证据（已 verify build 配置）

| repo | 判定 | gating 证据 |
|---|---|---|
| `DeepGEMM`(fp8 GEMM) | LOW | README 只讲 SM90/SM100 scaling 格式;无 sm_120 |
| `FlashMLA`(MLA attn) | LOW | `setup.py:43-45` 只 emit `sm_100f` 与 `sm_90a`;且 MLA 是 DeepSeek 专用注意力,非标准 DiT/joint-attn |
| `TileKernels` | LOW | README 明写 "SM90 or SM100";重 MoE(gating/routing) |
| `DeepEP`/`EPLB`/`LPLB`/`ESFT`/`Engram` | NONE | MoE-only(expert dispatch / load-balance / expert FT / memory)——dense 模型无 expert 路由 |
| `3FS`/`smallpond` | NONE | 180+ 节点 RDMA 分布式存储 / PB 级数据;单机正交 |
| `DualPipe`/`profile-data`/`open-infra-index` | MEDIUM(仅参考) | 算法/overlap 思路 + 真实 trace,但都是 Hopper 多机 MoE 尺度,非单卡可用 |
| `DeepSpec`(spec decode) | MEDIUM | draft+verify 思路;但本仓 **speculative-diffusion 已 NEGATIVE**(whole-latent coupling 高维墙),且 diffusion rollout 已 ~94% MFU,加速空间存疑 |
| `DeepSeek-V3.2-Exp`(DSA 稀疏注意力) | MEDIUM | DSA 是 dense 模型效率技术,但 rollout 序列短、收益不明,kernel 绑 sm_100 |

跨 `DeepGEMM/FlashMLA/TileKernels/DeepEP` 全文 grep `sm_120 / compute_120 / 5090 / RTX 50`：**0 命中**。

## 2. 与本仓既有结论的连接

- fp8 是**双重 gating**:既被硬件挡(sm_120),又被自家结论挡——fp8 lossy、走 off-policy 路径、污染 `old_log_prob`(verl 规则)。见 `SPRINT_lossless_diffusion_rl_research`。
- 真正缺的 attention lever 是 **FA-3 for sm_120**(flash-attn 未装),这批 repo 给不了(FlashMLA 是 MLA + sm_100)。

## 3. 解 park 的条件（什么时候再看）

- 换上 **sm_100 数据中心卡(B200)** 或 DeepSeek 放出 **sm_120** 编译目标 → 重评 DeepGEMM/FlashMLA 的 fp8 路径。
- 本仓引入 **MoE 模型** → 重评 DeepEP/EPLB/ESFT。
- 走**多机训练** → 重评 DualPipe / 3FS。

## 4. 非目标

- 不为 sm_120 自己 fork/port DeepGEMM/FlashMLA(工程量大、收益被 off-policy 正确性进一步压低)。
- 不引入 MoE 工具到 dense 栈。

> 注:与 infra/kernel 不同,DeepSeek 的**视觉模型 / reward / RL 方法论** repo 是有用的,已另开 3 个 planned sprint:`SPRINT_ocr2_reward_backend`、`SPRINT_janus_pro_upstream_reconcile`、`SPRINT_vl2_grounding_judge_reward`;GRPO 方法论参考 `DeepSeek-Math`(GRPO 出处)/`DeepSeek-R1`/`DeepSeek-V3`(fp8 训练配方 + MTP)。

# SPRINT: Config correctness audit —— 剔除跑不出真结果的设置

状态：**DONE cosmos / wan 待验（2026-07-02）。** 性质：**正确性审计（不是重复审计）。**

> ## ✅ 执行结果（2026-07-02，改在 `~/Desktop/VRL`，未 commit）—— `pytest tests/config/test_load_all_experiments.py tests/ray/test_resources.py::...async_reward... = 31 passed`
> - **240p cosmos（5）**：删 3 死配置（`droid_full_target_240p`、`fullparam_8bit_240p`、`v2w_reference_fullparam_240p`）；**转 480p 2 个**（原计划全删,但 pre-flight 发现两个被测试硬编码 → 改为转 480p 保测试）：`async_reward`（就地 240p→480p + native ckpt）、`droid_target_240p`→**改名** `droid_target_480p`（+ 更新 `tests/config/test_load_all_experiments.py:496`）。
> - **480p checkpoint 路径**：两处绝对路径 `/home/.../vrl2/VRL/checkpoints/cosmos_predict2_2b_v2w_480p_diffusers` → 统一相对 `checkpoints/cosmos-predict2-2b-480p-16fps-diffusers`（实测两份权重同架构、bf16 3.7G = fp32 7.3G 的一半 = 同权重不同精度；用已验证生成干净的 bf16 版）。转的 2 个 240p 配置也加了这个 native ckpt path（否则 480p 用 720p 权重仍扭曲）。
> - **ppo=1（2 cosmos）**：`v2w_reference`、`kling_video_reward` → `ppo_epochs: 4`。
> - **wan 3 个 240p 配置：实测后 KEEP（不删）**——240p 生成连贯的猫（非垃圾）,Wan 优雅降级 ≠ cosmos 崩噪声,详见 §1 wan 小节。evidence-first 赢了:没删无证之物,证据反而洗清了它们。
> - **⚠️ 未做（诚实）**：① `async_reward` header 还有若干 240p-era 文字（memory-smoke 指令等)没逐句改,你重组时可顺手清；② 改动都在 `~/Desktop/VRL`,与你未提交的 dedup 编辑叠加,**未 commit**——你自己 reconcile。 查的是「这个配置能不能跑出可信结果」，而不是「有没有重复」。与 [[SPRINT_config_duplication_audit]] 互补：那份治**重复/factoring**，这份治**坏设置**——两者有一处正面撞车（见 §3）。发现 3 类不合理设置：① 240p 视频尺寸=生成垃圾；② 480p checkpoint 路径混乱（加载失败/不可移植/两份不同权重）；③ `ppo_epochs=1`=平曲线根因。

> 证据（本 sprint 实测，2026-07-01，标准 diffusers `Cosmos2VideoToWorldPipeline`，DROID 参考帧）：
> - 分辨率 sweep（frame-drift = mid 帧对 anchor 帧的平均像素差，越大越发散）：**240p drift 0.336（per-frame std 0.274→0.349 递增=发散成彩虹噪声）**；720p 权重@480p drift 0.122（勉强连贯但扭曲）；**480p-native 权重@480p drift 0.064（干净，接近原生）**；704p 原生 drift 0.085（金标）。
> - 模型原生 shape = **1280×704×93**（`configs/model/diffusion/cosmos/predict2_2b.yaml` 注释 "native 1280x704x93f"）；240p=~11% 像素、35% 帧，深度 OOD。
> - **没有 240p-native 权重**：cosmos 仓库只发 720p（`transformer/` 默认）+ 480p-native（`model-480p-16fps.pt` 需转换）两套；240p 对两套都 OOD。
> - `online_grpo_droid_full_target_480p.yaml` 自己的 header 也记：720p ckpt 在 240p/480p 出 garbage；baseline eval reward **0.4487（480p）vs 0.2587（240p garbage）**。
> 相关：[[project_first_trustworthy_curve]]（"LoRA 梯度太小"是在 240p garbage 上测的,存疑）、[[project_cosmos_v2w_fullparam_trains_confirmed]]（240p garbage 先前已记）、[[project_flux_algo_validation]]（`ppo_epochs=1` 让 trust-region clip 恒等 0 = 平曲线根因,fix=4）、[[SPRINT_config_duplication_audit]]（dedup,§3 撞车）。

## 0. 一句话 + 范围

「设置不合理」= 就算跑通也拿不到可信结果的设置。三类：**240p 生成垃圾（reward 打噪声分,永远不学）**、**480p checkpoint 路径三重混乱（加载失败/写死绝对路径/两份不同权重）**、**`ppo_epochs=1` 让 GRPO 的 clip 空转→曲线不动**。前两类是 cosmos 专属,第三类跨家族。

## 1. 240p 视频尺寸 = 生成垃圾（8 个配置）

240p（416×240）纯为省显存选,任何视频扩散模型都没在这个分辨率训过 → 分布外 → 生成发散成彩虹噪声（frame0=参考锚点保持干净,frame1..N 是生成的→垃圾）→ reward 打的是噪声分 → **永远出不了真曲线**。这是 shape 问题,**换 checkpoint、调 step/cfg、加显存都救不了**。

**Cosmos（5 个,已实证 garbage,`sampling/video/240p_33f`)：**
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_async_reward.yaml`
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_droid_full_target_240p.yaml`
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_droid_target_240p.yaml`
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_fullparam_8bit_240p.yaml`
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference_fullparam_240p.yaml`

**Wan（3 个）——❌ 嫌疑被推翻,实测 240p 是「连贯的」不是垃圾,KEEP：**
- `configs/experiment/diffusion/wan_2_1/online_grpo_kling_video_reward.yaml`
- `configs/experiment/diffusion/wan_2_1/online_grpo_ocr.yaml`
- `configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml`
> 实测（2026-07-02,标准 diffusers `WanPipeline`,T2V-1.3B,"a cat walking"）：**240p 生成一只清晰可辨的猫在花园里走**（风格化、细节少、`spatial_hf=0.071` 偏高、`temporal_diff=0.006` 动作近乎冻结）——**连贯,不是彩虹噪声**。480p 照片级清晰（hf 0.041、motion 0.036）、native 832×480×81 最平滑。**Wan 1.3B 在低分辨率上优雅降级,cosmos 则崩成噪声**——两个模型对 OOD 分辨率的鲁棒性不同,不能一概而论。**结论:wan 240p 配置保留,不删。** 唯一小提醒:240p 动作近乎冻结,`physics`/`motion` 类 reward 可能想上 480p 换更好的运动信号——但这是质量取舍,不是正确性 bug。

**动作**：Cosmos 5 个删或隔离（240p 不该是"正常配置"）。Wan 3 个先用同样的眼看-mp4 方法验一眼再删。**例外**：若某个 240p 配置是**纯 perf smoke**（小 shape 跑得快、只测 timing,如 `fullparam_8bit_240p` 曾用于 sbs=4 计时 [[project_p1_sbs_confirmed]]),可保留但**必须在 header 显式标注 "PERF-ONLY: generates garbage, NOT for real curves"**,否则会被误当真曲线配置。真曲线一律走 480p。

## 2. 480p checkpoint 路径三重混乱（3 个配置）

- **相对路径在本 clone 缺失**：`online_grpo_droid_full_target_480p_lora.yaml` 用相对 `checkpoints/cosmos-predict2-2b-480p-16fps-diffusers`,**在 `~/Desktop/VRL` 里不存在**（只在 `~/Desktop/vrl2/VRL`,3.9GB bf16）。从 VRL clone 跑 → `from_pretrained` 加载失败。
- **写死绝对路径（不可移植）**：`online_grpo_droid_full_target_480p.yaml` 和 `online_grpo_v2w_reference_480p.yaml` 写死 `/home/mingfeiguo/Desktop/vrl2/VRL/checkpoints/cosmos_predict2_2b_v2w_480p_diffusers` —— 换机器/换目录/换用户即断,且从 VRL 配置指向 vrl2 clone,跨 clone 依赖。
- **两份不同的 480p checkpoint**：`cosmos-predict2-2b-480p-16fps-diffusers`（3.9GB,bf16）vs `cosmos_predict2_2b_v2w_480p_diffusers`（7.8GB,fp32）—— 名字不同、精度不同、没有单一 canonical 版本。到底哪个是对的？

**动作**：① 定**一个** canonical 480p checkpoint（我实测 3.9GB bf16 版生成干净,drift 0.064）；② 全部改**相对路径或 `${VRL_DATA_ROOT}` 环境变量**,禁绝 `/home/...` 绝对路径；③ 确保它在实际要跑的 clone 里存在（或写进 setup 文档的下载/转换步骤）。

## 3. `ppo_epochs=1` = 平曲线根因（11 个实验）—— 与 dedup 审计正面撞车

`ppo_epochs=1` 时 importance ratio 在唯一一遍里恒等 1.0 → **trust-region clip 是恒等 0 的空操作** → GRPO 退化成单步 REINFORCE,这是 [[project_flux_algo_validation]] 实证的"曲线不动"根因（fix=`ppo_epochs=4`,让 `clip_fraction>0`）。11 个实验用了 ppo=1,含 3 个 cosmos：`online_grpo_kling_video_reward.yaml`、`online_grpo_v2w_reference.yaml`、`online_grpo_v2w_reference_480p.yaml`。

**⚠️ 撞车**：[[SPRINT_config_duplication_audit]] 把 `ppo_epochs: 1`（数到重复 22 次）列为"family-common tuning → 提到 recipe 做默认"。**绝不能这么做**——那等于把平曲线 bug 焊死成所有实验默认。dedup 的正确做法在这里反了：重复 ≠ 应当固化。

**动作**：想看到学习的配置一律 `ppo_epochs=4`（你那几个能跑的 cosmos 配置正好都是 4）。若要 factor,**recipe 默认设 4**,只有确实要纯 on-policy REINFORCE 的实验才显式覆盖成 1（并注明理由）。绝不把 1 当默认。

## 4. 动作汇总

| 项 | 配置数 | 问题 | 动作 |
|---|---|---|---|
| 240p cosmos | 5 | 生成垃圾(实证) | 删；至多 1 个标 `PERF-ONLY` 保留 |
| 240p wan | 3 | 生成垃圾(嫌疑) | 眼看验证 → 删 |
| 480p 相对路径缺失 | 1 | 本 clone 加载失败 | 统一 checkpoint + 相对/env 路径 |
| 480p 绝对路径 | 2 | 不可移植/跨 clone | 同上,禁 `/home/...` |
| 两份 480p 权重 | — | 无 canonical | 定一个(3.9GB bf16 已验证干净) |
| ppo=1 | 11(3 cosmos) | 平曲线 | 真曲线配置改 4；recipe 默认 4 不是 1 |

## 5. 验证与 Non-goals

- **验证**：改完后 `vrl-train --cfg-only`（或等价的 config-resolve dry-run）确认每个保留的实验能解析 + checkpoint 路径存在；对保留的视频配置各跑一个 base-gen 眼看 mp4（[[SPRINT_config_duplication_audit]] 的 dry-run 同源）。
- **Non-goal**：不碰 tier 结构（那是 [[SPRINT_config_duplication_audit]] 的事,且它的结论是"更 factor 而非 flatten"——同意,不冲突）。不改 480p/704p 这些**能跑**的配置的算法超参(除 ppo=1)。不删 perf-profiling sprint 文档([[feedback_keep_perf_sprints]])。
- **顺序建议**：先做本 sprint（删坏配置、修路径、ppo=1）,**再**做 dedup factoring——否则会把坏设置一起 factor 进 recipe（见 §3 撞车）。
